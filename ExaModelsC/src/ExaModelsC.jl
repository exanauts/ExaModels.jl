"""
    ExaModelsC

Compile an [`ExaModels.ExaCore`](@ref) into a self-contained shared library that
exposes the model through the plain C interface consumed by
[CNLPModels.jl](https://github.com/MadNLP/CNLPModels.jl) (and its Python twin
`cnlpmodels`).

The core is the compile-time artifact.  It is built once against
[`ExaModels.ArgSource`](@ref) placeholders — sizes, starting points and bounds left
open — and the compiled library resolves them per instance:

```julia
using ExaModels, ExaModelsC

c, N = ExaCore(nargs = Val(1))
@add_var(c, x, N; start = 1.0)
@add_obj(c, (x[i] - 1)^2 for i in 1:N)

compile_library("/opt/models/rosen", c, 10)
```

`arg` here is an *example* value for the placeholder: it is never baked in, but
its type is.  `juliac --trim=safe` needs the whole call graph resolved
statically, so the example fixes what `N` *is* (an `Int`) while leaving what it
*equals* to `rosen_new(n)` at runtime.

## Consuming the result

The C interface is the one implemented by two companion packages, neither of
which is part of ExaModels:

  - [CNLPModels.jl](https://github.com/MadNLP/CNLPModels.jl) — loads the library
    as an `NLPModels.AbstractNLPModel`, so any JuliaSmoothOptimizers-compatible
    solver can solve it.
  - [cnlpmodels](https://github.com/MadNLP/cnlpmodels-py) — the same consumer
    for Python, over ctypes and numpy, needing no Julia runtime on the caller's
    side.

```julia
using CNLPModels, NLPModelsIpopt
lib = CNLPModels.load("/opt/models/rosen/lib/librosen.so")
ipopt(CNLPModel(lib, 1000; prefix = "rosen"))
```

```python
import cnlpmodels
lib = cnlpmodels.load("/opt/models/rosen/lib/librosen.so")
m = cnlpmodels.CModel(lib, 1000, prefix="rosen")
```

Both consumers default `prefix` to `"rec"` when handed a library handle, while
`compile_library` defaults it to the output directory's name — pass it
explicitly unless those coincide.  The test suite exercises both.
"""
module ExaModelsC

import ExaModels
import JuliaC
import Random
import Serialization
import TOML

export compile_library

# The generated app package is a throwaway: a fresh temporary directory with
# its own environment, never registered.  A constant UUID is therefore safe and
# saves a UUIDs dependency.
const _GEN_UUID = "b41c7e02-9f3d-4a58-8e6c-2d0f5a7c9b13"

"""
    compile_library(out, core, args...; prefix = basename(out), trim = "safe",
                    bundle = true, verbose = false)
        -> (; libpath, outdir, prefix)

Compile `core` into a shared library under `out`, and return the path to it.

## `bundle`

`bundle = true` (the default) emits a directory carrying the library together
with a **privatized** copy of the Julia runtime — around 80 MB, needing no Julia
on the consumer's side.  `bundle = false` emits a single ~2 MB library linked
against the Julia installation it was built with.

| consumer | `bundle = true` | `bundle = false` |
|:---------|:----------------|:-----------------|
| Python (`cnlpmodels`), C | works | works |
| **Julia (`CNLPModels.jl`)** | works | **aborts on the first call** |

!!! warning "Windows"
    `juliac` implements runtime privatization for Linux and macOS only — on
    Windows `privatize_libjulia!` warns "not implemented for this OS" and does
    nothing. A library compiled there therefore keeps the standard `libjulia`
    soname and **cannot be loaded back into Julia**, whatever `bundle` is set
    to; it is still usable from Python and C, where the calling thread is
    genuinely foreign.

Privatizing is the part that matters, not bundling.  A
`juliac` library links `libjulia`; `--trim` reduces the compiled *code*, not the
runtime.  Loaded into a process that is already Julia, a library sharing the
host's `libjulia` aborts on its first call: the `@ccallable` preamble adopts the
calling thread, and `jl_init_threadtls` guards that with
`if (jl_get_pgcstack() != NULL) abort();`.  `jl_get_pgcstack` reads *that
runtime's* thread-local storage, so a privatized runtime — a distinct one, whose
`pgcstack` is `NULL` — adopts legitimately, while a shared one aborts.  Linking
against the installed Julia instead, or bundling without privatizing, both
reproduce the abort; measured, not assumed.

Fixing that properly means skipping the adoption when the thread is already a
Julia thread, which is upstream in juliac's generated preamble.  Until then the
privatized bundle is the only form `CNLPModels.jl` can load, which is why it is
the default; pass `bundle = false` for a Python or C caller that would rather
have 2 MB than 80.

## Where it goes

`out` is `"@name"` to install on the CNLPModels search path
(`CNLPMODELS_PATH`), where both consumers find it by that name — or any
other string as a local path, exactly as written (`"rosen"` is `./rosen`,
`"/path/to/rosen"` is the full path). The same convention the consumers
apply to their string spec:

```julia
compile_library("@rosenrock", core, 1000)       # → \$CNLPMODELS_PATH/rosenrock/
CNLPModel("@rosenrock", 1000)                   # finds it, prefix defaults to the name
```

The example values are given exactly as they would be to `ExaModel(core, ...)`,
which is the point: compiling and instantiating a recipe should not be spelled
two different ways.  A recipe `core` must be built against a single
[`ExaModels.ArgSource`](@ref) — `ExaCore(nargs = Val(1))` — since the scalar ABI
carries one instantiation argument.  The example is used twice — to pin the
types `juliac` needs in order to trim, and to check that the core actually
instantiates before spending minutes compiling it.

**No example values means a fixed model.**  A core built with no placeholders
is a complete model, so `compile_library(out, core)` compiles it as-is:
`P_new` keeps its one-integer C signature but ignores the integer, and
`P_nargs()` reports `0` so a consumer handed only the library path knows that
no instantiation data is required.  A core that *declared* placeholders
(`nargs = Val(N)`, `N > 0`) is refused without examples.

The library exports, for `prefix` `P`: `P_nargs() -> 0 or 1` (how many
instantiation arguments `P_new` consumes), `P_new(n) -> id` (a positive
instance id; any number of instances may coexist), then id-first `P_nvar`,
`P_ncon`, `P_nnzj`, `P_nnzh`, `P_meta`, `P_obj`, `P_grad`, `P_cons`,
`P_jac_structure`, `P_jac`, `P_hess_structure`, `P_hess`.  Indices are 1-based;
the Hessian is the lower triangle of `obj_weight * ∇²f + Σᵢ yᵢ ∇²cᵢ`; every
function returns a `Cint` status, `0` on success, and none of them throws
across the boundary.

`P_new` takes a single integer, so the recipe form applies when the example
`arg` is an `Integer`, or a `NamedTuple` holding exactly one integer field —
the "scalable model" case (`rosenbrock` at size `N`).  Structured
instantiation (the schema + builder ABI, for data-defined models such as OPF)
is not built yet; `compile_library` says so rather than emitting a library
that would fail at load.
"""
function compile_library(
    out::AbstractString,
    core::ExaModels.ExaCore,
    args...;
    prefix::AbstractString = _default_out_prefix(out),
    trim::AbstractString = "safe",
    bundle::Bool = true,
    verbose::Bool = false,
)
    _check_prefix(prefix)
    core = _concretize(core)
    if isempty(args)
        # No examples means a FIXED model: the core has no placeholders and is
        # compiled as-is. A core that declared placeholders cannot be meant —
        # refuse it here with the arity it stated rather than letting the
        # instantiation probe below fail on a bare recipe. (An UNDECLARED
        # recipe — a bare `ArgSource()` written into a plain `ExaCore()` —
        # still reaches the probe, which rejects it just as loudly.)
        core.nargs isa Val{0} || throw(
            ArgumentError(
                "this core declared $(_nargs_count(core.nargs)) placeholder(s) " *
                "— give one example value per placeholder, as you would to " *
                "`ExaModel(core, ...)`; its types are what the compiler needs.",
            ),
        )
        arg = nothing
        field = FixedModel()
    else
        fields = _schema(args)
        _is_scalar_new(fields) || throw(
            ArgumentError(
                "this recipe needs the schema + builder interface, which is not " *
                "emitted yet — only a single integer placeholder (`<prefix>_new(n)`) " *
                "is. Its schema would be: " * _schema_json(fields),
            ),
        )
        arg = only(args)
        field = nothing
    end
    out = _resolve_out(out, bundle)

    # Instantiate here, in this process, before generating anything.  A core
    # that cannot be instantiated produces a library that cannot be loaded, and
    # the failure is far cheaper to read now than after a juliac run.
    probe = ExaModels.ExaModel(core, arg)
    verbose && @info "compile_library: core instantiates" nvar = probe.meta.nvar ncon =
        probe.meta.ncon prefix

    appdir = _generate_app(core, arg, field, prefix)
    verbose && @info "compile_library: generated app" appdir

    return _drive_juliac(appdir, prefix, out, trim, bundle, verbose)
end


# A core built in the default (non-concrete) mode accumulates its blocks in
# `Vector{Any}` rather than in its own type. That is what makes model
# construction cheap, and it is also what `juliac --trim=safe` cannot digest:
# the model type is not statically known, so the call graph will not resolve.
#
# It does not have to be compiled in that form, though. The core travels into
# the generated app as serialized DATA, and the blocks inside those vectors are
# already concretely typed — only the container is erased. Rebuilding it with
# tuple storage happens once here in the caller's process, and hands the
# generator exactly the artifact it gets from a `Val(true)` core. So a user
# can build the model in the default mode and still compile it. The rebuild
# itself lives in ExaModels (`_concretize`), where the `ExaModel` entry points
# use it for the same purpose.
_concretize(core::ExaModels.ExaCore) = ExaModels._concretize(core)

# ── Where the library goes ────────────────────────────────────────────────────

# `@name` installs on the CNLPModels search path, so
# `compile_library("@rosenrock", core, ...)` lands where
# `CNLPModel("@rosenrock")` and `cnlpmodels.CModel("@rosenrock")` will look
# for it; any other string is a local path exactly as written — the same
# convention the consumers apply to their string spec.
#
# Both layouts are ones the consumers already try: a bundle lands at
# `<dir>/<name>/lib/lib<name>.<ext>`, a single file at `<dir>/lib<name>.<ext>`.
function _resolve_out(out::AbstractString, bundle::Bool)
    startswith(out, "@") || return abspath(out)
    name = String(out[2:end])
    isempty(name) && throw(ArgumentError("`@` names a library — give one, like `@rosenrock`"))
    dirs = filter(!isempty, split(get(ENV, "CNLPMODELS_PATH", ""), ':'))
    isempty(dirs) && throw(
        ArgumentError(
            "`$out` names a library to install on the CNLPModels search path — " *
            "but CNLPMODELS_PATH is not set. Set it, or pass a path such as " *
            "`\"./$name\"`.",
        ),
    )
    # A bundle gets a directory of its own — `<dir>/<name>/lib/lib<name>.<ext>`,
    # the consumers' second layout.  A single file goes straight into the
    # directory as `<dir>/lib<name>.<ext>`, their first.
    return bundle ? joinpath(first(dirs), name) : first(dirs)
end

# The default symbol prefix: the name for `@name`, the directory's own name
# for a path. (`basename(abspath())` of an `@name` would keep the sigil,
# which is not a C identifier.)
_default_out_prefix(out::AbstractString) =
    startswith(out, "@") ? String(out[2:end]) : basename(abspath(out))

# ── Reading the example arguments ─────────────────────────────────────────────
#
# The schema is derived from the example values' TYPES, one field per
# placeholder.  Placeholders are positional, so the fields are named `arg1`,
# `arg2`, ... — and a consumer binds its own arguments positionally against
# that field order, `CNLPModel(lib, arg1, arg2, ...)`, the same spelling the
# example values are given in here.

struct Field
    name::String
    kind::String                            # "scalar" | "array" | "table"
    type::String                            # "f64" | "i64" — scalars and arrays
    columns::Vector{Pair{String,String}}    # tables only: name => type
end

_ctype(::Type{<:Integer}) = "i64"
_ctype(::Type{<:AbstractFloat}) = "f64"
_ctype(::Type{T}) where {T} = throw(
    ArgumentError(
        "an argument of type $T cannot cross the C boundary; the interface " *
        "carries 64-bit integers and floats, as scalars, arrays, or tables of " *
        "them.",
    ),
)

function _field(name::AbstractString, v)
    v isa Union{Integer,AbstractFloat} &&
        return Field(name, "scalar", _ctype(typeof(v)), Pair{String,String}[])
    if v isa AbstractVector
        isempty(v) && throw(
            ArgumentError(
                "the example for `$name` is empty, so there is nothing to read " *
                "its element type from — give one with at least one element.",
            ),
        )
        el = first(v)
        el isa NamedTuple && return Field(
            name, "table", "",
            [String(k) => _ctype(typeof(getfield(el, k))) for k in keys(el)],
        )
        return Field(name, "array", _ctype(typeof(el)), Pair{String,String}[])
    end
    throw(
        ArgumentError(
            "an argument of type $(typeof(v)) cannot cross the C boundary; give " *
            "a number, a vector of numbers, or a vector of NamedTuples (a table).",
        ),
    )
end

_schema(args) = [_field("arg$i", a) for (i, a) in enumerate(args)]

# The published schema, in the shape both consumers parse.
function _schema_json(fields::Vector{Field})
    parts = map(fields) do f
        f.kind == "table" ?
        """{"name":"$(f.name)","kind":"table","columns":[""" *
        join(["""{"name":"$(c.first)","type":"$(c.second)"}""" for c in f.columns], ",") *
        "]}" :
        """{"name":"$(f.name)","kind":"$(f.kind)","type":"$(f.type)"}"""
    end
    return """{"abi":2,"fields":[""" * join(parts, ",") * "]}"
end

# `P_new(n)` stays as the fast path: one integer placeholder needs no builder.
_is_scalar_new(fields) =
    length(fields) == 1 && fields[1].kind == "scalar" && fields[1].type == "i64"

# The prefix becomes a C symbol and a Julia identifier in generated source, so
# it has to be one. Checked here rather than discovered as a syntax error in a
# generated file nobody is looking at.
function _check_prefix(prefix::AbstractString)
    isempty(prefix) && throw(ArgumentError("prefix must not be empty"))
    ok = all(c -> isascii(c) && (isletter(c) || isdigit(c) || c == '_'), prefix)
    (ok && !isdigit(first(prefix))) || throw(
        ArgumentError(
            "prefix must be a C identifier (ASCII letters, digits, underscore; " *
            "not starting with a digit) — got $(repr(prefix))",
        ),
    )
    return prefix
end

# ── The packages a core's own types come from ─────────────────────────────────
#
# A recipe defers more than sizes.  A starting point that varies with the index,
# or the set a generator runs over when that set is computed from the size, is a
# function the MODELLING library wrote, and it travels inside the serialized core
# as a type that library owns.  Nothing is wrong with that — it is data, it
# serializes, and `instantiate` runs it — but the generated app is a different
# process, and it has to be able to name those types again.
#
# `Serialization` resolves a type's module through `Base.PkgId`, and only among
# modules that are LOADED.  So the app has to do two things, and doing one is
# not enough: depend on the package, and `import` it before deserializing.  With
# the dependency present but no import, the deserialize fails with exactly the
# same `KeyError` as with no dependency at all — which is why this is worth
# stating rather than leaving to whoever reads the generated `Project.toml`.
#
# An EXTENSION is where a modelling library naturally writes these functions,
# since they are the ones that use ExaModels.  An extension cannot be depended
# on directly, so it is resolved to its parent package: the app imports the
# parent, ExaModels is already imported, and Julia loads the extension itself.

struct _Pkg
    name::String
    uuid::Base.UUID
    dir::String
end

function _collect_modules!(mods::Set{Module}, seen::Base.IdSet{Any}, @nospecialize(T))
    T in seen && return mods
    push!(seen, T)
    if T isa UnionAll
        return _collect_modules!(mods, seen, T.body)
    elseif T isa Union
        _collect_modules!(mods, seen, T.a)
        return _collect_modules!(mods, seen, T.b)
    elseif T isa DataType
        push!(mods, parentmodule(T))
        for p in T.parameters
            p isa Type && _collect_modules!(mods, seen, p)
        end
    end
    return mods
end

# The package a module belongs to, or `nothing` for Base/Core/ExaModels, for
# `Main`, and for anything else with no `Project.toml` behind it.  A module
# whose entry point is not `<dir>/src/<Name>.jl` is an extension: its `pkgdir`
# is already the parent's, so the parent's name and uuid are read from there.
function _owning_package(m::Module)
    (m === Base || m === Core || m === ExaModels) && return nothing
    Base.moduleroot(m) === m || return nothing
    id = Base.PkgId(m)
    id.uuid === nothing && return nothing              # Main, and other non-packages
    dir = pkgdir(m)
    dir === nothing && return nothing
    path = Base.locate_package(id)
    if path !== nothing &&
       normpath(path) == normpath(joinpath(dir, "src", id.name * ".jl"))
        return _Pkg(id.name, id.uuid, abspath(dir))
    end
    # An extension — take the package it belongs to.
    for base in ("JuliaProject.toml", "Project.toml")
        proj = joinpath(dir, base)
        isfile(proj) || continue
        toml = TOML.parsefile(proj)
        name = get(toml, "name", nothing)
        uuid = get(toml, "uuid", nothing)
        (name isa String && uuid isa String) || continue
        return _Pkg(name, Base.UUID(uuid), abspath(dir))
    end
    return nothing
end

"""
    _core_packages(core) -> Vector{_Pkg}

Every package, other than ExaModels itself, that owns a type inside `core`.
These become dependencies of the generated app *and* imports in its module, so
that the serialized core can be read back there.
"""
function _core_packages(core)
    mods = _collect_modules!(Set{Module}(), Base.IdSet{Any}(), typeof(core))
    pkgs = _Pkg[]
    for m in sort!(collect(mods); by = string)
        p = _owning_package(m)
        p === nothing && continue
        any(q -> q.uuid == p.uuid, pkgs) || push!(pkgs, p)
    end
    return pkgs
end

# ── Generating the app package ────────────────────────────────────────────────

function _generate_app(core, arg, field, prefix::AbstractString)
    appdir = mktempdir(; prefix = "examodelsc_")
    modname = "ExaLib_" * prefix
    srcdir = joinpath(appdir, "src")
    mkpath(srcdir)

    # The core and the example travel as data.  A core built against `arg` is
    # plain data — trees of `Node1`/`Node2`/`Var` structs, arrays, tuples — with
    # no closures in the evaluated path, which is what makes this possible at
    # all.
    Serialization.serialize(joinpath(srcdir, "core.jls"), core)
    Serialization.serialize(joinpath(srcdir, "arg.jls"), arg)

    # JuliaC copies the app into a fresh temporary directory before
    # instantiating, which silently breaks relative `path =` entries: they would
    # resolve against the copy's parent. Absolute from the start.
    exadir = abspath(joinpath(dirname(dirname(pathof(ExaModels)))))
    pkgs = _core_packages(core)
    write(joinpath(appdir, "Project.toml"), _project_toml(modname, exadir, pkgs))
    write(
        joinpath(srcdir, modname * ".jl"),
        _module_source(modname, prefix, field, pkgs),
    )
    return appdir
end

_toml_path(dir::AbstractString) = replace(dir, '\\' => '/')

function _project_toml(modname::AbstractString, exadir::AbstractString, pkgs = _Pkg[])
    deps = join(("""$(p.name) = "$(p.uuid)"\n""" for p in pkgs))
    # A path source rather than a version bound: the app must compile the same
    # code that produced the core, not merely something compatible with it.
    sources = join(("""$(p.name) = {path = "$(_toml_path(p.dir))"}\n""" for p in pkgs))
    return """
    name = "$modname"
    uuid = "$_GEN_UUID"
    version = "0.1.0"

    [deps]
    ExaModels = "1037b233-b668-4ce9-9b63-f9f681f55dd2"
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
    $deps
    [sources]
    ExaModels = {path = "$(_toml_path(exadir))"}
    $sources"""
end

# A fixed model: the core carries no placeholders, so `<prefix>_new(n)`
# ignores `n`. The C symbol keeps its one-integer signature anyway — every
# consumer already speaks it, and a second ABI shape would buy nothing.
struct FixedModel end

@inline _nargs_count(::Val{N}) where {N} = N

# How `rec_new(n)` turns its one integer into the argument the core expects —
# or, for a fixed model, into the `nothing` that `ExaModel(core, nothing)`
# treats as "no instance data".
_arg_expr(::Nothing) = "Int(n)"
_arg_expr(field::Symbol) = "(; $field = Int(n))"
_arg_expr(::FixedModel) = "nothing"

# What `<prefix>_nargs()` reports: how many instantiation arguments
# `<prefix>_new` actually consumes. 0 is what lets a consumer handed only a
# library path know that no `args` are required.
_nargs_value(::FixedModel) = 0
_nargs_value(::Union{Nothing, Symbol}) = 1

function _module_source(modname::AbstractString, p::AbstractString, field, pkgs = _Pkg[])
    argexpr = _arg_expr(field)
    # Imported for their side effect on `Serialization`: a module has to be
    # loaded before a type it owns can be resolved by `PkgId`, and the
    # deserialize below is what needs them.  A dependency alone is not enough.
    imports = join(("import $(p.name)\n" for p in pkgs))
    return """
    module $modname

    import ExaModels
    import Serialization
    $imports
    # Deserialized at precompile time, so the core is baked into the package
    # image and no model-building code enters the compiled call graph.
    const CORE = Serialization.deserialize(joinpath(@__DIR__, "core.jls"))
    const ARG0 = Serialization.deserialize(joinpath(@__DIR__, "arg.jls"))

    # Building one model at precompile time fixes the concrete instance type.
    # Every runtime instantiation differs only in sizes, so they all land in
    # this vector without widening it.  `check = Val(false)` drops the
    # placeholder-leak guard, which walks types reflectively and is not
    # trimmable — the check already ran, on this exact core, in the process
    # that called `compile_library`.
    const MODELS = typeof(ExaModels.ExaModel(CORE, ARG0; check = Val(false)))[]

    @inline _valid(id::Cint) = 1 <= id <= length(MODELS)

    # How many instantiation arguments `$(p)_new` consumes (0 = fixed model,
    # `n` is ignored). Lets a consumer decide whether `args` are required
    # before instantiating anything.
    Base.@ccallable function $(p)_nargs()::Cint
        return Cint($(_nargs_value(field)))
    end

    Base.@ccallable function $(p)_new(n::Cint)::Cint
        try
            push!(MODELS, ExaModels.ExaModel(CORE, $argexpr; check = Val(false)))
            return Cint(length(MODELS))
        catch
            return Cint(0)          # 0 is the failure value for _new
        end
    end

    Base.@ccallable function $(p)_nvar(id::Cint)::Cint
        _valid(id) || return Cint(-1)
        return Cint(MODELS[Int(id)].meta.nvar)
    end

    Base.@ccallable function $(p)_ncon(id::Cint)::Cint
        _valid(id) || return Cint(-1)
        return Cint(MODELS[Int(id)].meta.ncon)
    end

    Base.@ccallable function $(p)_nnzj(id::Cint)::Cint
        _valid(id) || return Cint(-1)
        return Cint(MODELS[Int(id)].meta.nnzj)
    end

    Base.@ccallable function $(p)_nnzh(id::Cint)::Cint
        _valid(id) || return Cint(-1)
        return Cint(MODELS[Int(id)].meta.nnzh)
    end

    Base.@ccallable function $(p)_meta(
        id::Cint,
        x0::Ptr{Cdouble},
        lvar::Ptr{Cdouble},
        uvar::Ptr{Cdouble},
        lcon::Ptr{Cdouble},
        ucon::Ptr{Cdouble},
    )::Cint
        _valid(id) || return Cint(1)
        try
            m = MODELS[Int(id)]
            n = m.meta.nvar
            k = m.meta.ncon
            copyto!(unsafe_wrap(Array, x0, n), m.meta.x0)
            copyto!(unsafe_wrap(Array, lvar, n), m.meta.lvar)
            copyto!(unsafe_wrap(Array, uvar, n), m.meta.uvar)
            if k > 0
                copyto!(unsafe_wrap(Array, lcon, k), m.meta.lcon)
                copyto!(unsafe_wrap(Array, ucon, k), m.meta.ucon)
            end
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_obj(id::Cint, x::Ptr{Cdouble}, out::Ptr{Cdouble})::Cint
        _valid(id) || return Cint(1)
        try
            m = MODELS[Int(id)]
            unsafe_store!(out, ExaModels.obj(m, unsafe_wrap(Array, x, m.meta.nvar)))
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_grad(id::Cint, x::Ptr{Cdouble}, g::Ptr{Cdouble})::Cint
        _valid(id) || return Cint(1)
        try
            m = MODELS[Int(id)]
            n = m.meta.nvar
            ExaModels.grad!(m, unsafe_wrap(Array, x, n), unsafe_wrap(Array, g, n))
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_cons(id::Cint, x::Ptr{Cdouble}, c::Ptr{Cdouble})::Cint
        _valid(id) || return Cint(1)
        try
            m = MODELS[Int(id)]
            k = m.meta.ncon
            k == 0 && return Cint(0)
            ExaModels.cons_nln!(
                m,
                unsafe_wrap(Array, x, m.meta.nvar),
                unsafe_wrap(Array, c, k),
            )
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_jac_structure(
        id::Cint,
        rows::Ptr{Cint},
        cols::Ptr{Cint},
    )::Cint
        _valid(id) || return Cint(1)
        try
            m = MODELS[Int(id)]
            nz = m.meta.nnzj
            nz == 0 && return Cint(0)
            ExaModels.jac_structure!(
                m,
                unsafe_wrap(Array, rows, nz),
                unsafe_wrap(Array, cols, nz),
            )
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_jac(id::Cint, x::Ptr{Cdouble}, vals::Ptr{Cdouble})::Cint
        _valid(id) || return Cint(1)
        try
            m = MODELS[Int(id)]
            nz = m.meta.nnzj
            nz == 0 && return Cint(0)
            ExaModels.jac_coord!(
                m,
                unsafe_wrap(Array, x, m.meta.nvar),
                unsafe_wrap(Array, vals, nz),
            )
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_hess_structure(
        id::Cint,
        rows::Ptr{Cint},
        cols::Ptr{Cint},
    )::Cint
        _valid(id) || return Cint(1)
        try
            m = MODELS[Int(id)]
            nz = m.meta.nnzh
            nz == 0 && return Cint(0)
            ExaModels.hess_structure!(
                m,
                unsafe_wrap(Array, rows, nz),
                unsafe_wrap(Array, cols, nz),
            )
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_hess(
        id::Cint,
        x::Ptr{Cdouble},
        y::Ptr{Cdouble},
        obj_weight::Cdouble,
        vals::Ptr{Cdouble},
    )::Cint
        _valid(id) || return Cint(1)
        try
            m = MODELS[Int(id)]
            nz = m.meta.nnzh
            nz == 0 && return Cint(0)
            ExaModels.hess_coord!(
                m,
                unsafe_wrap(Array, x, m.meta.nvar),
                unsafe_wrap(Array, y, m.meta.ncon),
                unsafe_wrap(Array, vals, nz);
                obj_weight = obj_weight,
            )
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    end # module $modname
    """
end

# ── Driving juliac ────────────────────────────────────────────────────────────

function _drive_juliac(appdir, prefix, out, trim, bundle, verbose)
    outdir = abspath(out)
    mkpath(outdir)

    img = JuliaC.ImageRecipe(
        file = appdir,
        output_type = "--output-lib",
        add_ccallables = true,
        trim_mode = trim,
        julia_args = ["--experimental"],
        verbose = verbose,
    )
    JuliaC.compile_products(img)
    return _link(Val(bundle), img, prefix, outdir)
end

const _DLEXT = Base.BinaryPlatforms.platform_dlext()

# ── bundled: carries a privatized runtime; loadable from anything ────────────
function _link(::Val{true}, img, prefix, outdir)
    # The bundle owns its directory: the bundler refuses to overwrite an
    # existing one, and privatized runtime files carry randomized names that
    # would otherwise accumulate across rebuilds.  Removal is per-entry and
    # tolerant — on NFS a dying process leaves .nfs* ghosts that refuse to go
    # and collide with nothing the bundler writes.
    for entry in readdir(outdir; join = true)
        try
            rm(entry; recursive = true, force = true)
        catch
        end
    end

    link = JuliaC.LinkRecipe(
        image_recipe = img,
        outname = joinpath(outdir, "lib" * prefix),
        rpath = JuliaC.RPATH_BUNDLE,
    )
    JuliaC.link_products(link)
    # Privatization mangles the bundled runtime under a salt drawn from the
    # task-local RNG — and two sequential `compile_library` calls in one
    # process have been observed to draw the SAME salt (2026-08-11, rosen +
    # fixed in one test run). Identically-salted bundles cannot coexist in a
    # consumer: the dynamic loader satisfies the second library's NEEDED
    # entries with the first's already-loaded runtime, whose thread-adoption
    # guard then aborts the whole host process on the second library's first
    # call. Distinctly-salted bundles coexist fine (verified). So the salt is
    # forced unique here: bundling runs in its own task whose RNG is seeded
    # from the OS entropy pool — the caller's RNG state is untouched.
    fetch(Threads.@spawn begin
        Random.seed!(rand(Random.RandomDevice(), UInt128))
        JuliaC.bundle_products(
            JuliaC.BundleRecipe(link_recipe = link, output_dir = outdir, privatize = true),
        )
    end)

    libroot = Sys.iswindows() ? "bin" : "lib"
    libpath = joinpath(outdir, libroot, "lib" * prefix * "." * _DLEXT)
    isfile(libpath) || error("juliac produced no library at $libpath")
    return (; libpath, outdir, prefix)
end

# ── unbundled: one file, linked against the installed Julia ─────────────────
function _link(::Val{false}, img, prefix, outdir)
    # Not cleared: without a bundle each model is a single file, so a directory
    # may hold several and wiping it would delete a sibling.
    libpath = joinpath(outdir, "lib" * prefix * "." * _DLEXT)
    link = JuliaC.LinkRecipe(
        image_recipe = img,
        outname = splitext(libpath)[1],
        rpath = JuliaC.RPATH_JULIA,   # absolute paths into the Julia installation
    )
    JuliaC.link_products(link)
    isfile(libpath) || error("juliac produced no library at $libpath")
    return (; libpath, outdir, prefix)
end

end # module ExaModelsC
