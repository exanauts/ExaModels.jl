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

c, N = ExaCore(concrete = Val(true), nargs = Val(1))
@add_var(c, x, N; start = 1.0)
@add_obj(c, (x[i] - 1)^2 for i in 1:N)

compile_library(c, "/opt/models/rosen"; arg = 10)
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
ipopt(CNLPModel(lib; prefix = "rosen", args = 1000))
```

```python
import cnlpmodels
lib = cnlpmodels.load("/opt/models/rosen/lib/librosen.so")
m = cnlpmodels.CModel(lib, prefix="rosen", args=1000)
```

Both consumers default `prefix` to `"rec"` when handed a library handle, while
`compile_library` defaults it to the output directory's name — pass it
explicitly unless those coincide.  The test suite exercises both.
"""
module ExaModelsC

import ExaModels
import JuliaC
import Serialization

export compile_library

# The generated app package is a throwaway: a fresh temporary directory with
# its own environment, never registered.  A constant UUID is therefore safe and
# saves a UUIDs dependency.
const _GEN_UUID = "b41c7e02-9f3d-4a58-8e6c-2d0f5a7c9b13"

"""
    compile_library(core, out; arg, prefix = basename(out), trim = "safe",
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

## Where it goes

`out` may be a directory path, or a bare **name** — no directory part — in
which case the library is installed on the CNLPModels search path
(`CNLPMODELS_PATH`), where both consumers find it by that name:

```julia
compile_library(core, "rosenrock"; arg = 1000)   # → \$CNLPMODELS_PATH/librosenrock.so
CNLPModel("rosenrock"; args = 1000)              # finds it, prefix defaults to the name
```

`core` must be an `ExaCore` built against a single
[`ExaModels.ArgSource`](@ref) — i.e. `ExaCore(nargs = Val(1))`, since the
scalar ABI carries one instantiation argument.  `arg` is an
example argument of the shape the library will be instantiated with.  The
example is used twice — to pin the types `juliac` needs in order to trim, and to
check that the core actually instantiates before spending minutes compiling it.

The library exports, for `prefix` `P`: `P_new(n) -> id` (a positive instance id;
any number of instances may coexist), then id-first `P_nvar`, `P_ncon`,
`P_nnzj`, `P_nnzh`, `P_meta`, `P_obj`, `P_grad`, `P_cons`, `P_jac_structure`,
`P_jac`, `P_hess_structure`, `P_hess`.  Indices are 1-based; the Hessian is the
lower triangle of `obj_weight * ∇²f + Σᵢ yᵢ ∇²cᵢ`; every function returns a
`Cint` status, `0` on success, and none of them throws across the boundary.

`P_new` takes a single integer, so this form applies when the example `arg` is
an `Integer`, or a `NamedTuple` holding exactly one integer field — the
"scalable model" case (`rosenbrock` at size `N`).  Structured instantiation
(the schema + builder ABI, for data-defined models such as OPF) is not built
yet; `compile_library` says so rather than emitting a library that would fail
at load.
"""
function compile_library(
    core::ExaModels.ExaCore,
    out::AbstractString;
    arg,
    prefix::AbstractString = basename(abspath(out)),
    trim::AbstractString = "safe",
    bundle::Bool = true,
    verbose::Bool = false,
)
    _check_prefix(prefix)
    field = _scalar_field(arg)
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

# ── Where the library goes ────────────────────────────────────────────────────

# A bare name — no directory part — is resolved against the CNLPModels search
# path, so `compile_library(core, "rosenrock")` installs where
# `CNLPModel("rosenrock")` and `cnlpmodels.CModel("rosenrock")` will look for
# it.  No sigil is needed to say which is meant: a name is not a path.
#
# Both layouts are ones the consumers already try: a bundle lands at
# `<dir>/<name>/lib/lib<name>.<ext>`, a single file at `<dir>/lib<name>.<ext>`.
function _resolve_out(out::AbstractString, bundle::Bool)
    (isabspath(out) || !isempty(splitdir(out)[1])) && return abspath(out)
    dirs = filter(!isempty, split(get(ENV, "CNLPMODELS_PATH", ""), ':'))
    isempty(dirs) && throw(
        ArgumentError(
            "`$out` has no directory part, so it is taken as a library name to " *
            "install on the CNLPModels search path — but CNLPMODELS_PATH is " *
            "not set. Set it, or pass a path such as `\"./$out\"`.",
        ),
    )
    # A bundle gets a directory of its own — `<dir>/<name>/lib/lib<name>.<ext>`,
    # the consumers' second layout.  A single file goes straight into the
    # directory as `<dir>/lib<name>.<ext>`, their first.
    return bundle ? joinpath(first(dirs), out) : first(dirs)
end

# ── Reading the example argument ──────────────────────────────────────────────

# `P_new` carries one integer, so the example has to say where that integer
# goes.  Returns `nothing` when the argument *is* the integer, or the field name
# to wrap it in.  Anything else is the structured case, which needs the schema
# ABI rather than `P_new`.
function _scalar_field(arg)
    arg isa Integer && return nothing
    if arg isa NamedTuple && length(arg) == 1
        v = first(arg)
        v isa Integer && return first(keys(arg))
    end
    throw(
        ArgumentError(
            "compile_library currently emits the scalar instantiation ABI " *
            "(`<prefix>_new(n)`), so the example `arg` must be an Integer or a " *
            "NamedTuple with exactly one integer field — got $(typeof(arg)). " *
            "Structured data (the schema + builder ABI) is not implemented yet.",
        ),
    )
end

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
    write(joinpath(appdir, "Project.toml"), _project_toml(modname, exadir))
    write(joinpath(srcdir, modname * ".jl"), _module_source(modname, prefix, field))
    return appdir
end

function _project_toml(modname::AbstractString, exadir::AbstractString)
    return """
    name = "$modname"
    uuid = "$_GEN_UUID"
    version = "0.1.0"

    [deps]
    ExaModels = "1037b233-b668-4ce9-9b63-f9f681f55dd2"
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

    [sources]
    ExaModels = {path = "$(replace(exadir, '\\' => '/'))"}
    """
end

# How `rec_new(n)` turns its one integer into the argument the core expects.
_arg_expr(::Nothing) = "Int(n)"
_arg_expr(field::Symbol) = "(; $field = Int(n))"

function _module_source(modname::AbstractString, p::AbstractString, field)
    argexpr = _arg_expr(field)
    return """
    module $modname

    import ExaModels
    import Serialization

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
    JuliaC.bundle_products(
        JuliaC.BundleRecipe(link_recipe = link, output_dir = outdir, privatize = true),
    )

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
