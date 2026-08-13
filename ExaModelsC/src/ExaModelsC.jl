"""
    ExaModelsC

Compile an [`ExaModels.ExaCore`](@ref) into a shared library that
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
lib = CNLPModels.load("/opt/models/rosen/librosen.so")
ipopt(CNLPModel(lib, 1000; prefix = "rosen"))
```

```python
import cnlpmodels
lib = cnlpmodels.load("/opt/models/rosen/librosen.so")
m = cnlpmodels.CModel(lib, 1000, prefix="rosen")
```

Both consumers default `prefix` to `"rec"` when handed a library handle, while
`compile_library` defaults it to the output directory's name — pass it
explicitly unless those coincide.  The test suite exercises both.

## Several models in one library

A library can carry more than one model, each under its own symbol prefix.
Name them, and the consumers select by the same name:

```julia
compile_library("@grid", :acopf => (ac_core, 100), :dcopf => (dc_core, 100))
```
```julia
CNLPModel("@grid", :acopf, 100)                  # Julia
```
```python
cnlpmodels.CModel("grid", 100, prefix="acopf")   # Python — a bare name, no sigil
```

They share the library file and, in a bundle, its one privatized ~80 MB Julia
runtime — which is the reason to co-package a family of models rather than emit
a library each.
"""
module ExaModelsC

import ExaModels
import JuliaC
import Pkg
import Random
import Serialization
import TOML

export compile_library

# The generated app package is a throwaway: a fresh temporary directory with
# its own environment, never registered.  A constant UUID is therefore safe and
# saves a UUIDs dependency.
const _GEN_UUID = "b41c7e02-9f3d-4a58-8e6c-2d0f5a7c9b13"

# One model's compiled form: its prefix, its concretized core, the example value
# whose types `juliac` needs, and how `<prefix>_new(n)` rebuilds it (`field`).
struct ModelSpec
    prefix::String
    core::Any
    arg::Any
    field::Any
end

"""
    compile_library(out, core, args...; prefix = basename(out), trim = "safe",
                    bundle = false, verbose = false)
        -> (; libpath, outdir, prefix)

Compile `core` into a shared library under `out`, and return the path to it.

## `bundle`

`bundle = false` (the default) emits a single ~2 MB library linked against the
Julia installation it was built with, which the consumer's machine must then
have (same version, found through the library's recorded rpath or provided by
the consumer).  `bundle = true` emits a directory carrying the library together
with a **privatized** copy of the Julia runtime — around 80 MB, needing no
Julia on the consumer's side.

| consumer | `bundle = false` (default) | `bundle = true` |
|:---------|:---------------------------|:----------------|
| Python (`cnlpmodels`), C | works | works |
| Julia (`CNLPModels.jl`) | works on Linux; refused elsewhere | works |

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

The two forms therefore differ only in *when* privatization happens.  A bundle
carries its privatized runtime with it; an unbundled library gets one at load
time — on Linux, `CNLPModels.load` detects the standard `libjulia` soname in
its NEEDED entries and provisions a salted copy of the consumer's *installed*
runtime (the same transformation JuliaC's bundler applies, replayed in
scratch).  Python and C callers load either form as-is: there the calling
thread is genuinely foreign and the library's own runtime initializes on the
first call.  On macOS the load-time half is not implemented, so
`CNLPModels.jl` needs the bundle there; it refuses the unbundled form with an
explanation rather than aborting.

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

The library exports, for `prefix` `P`, one of two **instantiation surfaces**,
then the shared evaluators — id-first `P_nvar`, `P_ncon`, `P_nnzj`, `P_nnzh`,
`P_meta`, `P_obj`, `P_grad`, `P_cons`, `P_jac_structure`, `P_jac`,
`P_hess_structure`, `P_hess`.  Indices are 1-based; the Hessian is the lower
triangle of `obj_weight * ∇²f + Σᵢ yᵢ ∇²cᵢ`; every function returns a `Cint`
status, `0` on success, and none of them throws across the boundary.

**One integer** (`rosenbrock` at size `N` — a bare `Integer` example, or a
`NamedTuple` holding exactly one): `P_new(n) -> id` (a positive instance id;
any number of instances may coexist), and `P_nargs() -> 0 or 1` saying whether
`n` is consumed or ignored (0 is the fixed-model case).

**Anything else** — several example values, floats, arrays, tables (vectors
of NamedTuples), or NamedTuples of these, exactly as `ExaModel` takes them —
gets the schema + builder ABI instead of `P_new`: `P_schema` publishes a JSON
description of the fields, and `P_data_begin` / `P_set_scalar_{i64,f64}` /
`P_set_array_{i64,f64}` / `P_set_col_{i64,f64}` / `P_data_ready` /
`P_new_from_data -> id` take the values by field name and reassemble the
`ExaModel` arguments.  A `NamedTuple` example flattens into one schema field
per entry, named by its key; bare values are named `arg1`, `arg2`, ... by
position.  Both consumers already speak this surface, binding one value per
field positionally — `CNLPModel(lib, 3, lo, v)` — so a compiled model is
consumed the way it was written.  Builder examples must be `Int64`/`Float64`
exactly (as scalars, `Vector`s, or table entries): the example's type IS the
compiled storage's type.

**An argument function** (`argfun = f`, or `:name => (core, f, example)` in
the multi-model form — a function can never be a model argument, so second
position is unambiguous) is the third surface: the library carries `f` and
calls it at run time, so only `f`'s own argument crosses the boundary — one
string (`P_new_str(const char *)`) or one integer (`P_new(n)`) — and the
work that turns it into instantiation data happens inside the library.  It
composes with the builder rather than competing: the builder passes
structured data ACROSS the boundary, this keeps the data on the far side
and passes a path.  `f` must be a named function a package owns — it is
emitted by name, which is what `juliac` resolves statically — and it must
return the argument TUPLE the core is instantiated with.  The example is
`f`'s argument, and `f(example)` is probed before compiling.

Every model also exports `P_argkind() -> 0 | 1 | 2 | 3` (fixed, `P_new(n)`,
`P_new_str`, builder), so a consumer routes on the declared shape instead
of probing for symbols.  Kind 1 covers both n-is-the-size and
n-goes-to-an-argument-function: the two are indistinguishable to a consumer
**by design** — the call shape is identical, and what the library does with
`n` is its own business.  `P_nargs` says how MANY values instantiation
takes and cannot say what shape they are; `P_argkind` is the other half of
that pair, and a consumer handed only a library path needs both.

## Several models in one library

Give `:name => core` pairs instead of a single core to put more than one model
in the library — see the method below.
"""
function compile_library(
    out::AbstractString,
    core::ExaModels.ExaCore,
    args...;
    prefix::AbstractString = _default_out_prefix(out),
    trim::AbstractString = "safe",
    bundle::Bool = false,
    verbose::Bool = false,
    argfun = nothing,
)
    spec = _model_spec(prefix, core, argfun, args)
    r = _compile([spec], out, prefix, trim, bundle, verbose)
    # One model keeps the singular field it has always returned.
    return (; libpath = r.libpath, outdir = r.outdir, prefix = spec.prefix)
end

"""
    compile_library(out, :name1 => (core1, args1...), :name2 => core2, ...;
                    trim = "safe", bundle = true, verbose = false)
        -> (; libpath, outdir, prefixes)

Compile **several models into one shared library**, each under its own symbol
prefix — the same names the consumer selects with, `CNLPModel(lib, :name1,
args...)`.

Each model is a pair: the name, then either a bare `core` (a fixed model,
taking no instantiation data) or a tuple `(core, args...)` giving the example
values exactly as `ExaModel(core, ...)` would take them.

```julia
compile_library("@grid",
    :acopf => (ac_core, 100),      # acopf_new(n) inside libgrid.so
    :dcopf => (dc_core, 100),      # dcopf_new(n) in the same library
    :fixed => small_core,          # fixed_nargs() == 0
)
CNLPModel("@grid", :acopf, 100)    # the consumer's spelling
```

The models share one library file and, in a bundle, one privatized ~80 MB Julia
runtime — which is the reason to co-package them rather than emit one library
each. They are otherwise independent: separate cores, separate instance tables,
separate ids.

The library FILE is named after `out` (`@grid` → `libgrid.so`), not after any
one model, so `prefix =` has no meaning here and is not accepted — the model
names supply the prefixes. Every other keyword behaves as in the single-model
form.

Each model gets whichever instantiation surface its examples call for, exactly
as in the single-model form: `P_new(n)` for one integer, the schema + builder
ABI for anything else, per prefix — the surfaces coexist freely in one
library.
"""
function compile_library(
    out::AbstractString,
    models::Pair{Symbol}...;
    trim::AbstractString = "safe",
    bundle::Bool = false,
    verbose::Bool = false,
)
    isempty(models) && throw(ArgumentError("give at least one `:name => core` model"))
    specs = ModelSpec[
        _model_spec(String(name), _core_and_args(name, value)...) for (name, value) in models
    ]
    _check_distinct(specs)
    # The FILE is named after `out`; the models are named by their symbols.
    r = _compile(specs, out, _default_out_prefix(out), trim, bundle, verbose)
    return (; libpath = r.libpath, outdir = r.outdir, prefixes = [s.prefix for s in specs])
end

# `:name => core` is a fixed model; `:name => (core, args...)` carries example
# values. Anything else is named precisely here — the pair spelling is easy to
# get subtly wrong, and a mistake that reached `_model_spec` would be reported
# as though the core itself were at fault.
# A pair cannot carry keywords, so an argument function is positional: it goes
# SECOND, `:name => (core, argfun, example)`. Unambiguous because nothing
# callable can ever be a model argument — the C boundary carries numbers,
# arrays and tables, never a function.
_core_and_args(::Symbol, core::ExaModels.ExaCore) = (core, nothing, ())
function _core_and_args(name::Symbol, value::Tuple)
    isempty(value) && throw(
        ArgumentError(
            "`:$name => ()` names no core — give `:$name => core` for a fixed " *
            "model, or `:$name => (core, args...)`.",
        ),
    )
    first(value) isa ExaModels.ExaCore || throw(
        ArgumentError(
            "`:$name` must begin with an `ExaCore` — got a $(typeof(first(value))).",
        ),
    )
    rest = Base.tail(value)
    if !isempty(rest) && first(rest) isa Function
        return (first(value), first(rest), Base.tail(rest))
    end
    return (first(value), nothing, rest)
end
_core_and_args(name::Symbol, value) = throw(
    ArgumentError(
        "`:$name => $(typeof(value))` is not a model — give `:$name => core`, or " *
        "`:$name => (core, args...)` with the example values `ExaModel` takes.",
    ),
)

# Two models under one prefix would emit duplicate `@ccallable` names, which
# juliac reports as a redefinition inside a generated file. Caught here instead.
function _check_distinct(specs::Vector{ModelSpec})
    seen = Set{String}()
    for s in specs
        s.prefix in seen && throw(
            ArgumentError(
                "two models are both named `$(s.prefix)` — one library cannot " *
                "carry the same symbol prefix twice.",
            ),
        )
        push!(seen, s.prefix)
    end
    return specs
end

# The per-model validation both entry points share. Everything here is checked
# BEFORE any code generation, so a bad model is reported in the caller's process
# rather than as a compile error in a generated file nobody is looking at.
# The three-argument form: no argument function. Kept as the spelling for
# every caller that has none, including the tests' hermetic probes.
_model_spec(prefix::AbstractString, core::ExaModels.ExaCore, args::Tuple) =
    _model_spec(prefix, core, nothing, args)

function _model_spec(prefix::AbstractString, core::ExaModels.ExaCore, argfun, args::Tuple)
    _check_prefix(prefix)
    core = _concretize(core)
    if argfun !== nothing
        # `args` are the example arguments to the FUNCTION, not to the core:
        # the library carries the function, calls it at run time, and
        # instantiates from whatever it returns. That is what lets a
        # data-defined model be compiled once and instantiated at any data
        # without a table crossing the boundary.
        isempty(args) && throw(
            ArgumentError(
                "`$prefix`: `argfun` was given, so the examples are the " *
                "arguments to call it with — e.g. `:$prefix => (core, " *
                "ac_opf_args, \"case14.m\")`.",
            ),
        )
        kind = _argfun_kind(prefix, args)
        arg = argfun(args...)
        arg isa Tuple || throw(
            ArgumentError(
                "`$prefix`: `argfun` must return the argument TUPLE the core is " *
                "instantiated with — what you would splat into " *
                "`ExaModel(core, ...)`. Got a $(typeof(arg)); wrap it, as `(data,)`.",
            ),
        )
        return ModelSpec(String(prefix), core, arg, UserArgs{kind}(_argfun_call(prefix, argfun), argfun))
    end
    if isempty(args)
        # No examples means a FIXED model: the core has no placeholders and is
        # compiled as-is. A core that declared placeholders cannot be meant —
        # refuse it here with the arity it stated rather than letting the
        # instantiation probe below fail on a bare recipe. (An UNDECLARED
        # recipe — a bare `ArgSource()` written into a plain `ExaCore()` —
        # still reaches the probe, which rejects it just as loudly.)
        core.nargs isa Val{0} || throw(
            ArgumentError(
                "`$prefix`: this core declared $(_nargs_count(core.nargs)) " *
                "placeholder(s) — give one example value per placeholder, as you " *
                "would to `ExaModel(core, ...)`; its types are what the compiler " *
                "needs.",
            ),
        )
        return ModelSpec(String(prefix), core, nothing, FixedModel())
    end
    bm = _builder_model(args)
    # One integer placeholder — bare, or a one-key NamedTuple — keeps the
    # `P_new(n)` fast path: one C call, no builder.  The two surfaces are
    # disjoint on purpose: a library exports `P_new` exactly when its schema
    # is a single integer scalar, which is what lets a consumer route a lone
    # integer without guessing.
    if _is_scalar_new(bm.fields)
        spec = only(bm.argspec)
        field = spec isa String ? nothing : first(only(spec))
        return ModelSpec(String(prefix), core, only(args), field)
    end
    return ModelSpec(String(prefix), core, args, bm)
end

# The shared back half: probe every model, generate one app carrying all of
# them, compile once. `libname` names the library FILE (`lib<libname>.<ext>`) —
# the library's identity to a consumer resolving an `@name` or a bundle
# directory, and distinct from the models' prefixes as soon as there is more
# than one model.
function _compile(specs::Vector{ModelSpec}, out, libname, trim, bundle, verbose)
    outdir = _resolve_out(out, bundle)

    # Instantiate here, in this process, before generating anything.  A core
    # that cannot be instantiated produces a library that cannot be loaded, and
    # the failure is far cheaper to read now than after a juliac run — the more
    # so with several models, where one bad core would waste the whole compile.
    for s in specs
        # A builder spec carries its examples as a tuple, one per placeholder;
        # the other forms carry a single value (or `nothing` for a fixed core).
        probe = try
            s.arg isa Tuple ? ExaModels.ExaModel(s.core, s.arg...) :
            ExaModels.ExaModel(s.core, s.arg)
        catch err
            # A shape mismatch between the examples and how the core reads its
            # placeholders (a NamedTuple where a bare size is expected, a
            # missing key, ...) surfaces here — as the caller's error, before
            # minutes are spent compiling.
            throw(
                ArgumentError(
                    "`$(s.prefix)`: the example values do not instantiate this " *
                    "core — `ExaModel(core, example...)` failed with: " *
                    sprint(showerror, err),
                ),
            )
        end
        verbose && @info "compile_library: core instantiates" prefix = s.prefix nvar =
            probe.meta.nvar ncon = probe.meta.ncon
    end

    appdir = _generate_app(specs, libname)
    verbose &&
        @info "compile_library: generated app" appdir models = [s.prefix for s in specs]

    return _drive_juliac(appdir, libname, outdir, trim, bundle, verbose)
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
# The schema is derived from the example values' TYPES.  A bare number or
# vector is one field, named `arg1`, `arg2`, ... by position; a NamedTuple
# example — the shape `ExaModel` takes for a source carrying several values —
# flattens into one field per entry, named by its key.  A consumer binds its
# own values positionally against the flat field order,
# `CNLPModel(lib, v1, v2, ...)`, and the library reassembles the NamedTuples
# before instantiating.

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

# Everything else gets the schema + builder surface (ABI v2): the consumer
# opens a builder, sets each field by name, and `P_new_from_data` reassembles
# the `ExaModel` arguments exactly as the example values were given here.
# `argspec` records that mapping — one entry per `ExaModel` positional
# argument:
#
#   a `String`                      — a bare value, stored under that field
#   a `Vector{Pair{Symbol,String}}` — a NamedTuple, one (key => field) per entry
struct BuilderModel
    fields::Vector{Field}
    argspec::Vector{Union{String, Vector{Pair{Symbol,String}}}}
end

# Builder storage is GENERATED from the example types, and the model vector's
# element type is fixed by instantiating the example at precompile time — so
# the example must BE the type the builder will reconstruct: Int64/Float64
# exactly, as scalars, `Vector`s, or vectors of NamedTuples of them.  A looser
# example (`Int32`, a range, `Real[]`) would compile a MODELS vector the
# reconstruction cannot feed, and the mismatch would surface only inside the
# compiled library; refused here instead.
_check_exact(name, v::Union{Int64, Float64, Vector{Int64}, Vector{Float64}}) = v
function _check_exact(name, v::Vector{T}) where {T <: NamedTuple}
    (isconcretetype(T) && all(t -> t <: Union{Int64, Float64}, fieldtypes(T))) ||
        _exact_err(name, v)
    return v
end
_check_exact(name, v) = _exact_err(name, v)
_exact_err(name, v) = throw(
    ArgumentError(
        "`$name`: builder examples must be Int64/Float64 values — as scalars, " *
        "as Vector{Int64}/Vector{Float64}, or as a Vector of NamedTuples of " *
        "them (a table); got $(typeof(v)). The storage the library compiles " *
        "is exactly the example's type.",
    ),
)

function _builder_model(args)
    fields = Field[]
    argspec = Union{String, Vector{Pair{Symbol,String}}}[]
    seen = Set{String}()
    claim = function (name, where_)
        name in seen && throw(
            ArgumentError(
                "two schema fields would both be named `$name` (the second from " *
                "$where_) — field names come from NamedTuple keys and `argN` " *
                "positions, and must be distinct across all placeholders; " *
                "rename one.",
            ),
        )
        push!(seen, name)
        return name
    end
    for (i, a) in enumerate(args)
        if a isa NamedTuple
            entries = Pair{Symbol,String}[]
            for k in keys(a)
                name = claim(String(k), "argument $i")
                push!(fields, _field(name, _check_exact(name, getfield(a, k))))
                push!(entries, k => name)
            end
            push!(argspec, entries)
        else
            name = claim("arg$i", "argument $i")
            push!(fields, _field(name, _check_exact(name, a)))
            push!(argspec, name)
        end
    end
    # Tables flatten further, into one storage slot per column — those names
    # must be distinct too, and a table needs at least one column to have an
    # element type at all.
    slots = String[]
    for f in fields
        f.kind == "table" && isempty(f.columns) && throw(
            ArgumentError(
                "`$(f.name)`: the example table has no columns — give " *
                "NamedTuples with at least one entry.",
            ),
        )
        append!(slots, (s for (s, _, _) in _slots(f)))
    end
    allunique(slots) || throw(
        ArgumentError(
            "field and table-column names collide once flattened to storage " *
            "slots ($(join(slots, ", "))) — rename one of the duplicates.",
        ),
    )
    return BuilderModel(fields, argspec)
end

# The flat storage behind a builder: (slot identifier, storage type, zero
# value), one slot per scalar/array field and per table column.  Each slot
# also carries a `<slot>_set::Bool` beside it in the generated struct.
function _slots(f::Field)
    jt(t) = t == "i64" ? "Int64" : "Float64"
    jz(t) = t == "i64" ? "0" : "0.0"
    jvt(t) = t == "i64" ? "Vector{Int64}" : "Vector{Float64}"
    jvz(t) = t == "i64" ? "Int64[]" : "Float64[]"
    f.kind == "scalar" && return [(f.name, jt(f.type), jz(f.type))]
    f.kind == "array" && return [(f.name, jvt(f.type), jvz(f.type))]
    return [("$(f.name)_$(c.first)", jvt(c.second), jvz(c.second)) for c in f.columns]
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
# `Main`, and for anything else with no `Project.toml` behind it.
function _owning_package(m::Module)
    (m === Base || m === Core || m === ExaModels) && return nothing
    Base.moduleroot(m) === m || return nothing
    id = Base.PkgId(m)
    id.uuid === nothing && return nothing              # Main, and other non-packages
    dir = pkgdir(m)
    dir === nothing && return nothing
    # Read the name and uuid from the project rather than from the module's own
    # `PkgId`: for an extension those differ, and `pkgdir` is already the parent
    # package's directory, so this one path covers both.  An extension cannot be
    # depended on by name, and does not need to be — importing the parent loads
    # it, since ExaModels is imported too.
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
_core_packages(core) = _core_packages_of(typeof(core))

function _core_packages_of(@nospecialize(T))
    mods = _collect_modules!(Set{Module}(), Base.IdSet{Any}(), T)
    pkgs = _Pkg[]
    for m in sort!(collect(mods); by = string)
        p = _owning_package(m)
        p === nothing && continue
        any(q -> q.uuid == p.uuid, pkgs) || push!(pkgs, p)
    end
    return pkgs
end

"""
    _developed_packages() -> Vector{_Pkg}

Every package tracked by path in the caller's active environment, other than
ExaModels itself (always carried separately). These become `[sources]` entries
in the generated app so it compiles the same code the caller is running —
without them, transitive dependencies resolve from the registry, and the app
can silently compile DIFFERENT code than the process that produced the core
(the build succeeds; the only trace is a path inside a stack frame).
"""
function _developed_packages()
    out = _Pkg[]
    for (uuid, info) in Pkg.dependencies()
        info.is_tracking_path || continue
        info.name == "ExaModels" && continue
        src = info.source
        src === nothing && continue
        push!(out, _Pkg(info.name, uuid, abspath(src)))
    end
    return sort!(out; by = p -> p.name)
end

# ── Generating the app package ────────────────────────────────────────────────

function _generate_app(specs::Vector{ModelSpec}, libname::AbstractString)
    appdir = mktempdir(; prefix = "examodelsc_")
    modname = _modname(libname)
    srcdir = joinpath(appdir, "src")
    mkpath(srcdir)

    # Each core and example travels as data, under its own model's name so any
    # number of them can share one app.  A core built against `arg` is plain
    # data — trees of `Node1`/`Node2`/`Var` structs, arrays, tuples — with no
    # closures in the evaluated path, which is what makes this possible at all.
    for s in specs
        Serialization.serialize(joinpath(srcdir, "core_$(s.prefix).jls"), s.core)
        Serialization.serialize(joinpath(srcdir, "arg_$(s.prefix).jls"), s.arg)
    end

    # JuliaC copies the app into a fresh temporary directory before
    # instantiating, which silently breaks relative `path =` entries: they would
    # resolve against the copy's parent. Absolute from the start.
    exadir = abspath(joinpath(dirname(dirname(pathof(ExaModels)))))
    # The union across every core: a package that owns a type inside ANY of
    # them must be a dependency and an import of the one generated app.
    pkgs = _Pkg[]
    for s in specs
        # The core's own packages, plus — for a model with an argument
        # function — the package that owns the function. That one must be
        # IMPORTED, not merely pinned: the generated module calls it by name,
        # so pinning it as a dependency leaves `ExaModelsPower.opf_args` a
        # global of unknown type and `--trim=safe` refuses the call.
        srcs = s.field isa UserArgs ?
            (_core_packages(s.core)..., _core_packages_of(typeof(_argfun_of(s)))...) :
            _core_packages(s.core)
        for p in srcs
            any(q -> q.uuid == p.uuid, pkgs) || push!(pkgs, p)
        end
    end
    # Every package the caller is DEVELOPING joins as a dependency and a path
    # source, but NOT as an import: resolution needs the pin (measured — a
    # verified `[deps]` + `[sources]` entry tracks the path with no import
    # anywhere), while only the deserialization of a core's own types needs
    # the import. Importing everything a caller happens to be developing
    # would drag unrelated packages into every app's compile.
    dev = [q for q in _developed_packages() if !any(r -> r.uuid == q.uuid, pkgs)]
    write(joinpath(appdir, "Project.toml"), _project_toml(modname, exadir, pkgs, dev))
    write(joinpath(srcdir, modname * ".jl"), _module_source(modname, specs, pkgs))
    return appdir
end

_toml_path(dir::AbstractString) = replace(dir, '\\' => '/')

# The generated MODULE name must be a Julia identifier. The library FILE name
# need not be — a consumer resolves `libmy-lib.so` out of a `my-lib/` bundle
# quite happily — so the module name is sanitized here rather than the file
# name being restricted. For a single model this is the prefix, already checked
# by `_check_prefix`, and the sanitizing is a no-op.
_modname(libname::AbstractString) = "ExaLib_" * replace(libname, r"[^A-Za-z0-9_]" => "_")

function _project_toml(
    modname::AbstractString, exadir::AbstractString, pkgs = _Pkg[], dev = _Pkg[],
)
    both = vcat(pkgs, dev)
    deps = join(("""$(p.name) = "$(p.uuid)"\n""" for p in both))
    # A path source rather than a version bound: the app must compile the same
    # code that produced the core, not merely something compatible with it.
    sources = join(("""$(p.name) = {path = "$(_toml_path(p.dir))"}\n""" for p in both))
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

# A user-supplied argument function, carried into the library and called at run
# time. `K` is the shape of the one value that crosses the C boundary — `:str`
# for `<prefix>_new_str(const char *)`, `:int` for `<prefix>_new(n)` — and
# `call` is the function spelled as source, `Pkg.fun`.
#
# This is the third instantiation surface, and it composes with the builder
# rather than competing: the builder passes structured data ACROSS the
# boundary, this keeps the data on the far side and passes a path. A caller
# holding tables wants the builder; a caller holding a filename wants this.
#
# The function is emitted BY NAME rather than serialized. A name is what
# `juliac --trim=safe` resolves statically; a deserialized function object
# would be a runtime value, and the call through it would not be.
struct UserArgs{K}
    call::String
    fun::Any
end

# The function itself, for the package walk. Stored alongside the spelling
# rather than re-resolved from it: `Pkg.fun` is a string by then.
_argfun_of(s::ModelSpec) = s.field.fun

# Which C entry point instantiates a model. Emitted for EVERY model, so a
# consumer routes on it rather than probing for symbols.
_argkind_value(::FixedModel) = 0            # `_new(n)`, `n` ignored
_argkind_value(::Union{Nothing, Symbol}) = 1  # `_new(n)`, n is the size
_argkind_value(::UserArgs{:int}) = 1        # `_new(n)`, n goes to the function
_argkind_value(::UserArgs{:str}) = 2        # `_new_str(const char *)`
_argkind_value(::BuilderModel) = 3          # `_data_begin` / `_new_from_data`

_nargs_value(::UserArgs) = 1

# One argument, because the scalar ABI carries one — a function of several is a
# function of a tuple the caller builds, and there is nowhere on the boundary
# to put the rest.
function _argfun_kind(prefix, args)
    length(args) == 1 || throw(
        ArgumentError(
            "`$prefix`: `argfun` is called with exactly one argument across the " *
            "C boundary — got $(length(args)). Give it one value (a case file " *
            "path, a size) and derive the rest inside the function.",
        ),
    )
    a = only(args)
    a isa AbstractString && return :str
    a isa Integer && return :int
    throw(
        ArgumentError(
            "`$prefix`: the example argument for `argfun` is a $(typeof(a)); the " *
            "C boundary carries one string or one 64-bit integer. Anything " *
            "richer belongs INSIDE the function — that is what it is for.",
        ),
    )
end

# `argfun` spelled as source. It must be a named function a package owns: the
# generated library names it, so an anonymous function or a closure has nothing
# to name, and would also defeat the static resolution above.
function _argfun_call(prefix, f)
    m = parentmodule(f)
    n = nameof(f)
    ok = try
        isdefined(m, n) && getfield(m, n) === f && Base.moduleroot(m) === m &&
            _owning_package(m) !== nothing
    catch
        false
    end
    ok || throw(
        ArgumentError(
            "`$prefix`: `argfun` must be a named function defined by a package — " *
            "the generated library calls it by name, from another process, so a " *
            "function defined in a script or the REPL cannot be reached. Got " *
            "$(repr(f)) in module $m; define it at the top level of a package " *
            "and pass that.",
        ),
    )
    return string(nameof(m), ".", n)
end

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

# ── Generating one model's instantiation surface ─────────────────────────────
#
# A fixed or one-integer model gets `P_nargs` + `P_new(n)`.  Everything else
# gets the schema + builder surface (ABI v2) and NO `P_new` — the consumers
# rely on that disjointness to route a lone integer.  All storage is
# concretely typed from the example values, so `--trim=safe` sees no dynamic
# containers.

function _instantiation_source(p::AbstractString, field::Union{Nothing, Symbol, FixedModel})
    argexpr = _arg_expr(field)
    return """
    # How many instantiation arguments `$(p)_new` consumes (0 = fixed model,
    # `n` is ignored). Lets a consumer decide whether `args` are required
    # before instantiating anything.
    Base.@ccallable function $(p)_nargs()::Cint
        return Cint($(_nargs_value(field)))
    end

$(_argkind_source(p, field))

    Base.@ccallable function $(p)_new(n::Cint)::Cint
        try
            push!(MODELS_$p, ExaModels.ExaModel(CORE_$p, $argexpr; check = Val(false)))
            return Cint(length(MODELS_$p))
        catch
            return Cint(0)          # 0 is the failure value for _new
        end
    end
"""
end

# How a slot is read back out of a builder when the model is instantiated: a
# table reassembles into the example's row type, column by column.
function _slot_expr(f::Field)
    f.kind == "table" || return "B.$(f.name)"
    row = join(("$(c.first) = B.$(f.name)_$(c.first)[_k]" for c in f.columns), ", ")
    return "[(; $row) for _k in eachindex(B.$(f.name)_$(first(f.columns).first))]"
end

# One `if` arm per field a setter can legitimately name; a name that matches
# no arm is status 1, the caller's error, not ours.
function _setter_arms(fields, render)
    isempty(fields) && return ""
    return join((render(f) for f in fields), "") * "\n"
end

# `_argkind` is emitted for every model, whatever its surface: `_nargs` says
# how many values are needed and cannot say what SHAPE they are, and a consumer
# handed only a library path needs both. Routing on it beats probing for
# symbols, which is what a consumer had to do before.
_argkind_source(p::AbstractString, field) = """
    # Which entry point instantiates this model: 0 fixed, 1 `$(p)_new(n)`,
    # 2 `$(p)_new_str(const char *)`, 3 the `$(p)_data_begin` builder.
    Base.@ccallable function $(p)_argkind()::Cint
        return Cint($(_argkind_value(field)))
    end
"""

# The argument-function surface. `unsafe_string` copies out of the caller's
# buffer, so the library never holds a pointer it does not own.
function _instantiation_source(p::AbstractString, u::UserArgs{:str})
    return """
    Base.@ccallable function $(p)_nargs()::Cint
        return Cint(1)
    end

$(_argkind_source(p, u))
    # This model instantiates from a string; `$(p)_new(n)` is not its entry
    # point, so it returns the documented failure value rather than building
    # something.
    Base.@ccallable function $(p)_new(n::Cint)::Cint
        return Cint(0)
    end

    Base.@ccallable function $(p)_new_str(s_ptr::Ptr{Cchar})::Cint
        try
            s = unsafe_string(s_ptr)
            push!(MODELS_$p, ExaModels.ExaModel(CORE_$p, $(u.call)(s)...; check = Val(false)))
            return Cint(length(MODELS_$p))
        catch
            return Cint(0)
        end
    end
"""
end

function _instantiation_source(p::AbstractString, u::UserArgs{:int})
    return """
    Base.@ccallable function $(p)_nargs()::Cint
        return Cint(1)
    end

$(_argkind_source(p, u))
    Base.@ccallable function $(p)_new(n::Cint)::Cint
        try
            push!(MODELS_$p, ExaModels.ExaModel(CORE_$p, $(u.call)(Int(n))...; check = Val(false)))
            return Cint(length(MODELS_$p))
        catch
            return Cint(0)
        end
    end
"""
end

function _instantiation_source(p::AbstractString, bm::BuilderModel)
    fields = bm.fields
    byname = Dict(f.name => f for f in fields)
    slots = [sl for f in fields for sl in _slots(f)]
    json = _schema_json(fields)

    decls = join(("        $s::$t\n        $(s)_set::Bool\n" for (s, t, _) in slots))
    zeros = join(("$z, false" for (_, _, z) in slots), ", ")
    flags = join(("B.$(s)_set" for (s, _, _) in slots), " && ")

    scalar(f) = """
            if f == $(repr(f.name))
                B.$(f.name) = v
                B.$(f.name)_set = true
                return Cint(0)
            end
    """
    array(f) = """
            if f == $(repr(f.name))
                B.$(f.name) = copy(unsafe_wrap(Array, v, Int(len)))
                B.$(f.name)_set = true
                return Cint(0)
            end
    """
    col(f, c) = """
            if t == $(repr(f.name)) && c == $(repr(c.first))
                B.$(f.name)_$(c.first) = copy(unsafe_wrap(Array, v, Int(len)))
                B.$(f.name)_$(c.first)_set = true
                return Cint(0)
            end
    """
    pick(kind, type) = [f for f in fields if f.kind == kind && f.type == type]
    cols(type) = [
        (f, c) for f in fields if f.kind == "table" for c in f.columns if c.second == type
    ]

    # Table columns must agree in length before rows can be reassembled.
    samelen = join(
        (
            "        (" *
            join(
                ("length(B.$(f.name)_$(c.first)) == length(B.$(f.name)_$(first(f.columns).first))"
                 for c in f.columns[2:end]),
                " && ",
            ) *
            ") || return Cint(0)\n"
            for f in fields if f.kind == "table" && length(f.columns) > 1
        ),
    )

    asm = join(
        (
            spec isa String ? _slot_expr(byname[spec]) :
            "(; " * join(("$(k) = $(_slot_expr(byname[n]))" for (k, n) in spec), ", ") * ")"
            for spec in bm.argspec
        ),
        ",\n                ",
    )

    return """
$(_argkind_source(p, bm))
    # ── builder for `$p` (schema + typed setters, ABI v2) ────────────────────

    const SCHEMA_$p = Vector{UInt8}($(repr(json)))

    # Returns the schema's byte length; copies what fits in `cap`.
    Base.@ccallable function $(p)_schema(buf::Ptr{UInt8}, cap::Cint)::Cint
        n = length(SCHEMA_$p)
        k = min(Int(cap), n)
        if k > 0 && buf != Ptr{UInt8}(0)
            GC.@preserve SCHEMA_$p unsafe_copyto!(buf, pointer(SCHEMA_$p), k)
        end
        return Cint(n)
    end

    # One concretely-typed slot per scalar/array field and per table column;
    # the `_set` flags are what make completeness checkable without sentinels.
    mutable struct Builder_$p
$decls    end
    Builder_$p() = Builder_$p($zeros)
    const BUILDERS_$p = Builder_$p[]

    @inline _bvalid_$p(b::Cint) = 1 <= b <= length(BUILDERS_$p)

    Base.@ccallable function $(p)_data_begin()::Cint
        try
            push!(BUILDERS_$p, Builder_$p())
            return Cint(length(BUILDERS_$p))
        catch
            return Cint(0)
        end
    end

    Base.@ccallable function $(p)_set_scalar_i64(b::Cint, field::Ptr{UInt8}, v::Clonglong)::Cint
        _bvalid_$p(b) || return Cint(1)
        try
            f = unsafe_string(field)
            B = BUILDERS_$p[Int(b)]
$(_setter_arms(pick("scalar", "i64"), scalar))            return Cint(1)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_set_scalar_f64(b::Cint, field::Ptr{UInt8}, v::Cdouble)::Cint
        _bvalid_$p(b) || return Cint(1)
        try
            f = unsafe_string(field)
            B = BUILDERS_$p[Int(b)]
$(_setter_arms(pick("scalar", "f64"), scalar))            return Cint(1)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_set_array_i64(
        b::Cint, field::Ptr{UInt8}, v::Ptr{Clonglong}, len::Cint,
    )::Cint
        _bvalid_$p(b) || return Cint(1)
        try
            f = unsafe_string(field)
            B = BUILDERS_$p[Int(b)]
$(_setter_arms(pick("array", "i64"), array))            return Cint(1)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_set_array_f64(
        b::Cint, field::Ptr{UInt8}, v::Ptr{Cdouble}, len::Cint,
    )::Cint
        _bvalid_$p(b) || return Cint(1)
        try
            f = unsafe_string(field)
            B = BUILDERS_$p[Int(b)]
$(_setter_arms(pick("array", "f64"), array))            return Cint(1)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_set_col_i64(
        b::Cint, table::Ptr{UInt8}, column::Ptr{UInt8}, v::Ptr{Clonglong}, len::Cint,
    )::Cint
        _bvalid_$p(b) || return Cint(1)
        try
            t = unsafe_string(table)
            c = unsafe_string(column)
            B = BUILDERS_$p[Int(b)]
$(_setter_arms(cols("i64"), fc -> col(fc...)))            return Cint(1)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_set_col_f64(
        b::Cint, table::Ptr{UInt8}, column::Ptr{UInt8}, v::Ptr{Cdouble}, len::Cint,
    )::Cint
        _bvalid_$p(b) || return Cint(1)
        try
            t = unsafe_string(table)
            c = unsafe_string(column)
            B = BUILDERS_$p[Int(b)]
$(_setter_arms(cols("f64"), fc -> col(fc...)))            return Cint(1)
        catch
            return Cint(2)
        end
    end

    # 1 iff every field is set and every table's columns agree in length.
    Base.@ccallable function $(p)_data_ready(b::Cint)::Cint
        _bvalid_$p(b) || return Cint(0)
        B = BUILDERS_$p[Int(b)]
        ($flags) || return Cint(0)
$samelen        return Cint(1)
    end

    Base.@ccallable function $(p)_new_from_data(b::Cint)::Cint
        $(p)_data_ready(b) == Cint(1) || return Cint(0)
        try
            B = BUILDERS_$p[Int(b)]
            push!(MODELS_$p, ExaModels.ExaModel(
                CORE_$p,
                $asm;
                check = Val(false),
            ))
            return Cint(length(MODELS_$p))
        catch
            return Cint(0)
        end
    end

    # Informative — there is no `$(p)_new` here; the consumers bind one value
    # per schema field, positionally, and instantiate through the builder.
    Base.@ccallable function $(p)_nargs()::Cint
        return Cint($(length(fields)))
    end
"""
end

function _module_source(modname::AbstractString, specs::Vector{ModelSpec}, pkgs = _Pkg[])
    # Imported for their side effect on `Serialization`: a module has to be
    # loaded before a type it owns can be resolved by `PkgId`, and the
    # deserialize below is what needs them.  A dependency alone is not enough.
    imports = join(("import $(p.name)\n" for p in pkgs))
    return """
    module $modname

    import ExaModels
    import Serialization
    $imports
    $(join((_model_source(s) for s in specs), "\n"))
    end # module $modname
    """
end

# One model's share of the generated module. Every piece of per-model state
# carries the prefix in its name, which is what lets any number of models
# coexist in the one module: separate cores, separate instance tables,
# separate ids. The entry points are `Base.@ccallable`, and juliac's
# `add_ccallables` picks up all of them regardless of how many there are.
function _model_source(s::ModelSpec)
    p = s.prefix
    # A builder spec's example is the whole tuple of values, splatted back the
    # way `ExaModel` takes them; the other forms carry a single value.
    example = (s.field isa BuilderModel || s.field isa UserArgs) ? "ARG0_$p..." : "ARG0_$p"
    return """
    # ── model `$p` ───────────────────────────────────────────────────────────

    # Deserialized at precompile time, so the core is baked into the package
    # image and no model-building code enters the compiled call graph.
    const CORE_$p = Serialization.deserialize(joinpath(@__DIR__, "core_$p.jls"))
    const ARG0_$p = Serialization.deserialize(joinpath(@__DIR__, "arg_$p.jls"))

    # Building one model at precompile time fixes the concrete instance type.
    # Every runtime instantiation differs only in sizes, so they all land in
    # this vector without widening it.  `check = Val(false)` drops the
    # placeholder-leak guard, which walks types reflectively and is not
    # trimmable — the check already ran, on this exact core, in the process
    # that called `compile_library`.
    const MODELS_$p = typeof(ExaModels.ExaModel(CORE_$p, $example; check = Val(false)))[]

    @inline _valid_$p(id::Cint) = 1 <= id <= length(MODELS_$p)

$(_instantiation_source(p, s.field))
    Base.@ccallable function $(p)_nvar(id::Cint)::Cint
        _valid_$p(id) || return Cint(-1)
        return Cint(MODELS_$p[Int(id)].meta.nvar)
    end

    Base.@ccallable function $(p)_ncon(id::Cint)::Cint
        _valid_$p(id) || return Cint(-1)
        return Cint(MODELS_$p[Int(id)].meta.ncon)
    end

    Base.@ccallable function $(p)_nnzj(id::Cint)::Cint
        _valid_$p(id) || return Cint(-1)
        return Cint(MODELS_$p[Int(id)].meta.nnzj)
    end

    Base.@ccallable function $(p)_nnzh(id::Cint)::Cint
        _valid_$p(id) || return Cint(-1)
        return Cint(MODELS_$p[Int(id)].meta.nnzh)
    end

    Base.@ccallable function $(p)_meta(
        id::Cint,
        x0::Ptr{Cdouble},
        lvar::Ptr{Cdouble},
        uvar::Ptr{Cdouble},
        lcon::Ptr{Cdouble},
        ucon::Ptr{Cdouble},
    )::Cint
        _valid_$p(id) || return Cint(1)
        try
            m = MODELS_$p[Int(id)]
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
        _valid_$p(id) || return Cint(1)
        try
            m = MODELS_$p[Int(id)]
            unsafe_store!(out, ExaModels.obj(m, unsafe_wrap(Array, x, m.meta.nvar)))
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_grad(id::Cint, x::Ptr{Cdouble}, g::Ptr{Cdouble})::Cint
        _valid_$p(id) || return Cint(1)
        try
            m = MODELS_$p[Int(id)]
            n = m.meta.nvar
            ExaModels.grad!(m, unsafe_wrap(Array, x, n), unsafe_wrap(Array, g, n))
            return Cint(0)
        catch
            return Cint(2)
        end
    end

    Base.@ccallable function $(p)_cons(id::Cint, x::Ptr{Cdouble}, c::Ptr{Cdouble})::Cint
        _valid_$p(id) || return Cint(1)
        try
            m = MODELS_$p[Int(id)]
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
        _valid_$p(id) || return Cint(1)
        try
            m = MODELS_$p[Int(id)]
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
        _valid_$p(id) || return Cint(1)
        try
            m = MODELS_$p[Int(id)]
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
        _valid_$p(id) || return Cint(1)
        try
            m = MODELS_$p[Int(id)]
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
        _valid_$p(id) || return Cint(1)
        try
            m = MODELS_$p[Int(id)]
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

    """
end

# ── Driving juliac ────────────────────────────────────────────────────────────

function _drive_juliac(appdir, libname, out, trim, bundle, verbose)
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
    return _link(Val(bundle), img, libname, outdir)
end

const _DLEXT = Base.BinaryPlatforms.platform_dlext()

# ── bundled: carries a privatized runtime; loadable from anything ────────────
function _link(::Val{true}, img, libname, outdir)
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
        outname = joinpath(outdir, "lib" * libname),
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
    libpath = joinpath(outdir, libroot, "lib" * libname * "." * _DLEXT)
    isfile(libpath) || error("juliac produced no library at $libpath")
    return (; libpath, outdir, libname)
end

# ── unbundled: one file, linked against the installed Julia ─────────────────
function _link(::Val{false}, img, libname, outdir)
    # Not cleared: without a bundle each model is a single file, so a directory
    # may hold several and wiping it would delete a sibling.
    libpath = joinpath(outdir, "lib" * libname * "." * _DLEXT)
    link = JuliaC.LinkRecipe(
        image_recipe = img,
        outname = splitext(libpath)[1],
        rpath = JuliaC.RPATH_JULIA,   # absolute paths into the Julia installation
    )
    JuliaC.link_products(link)
    isfile(libpath) || error("juliac produced no library at $libpath")
    return (; libpath, outdir, libname)
end

end # module ExaModelsC
