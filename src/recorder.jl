# ── ExaTape: record/instantiate of model construction ──────────────────────────────
#
# Records the sequence of `add_*` calls made against a stand-in core (ExaTape)
# with a stand-in data object (DataTracer), so that model construction can be
# instantiated later against real data through the real ExaCore API. The user's
# model-building code runs only at record time (ordinary dynamic Julia);
# instantiate is a type-stable fold over a concretely-typed tape, suitable for
# `juliac --trim=safe`. Design notes: docs/design/recorder.md.

# ── Tracer IR ─────────────────────────────────────────────────────────────────

"""
    TracerValue{T}

Abstract supertype for record-time stand-ins for values that only become
available at instantiate time (fields of the data named tuple and computations on
them). `T` is the concrete type the value will have at instantiate time.
"""
abstract type TracerValue{T} end

@inline instantiate_type(::Type{<:TracerValue{T}}) where {T} = T
@inline instantiate_type(::TracerValue{T}) where {T} = T
@inline instantiate_type(::Type{T}) where {T} = T   # plain types stand for themselves
@inline instantiate_type(x) = typeof(x)             # plain values stand for themselves

"""
    DataField{T, name} <: TracerValue{T}

Tracer referencing field `name` of the data named tuple, of instantiate-time type
`T`. Created by `getproperty` on a [`DataTracer`](@ref).
"""
struct DataField{T, name} <: TracerValue{T} end

"""
    TracerExpr{T, F, A} <: TracerValue{T}

Tracer for a deferred computation `f(args...)` over tracer values and plain
constants, with instantiate-time result type `T`.
"""
struct TracerExpr{T, F, A} <: TracerValue{T}
    f::F
    args::A
end
@inline TracerExpr{T}(f::F, args::A) where {T, F, A} = TracerExpr{T, F, A}(f, args)

"""
    DataTracer(template::NamedTuple)

Record-time stand-in for the data named tuple. Field access (`data.N`) returns
a [`DataField`](@ref) tracer typed by the corresponding field of `template`,
instead of a value. Construct one from a template `NamedTuple` whose
field *types* define the schema.
"""
struct DataTracer{NT} end
DataTracer(::NT) where {NT <: NamedTuple} = DataTracer{NT}()

@inline Base.getproperty(::DataTracer{NT}, s::Symbol) where {NT} =
    DataField{fieldtype(NT, s), s}()

"""
    resolve(x, args)

Evaluate a tracer value against the instantiation `args`. Plain (non-tracer)
values resolve to themselves.
"""
@inline resolve(x, args) = x
@inline resolve(::DataField{T, name}, args::NamedTuple) where {T, name} =
    getfield(args, name)::T
# A bare value binds every data-field access — the ergonomic form for
# single-field schemas: `ExaModel(tape, 3)` ≡ `ExaModel(tape, (; N = 3))`.
@inline resolve(::DataField{T, name}, args) where {T, name} = convert(T, args)::T
@noinline _args_error(name) = throw(ArgumentError(
    "this tape references data field `$name`: instantiate with `args` — a " *
    "NamedTuple binding fields by name, or a bare value for a single-field schema"))
@inline resolve(::DataField{T, name}, ::Nothing) where {T, name} = _args_error(name)
# Positional tuples are refused: the tape records field names, not an order.
@inline resolve(::DataField{T, name}, ::Tuple) where {T, name} = throw(ArgumentError(
    "tuple args are positional, but a tape's schema binds by name: use (; $name = ...)"))
@inline resolve(t::TracerExpr{T}, args) where {T} =
    (t.f(map(a -> resolve(a, args), t.args)...))::T
# Multi-dimensional generators iterate a ProductIterator whose component
# ranges may be traced; rebuild it with the components resolved.
@inline resolve(p::Iterators.ProductIterator, args) =
    Iterators.product(map(r -> resolve(r, args), p.iterators)...)

@inline function _record_op(f::F, args...) where {F}
    T = Base.promote_op(f, map(instantiate_type, args)...)
    TracerExpr{T}(f, args)
end

for op in (:+, :-, :*, :div, :/, :rem, :mod, :max, :min)
    @eval begin
        @inline Base.$op(a::TracerValue, b::TracerValue) = _record_op($op, a, b)
        @inline Base.$op(a::TracerValue, b::Number) = _record_op($op, a, b)
        @inline Base.$op(a::Number, b::TracerValue) = _record_op($op, a, b)
    end
end
@inline Base.:-(a::TracerValue) = _record_op(-, a)
struct _RoundTo{T, F} end
@inline (::_RoundTo{T, F})(x) where {T, F} = F(T, x)
@inline Base.floor(::Type{T}, a::TracerValue) where {T} = _record_op(_RoundTo{T, floor}(), a)
@inline Base.ceil(::Type{T}, a::TracerValue) where {T} = _record_op(_RoundTo{T, ceil}(), a)

@inline Base.:(:)(a::TracerValue{<:Integer}, b::TracerValue{<:Integer}) = _record_op(:, a, b)
@inline Base.:(:)(a::Integer, b::TracerValue{<:Integer}) = _record_op(:, a, b)
@inline Base.:(:)(a::TracerValue{<:Integer}, b::Integer) = _record_op(:, a, b)
for (A, B, C) in (
    (:TracerValue, :TracerValue, :TracerValue), (:TracerValue, :TracerValue, :Integer),
    (:TracerValue, :Integer, :TracerValue), (:Integer, :TracerValue, :TracerValue),
    (:TracerValue, :Integer, :Integer), (:Integer, :TracerValue, :Integer),
    (:Integer, :Integer, :TracerValue),
)
    @eval @inline Base.:(:)(a::$A, s::$B, b::$C) = _record_op(:, a, s, b)
end

@inline Base.length(t::TracerValue{<:Union{AbstractArray, AbstractRange}}) =
    _record_op(length, t)
@inline Base.fill(v, t::TracerValue{<:Integer}) = _record_op(fill, v, t)

# A comprehension over a traced range (`[f(i) for i in 1:2:data.n]`) reaches
# `collect` with a generator whose iterable is a tracer. Defer the whole
# collect to instantiate time; the body may itself capture tracers (`k / data.nh`),
# in which case each element is a tracer expression that gets resolved.
"""
    DeferredCollect{ET, F, I} <: TracerValue{Vector{ET}}

Tracer for a comprehension whose range (and possibly body) is traced. At
instantiate time the range is resolved, the body runs per element, and any tracer
elements are resolved against the data.
"""
struct DeferredCollect{ET, F, I} <: TracerValue{Vector{ET}}
    f::F
    iter::I
end

@inline function Base.collect(g::Base.Generator{I}) where {I <: TracerValue}
    # Inference through the body is unreliable (the tracer ops compute their
    # result types dynamically), so determine the element type by probing the
    # body with a sample index — recording is a dynamic phase, so an actual
    # call is both allowed and exact. Falls back to inference if the body
    # cannot run at the sample.
    ET = try
        instantiate_type(typeof(g.f(one(eltype(instantiate_type(I))))))
    catch
        instantiate_type(Base.promote_op(g.f, eltype(instantiate_type(I))))
    end
    DeferredCollect{ET, typeof(g.f), I}(g.f, g.iter)
end

@inline resolve(d::DeferredCollect{ET}, args) where {ET} =
    ET[resolve(d.f(x), args) for x in resolve(d.iter, args)]

"""
    RecorderStructureError

Thrown at record time when the model-building code attempts an operation whose
result would freeze recording-time information into the tape (e.g. branching
on a data value, or iterating a traced value outside a generator passed to an
`add_*` call).
"""
struct RecorderStructureError <: Exception
    msg::String
end
Base.showerror(io::IO, e::RecorderStructureError) =
    print(io, "RecorderStructureError: ", e.msg)

const _STRUCTURE_MSG = "the structure of a recorded model cannot depend on data \
values: this operation would freeze the recording-time result into the tape. \
Compute structural constants before recording, or extend the tracer op set if \
this operation should be recordable."

for op in (:(==), :(<), :(<=), :(>), :(>=), :isless)
    @eval begin
        Base.$op(::TracerValue, ::Number) = throw(RecorderStructureError(_STRUCTURE_MSG))
        Base.$op(::Number, ::TracerValue) = throw(RecorderStructureError(_STRUCTURE_MSG))
        Base.$op(::TracerValue, ::TracerValue) = throw(RecorderStructureError(_STRUCTURE_MSG))
    end
end
Base.iterate(::TracerValue, state...) = throw(
    RecorderStructureError(
        "cannot iterate a traced value at record time. Pass generators over " *
        "traced ranges directly to add_var/add_con/add_obj — they are iterated " *
        "at instantiate time.",
    ),
)

# ── Tape handles ──────────────────────────────────────────────────────────────

"""
    TapeVar{V}

Record-time stand-in for a [`Variable`](@ref), returned by `add_var` on an
[`ExaTape`](@ref). Holds an initially-empty `Ref` that [`instantiate`](@ref) binds
to the real `Variable`; indexing delegates to the bound variable, so closures
that captured a `TapeVar` trace correctly inside the real `add_con`/`add_obj`.
"""
struct TapeVar{V}
    ref::Base.RefValue{V}
end

"""
    TapeVarIndexed / TapeParIndexed

Symbolic reference to an entry of a not-yet-bound [`TapeVar`](@ref) /
[`TapePar`](@ref), produced when a tape handle is indexed *outside* a instantiate
(i.e. while building an expression tree programmatically, e.g. from Python).
Instantiate rewrites these into real `Var` / `ParameterNode` nodes via
[`_rebind`](@ref) once the handles are bound.
"""
struct TapeVarIndexed{V, I} <: AbstractNode
    var::TapeVar{V}
    i::I
end

# Indexing a tape handle ALWAYS produces a symbolic reference — never a
# branch on binding state, which would make traced tree types a Union and
# destroy instantiate inferability. Instantiate resolves sentinels via _rebind.
@inline Base.getindex(v::TapeVar, i...) = TapeVarIndexed(v, i)

"""
    TapePar{P}

Record-time stand-in for a [`Parameter`](@ref), returned by `add_par` on an
[`ExaTape`](@ref). Same `Ref`-binding mechanism as [`TapeVar`](@ref).
"""
struct TapePar{P}
    ref::Base.RefValue{P}
end

struct TapeParIndexed{P, I} <: AbstractNode
    par::TapePar{P}
    i::I
end

@inline Base.getindex(p::TapePar, i...) = TapeParIndexed(p, i)

"""
    _rebind(node)

Rewrite an expression tree built at record time (with
[`TapeVarIndexed`](@ref)/[`TapeParIndexed`](@ref) sentinels) into a real
ExaModels tree by indexing the now-bound handles. Called during instantiate of
tree entries; structure and all other leaves are preserved.
"""
@inline _rebind(x) = x
@inline _rebind(n::TapeVarIndexed) = getindex(n.var.ref[], map(_rebind, n.i)...)
@inline _rebind(n::TapeParIndexed) = getindex(n.par.ref[], map(_rebind, n.i)...)
@inline _rebind(n::Node1{F, I}) where {F, I} = Node1(F.instance, _rebind(n.inner))
@inline _rebind(n::Node2{F, I1, I2}) where {F, I1, I2} =
    Node2(F.instance, _rebind(n.inner1), _rebind(n.inner2))
@inline _rebind(n::ParameterNode) = ParameterNode(_rebind(n.i))
@inline _rebind(n::DataIndexed{I, J}) where {I, J} = DataIndexed(_rebind(getfield(n, :inner)), J)
@inline _rebind(n::SumNode) = SumNode(map(_rebind, n.inners))
@inline _rebind(n::ProdNode) = ProdNode(map(_rebind, n.inners))
@inline _rebind(p::Pair) = _rebind(p.first) => _rebind(p.second)

"""
    FixedExpr{N}

A named single-argument functor returning a fixed, pre-built expression tree —
the sanctioned constant-generator vehicle (`Base.Generator(FixedExpr(node),
itr)`): the tree already references the iteration element through
`DataSource`, so the argument is ignored. Named (rather than an anonymous
closure) so tapes built from trees remain serializable.
"""
struct FixedExpr{N}
    node::N
end
@inline (f::FixedExpr)(_) = f.node

"""
    TapeCon{K}

Record-time stand-in for a [`Constraint`](@ref) created by the `K`-th tape
entry. Constraint handles are only ever used as explicit arguments to
`add_con!` (never inside traced closures), so instead of a `Ref` — whose
concrete type would need the traced `SIMDFunction` type, unknowable at record
time — instantiate threads a positional tuple of realized handles and `add_con!`
entries look theirs up by `K`.
"""
struct TapeCon{K} end
struct TapeObj end
struct TapeConAug end

# ── Tape entries ──────────────────────────────────────────────────────────────

struct VarEntry{D, St, Lv, Uv, Nm, Tg, R}
    dims::D
    start::St
    lvar::Lv
    uvar::Uv
    name::Nm
    tag::Tg
    var::R
end

struct ParEntry{D, V, Nm, Tg, R}
    dims::D
    value::V
    name::Nm
    tag::Tg
    par::R
end

struct ConEntry{F, I, St, Lc, Uc, Nm, Tg}
    f::F
    itr::I
    start::St
    lcon::Lc
    ucon::Uc
    name::Nm
    tag::Tg
end

struct ConAugEntry{K, F, I, Tg}
    f::F
    itr::I
    tag::Tg
end
@inline ConAugEntry{K}(f::F, itr::I, tag::Tg) where {K, F, I, Tg} =
    ConAugEntry{K, F, I, Tg}(f, itr, tag)

struct ObjEntry{F, I, Nm}
    f::F
    itr::I
    name::Nm
end

# Tree-based entries: the expression is a pre-built Node tree (with
# TapeVarIndexed/TapeParIndexed sentinels) instead of an uncalled closure —
# the Python-friendly path, since no Julia function needs to be written.
struct ConTreeEntry{E, I, St, Lc, Uc, Nm, Tg}
    expr::E
    itr::I
    start::St
    lcon::Lc
    ucon::Uc
    name::Nm
    tag::Tg
end

struct ObjTreeEntry{E, I, Nm}
    expr::E
    itr::I
    name::Nm
end

struct ConAugTreeEntry{K, E, I, Tg}
    expr::E   # a Pair: row_index_expr => expression
    itr::I
    tag::Tg
end
@inline ConAugTreeEntry{K}(expr::E, itr::I, tag::Tg) where {K, E, I, Tg} =
    ConAugTreeEntry{K, E, I, Tg}(expr, itr, tag)

"""
    ExaTape(; minimize = true)

Record-time stand-in for an [`ExaCore`](@ref). `add_var`, `add_con`, and
`add_obj` (and their macro forms) dispatch on it and record their arguments
into a concretely-typed entry tuple instead of building a model, threading the
tape through `(c, x) = add_var(c, ...)` exactly like the real API:

    data = DataTracer((; N = 4))     # schema template — values are never read
    tape = ExaTape()
    tape, x = add_var(tape, data.N)
    ...
    m = ExaModel(tape, (; N = 1000))
"""
struct ExaTape{E, C}
    entries::E
    config::C
end
ExaTape(; minimize = true) = ExaTape((), (; minimize = minimize))

@inline _append(tape::ExaTape, entry) = ExaTape((tape.entries..., entry), tape.config)

function Base.show(io::IO, tape::ExaTape)
    print(io, "An ExaTape with $(length(tape.entries)) recorded call")
    length(tape.entries) == 1 || print(io, "s")
end

# ── Recording methods (mirror the ExaCore signatures) ─────────────────────────

@inline function add_var(
    tape::ExaTape,
    ns...;
    tag = nothing,
    name = nothing,
    start = nothing,
    lvar = nothing,
    uvar = nothing,
)
    V = Variable{Tuple{map(instantiate_type, ns)...}, Int, typeof(tag)}
    var = TapeVar(Ref{V}())
    entry = VarEntry(ns, start, lvar, uvar, name, tag, var)
    (_append(tape, entry), var)
end

@inline function add_par(
    tape::ExaTape,
    ns...;
    tag = nothing,
    name = nothing,
    value = nothing,
)
    P = Parameter{Tuple{map(instantiate_type, ns)...}, Int, typeof(tag)}
    par = TapePar(Ref{P}())
    entry = ParEntry(ns, value, name, tag, par)
    (_append(tape, entry), par)
end

@inline function add_par(tape::ExaTape, value::AbstractArray; tag = nothing, name = nothing)
    add_par(tape, Base.size(value)...; tag = tag, name = name, value = value)
end

@inline function add_par(tape::ExaTape, value::TracerValue{<:AbstractArray}; tag = nothing, name = nothing)
    add_par(tape, length(value); tag = tag, name = name, value = value)
end

@inline function add_con(
    tape::ExaTape,
    gen::Base.Generator;
    tag = nothing,
    name = nothing,
    start = nothing,
    lcon = nothing,
    ucon = nothing,
)
    entry = ConEntry(gen.f, gen.iter, start, lcon, ucon, name, tag)
    (_append(tape, entry), TapeCon{length(tape.entries) + 1}())
end

@inline function add_con!(tape::ExaTape, ::TapeCon{K}, gen::Base.Generator; tag = nothing) where {K}
    entry = ConAugEntry{K}(gen.f, gen.iter, tag)
    (_append(tape, entry), TapeConAug())
end

@inline function add_obj(tape::ExaTape, gen::Base.Generator; name = nothing)
    entry = ObjEntry(gen.f, gen.iter, name)
    (_append(tape, entry), TapeObj())
end

# Tree forms, mirroring the real API's `add_con(c, expr::AbstractNode, pars)`
# and `add_obj(c, expr::AbstractNode, pars)`.
@inline function add_con(
    tape::ExaTape,
    expr::AbstractNode,
    itr = 1:1;
    tag = nothing,
    name = nothing,
    start = nothing,
    lcon = nothing,
    ucon = nothing,
)
    entry = ConTreeEntry(expr, itr, start, lcon, ucon, name, tag)
    (_append(tape, entry), TapeCon{length(tape.entries) + 1}())
end

@inline function add_obj(tape::ExaTape, expr::AbstractNode, itr = 1:1; name = nothing)
    entry = ObjTreeEntry(expr, itr, name)
    (_append(tape, entry), TapeObj())
end

# Tree form of constraint augmentation: `expr` is `row_index_expr => body`,
# both referencing the iteration element through DataSource.
@inline function add_con!(tape::ExaTape, ::TapeCon{K}, expr::Pair, itr; tag = nothing) where {K}
    entry = ConAugTreeEntry{K}(expr, itr, tag)
    (_append(tape, entry), TapeConAug())
end

# ── instantiate ────────────────────────────────────────────────────────────────────

"""
    ExaModel(tape::ExaTape, args = nothing; T = Float64, backend = nothing, kwargs...)

Instantiate `tape` at `args` and build the model in one call — the standard
way to turn a recorded tape into a solvable `ExaModel`. `args` has no
privileged shape: a `NamedTuple` binds data fields by name, a bare value
binds a single-field schema, and `nothing` (the default) instantiates a tape
that never touched the data tracer — in which case `ExaModel(tape)` builds
exactly the model the same calls against an `ExaCore` would:

    m = ExaModel(tape, (; N = 1000))
    m = ExaModel(tape, 1000)                 # single-field schema
    m = ExaModel(tape)                       # tape with no data references
    m = ExaModel(tape, args; T = Float32, backend = CUDABackend())

Element type and backend are chosen here; remaining keyword arguments are
passed to the `ExaModel` constructor. (The underlying two-step form,
`ExaModels.instantiate(tape, args) -> ExaCore`, remains available — unexported —
for workflows that need the intermediate core.)
"""
@inline function ExaModel(
    tape::ExaTape,
    args = nothing;
    T::Type{<:AbstractFloat} = Float64,
    backend = nothing,
    kwargs...,
)
    return ExaModel(_instantiate_impl(tape, args, T, backend); kwargs...)
end

"""
    instantiate(tape::ExaTape, args = nothing; T = Float64, backend = nothing) -> ExaCore

Rebuild a real [`ExaCore`](@ref) by folding over the recorded entries and
making the real `add_var`/`add_con`/`add_obj` calls with all tracer values
resolved against `args` (see [`ExaModel`](@ref) for the accepted shapes).
Element type and backend are chosen here, not at
record time. The fold is fully type-inferrable (gated by `@inferred` in
`test/RecorderTest`).

A tape may be instantiated any number of times, but not concurrently from multiple
threads: instantiate binds each recorded variable handle by mutating its `Ref`.
"""
function instantiate(
    tape::ExaTape,
    args = nothing;
    T::Type{<:AbstractFloat} = Float64,
    backend = nothing,
)
    return _instantiate_impl(tape, args, T, backend)
end

# Positional core: Type{T} dispatch keeps inference exact through the
# keyword seam (a Type-valued keyword argument loses its constant-ness in
# kwcall, which surfaces as an abstract ExaCore under juliac's verifier).
@inline function _instantiate_impl(tape::ExaTape, args, ::Type{T}, backend) where {T <: AbstractFloat}
    c = ExaCore(T; backend = backend, minimize = tape.config.minimize, concrete = Val(true))
    return _instantiate(c, (), args, tape.entries...)
end

# The fold threads a positional tuple of realized handles (one slot per entry;
# `nothing` for entries whose handles bind through Refs) so that ConAugEntry{K}
# can look up its target Constraint type-stably.
@inline _instantiate(c::ExaCore, handles, args) = c
@inline function _instantiate(c::ExaCore, handles, args, entry, rest...)
    c, h = _instantiate_entry(c, handles, entry, args)
    return _instantiate(c, (handles..., h), args, rest...)
end

# The low-level (tree, pars) forms do not run the generator path's
# _adapt_gen, so multi-dimensional index sets must be collected here.
@inline _collect_pars(p::Iterators.ProductIterator) = collect(p)
@inline _collect_pars(x) = x

# Optional-kwarg resolution: `nothing` means "not given at record time" and
# falls back to the real API's default; generators get their (possibly traced)
# iterable resolved.
@inline _kw(::Nothing, args, default) = default
@inline _kw(x, args, default) = _resolve_arg(x, args)
@inline _resolve_arg(g::Base.Generator, args) = Base.Generator(g.f, resolve(g.iter, args))
@inline _resolve_arg(x, args) = resolve(x, args)

@inline function _instantiate_entry(c::ExaCore{T}, handles, e::VarEntry, args) where {T}
    dims = map(d -> resolve(d, args), e.dims)
    c, v = add_var(
        c,
        dims...;
        tag = e.tag,
        name = e.name,
        start = _kw(e.start, args, zero(T)),
        lvar = _kw(e.lvar, args, T(-Inf)),
        uvar = _kw(e.uvar, args, T(Inf)),
    )
    e.var.ref[] = v
    return (c, nothing)
end

@inline function _instantiate_entry(c::ExaCore{T}, handles, e::ParEntry, args) where {T}
    dims = map(d -> resolve(d, args), e.dims)
    c, p = add_par(
        c,
        dims...;
        tag = e.tag,
        name = e.name,
        value = _kw(e.value, args, zero(T)),
    )
    e.par.ref[] = p
    return (c, nothing)
end

@inline function _instantiate_entry(c::ExaCore{T}, handles, e::ConEntry, args) where {T}
    c, con = add_con(
        c,
        _rebind(e.f(DataSource())),
        _collect_pars(resolve(e.itr, args));
        tag = e.tag,
        name = e.name,
        start = _kw(e.start, args, zero(T)),
        lcon = _kw(e.lcon, args, zero(T)),
        ucon = _kw(e.ucon, args, zero(T)),
    )
    return (c, con)
end

@inline function _instantiate_entry(c::ExaCore{T}, handles, e::ConAugEntry{K}, args) where {T, K}
    pair = _rebind(e.f(DataSource()))
    gen = Base.Generator(FixedExpr(pair), resolve(e.itr, args))
    c, _ = add_con!(c, handles[K], gen; tag = e.tag)
    return (c, nothing)
end

@inline function _instantiate_entry(c::ExaCore{T}, handles, e::ObjEntry, args) where {T}
    c, _ = add_obj(c, _rebind(e.f(DataSource())), _collect_pars(resolve(e.itr, args)); name = e.name)
    return (c, nothing)
end

@inline function _instantiate_entry(c::ExaCore{T}, handles, e::ConTreeEntry, args) where {T}
    c, con = add_con(
        c,
        _rebind(e.expr),
        _collect_pars(resolve(e.itr, args));
        tag = e.tag,
        name = e.name,
        start = _kw(e.start, args, zero(T)),
        lcon = _kw(e.lcon, args, zero(T)),
        ucon = _kw(e.ucon, args, zero(T)),
    )
    return (c, con)
end

@inline function _instantiate_entry(c::ExaCore{T}, handles, e::ConAugTreeEntry{K}, args) where {T, K}
    gen = Base.Generator(FixedExpr(_rebind(e.expr)), resolve(e.itr, args))
    c, _ = add_con!(c, handles[K], gen; tag = e.tag)
    return (c, nothing)
end

@inline function _instantiate_entry(c::ExaCore{T}, handles, e::ObjTreeEntry, args) where {T}
    c, _ = add_obj(c, _rebind(e.expr), _collect_pars(resolve(e.itr, args)); name = e.name)
    return (c, nothing)
end

# ── One-command shared-library compilation (ExaModelsJuliaC extension) ────────

"""
    compile_library(model_file; prefix = "rec", out = "lib_out",
                    template_n = 4, trim = "safe", privatize = true,
                    verbose = false) -> (; libpath, outdir)

Compile a recorded model into a self-contained shared library exposing the
NLP through a C interface, in one command. Requires `using JuliaC`
(implemented in the `ExaModelsJuliaC` package extension).

`model_file` is a Julia source file defining

- `build(c, args)` — the model, written against the tape exactly as against
  an `ExaCore` (it is passed to [`record`](@ref)), and
- `make_data(n::Integer)::NamedTuple` — the args for size `n` (also used at
  `template_n` as the recording schema).

The generated library exports, for the chosen `prefix` (C ABI: 1-based
indices, lower-triangle Lagrangian Hessian with `obj_weight`, `Cint` status
returns): `<prefix>_new(n) -> id` (any number of instances may coexist),
and id-first `<prefix>_nvar/_ncon/_nnzj/_nnzh`, `<prefix>_meta`,
`<prefix>_obj/_grad/_cons/_jac/_hess` and the two `_structure` functions —
the convention consumed by CNLPModels.jl. The tape
is recorded at the generated package's precompile time, so the compiled
call graph contains no user model code.
"""
function compile_library end

# Post-solve access through tape handles: after a instantiate, the handle's Ref
# points at the instantiated model's variable, so solution retrieval forwards to
# it (semantics: the LAST instantiate of this tape).
solution(result::SolverCore.AbstractExecutionStats, tv::TapeVar) =
    solution(result, tv.ref[])
