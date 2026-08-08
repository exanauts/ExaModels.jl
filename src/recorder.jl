# ── ExaTape: record/replay of model construction ──────────────────────────────
#
# Records the sequence of `add_*` calls made against a stand-in core (ExaTape)
# with a stand-in data object (DataTracer), so that model construction can be
# replayed later against real data through the real ExaCore API. The user's
# model-building code runs only at record time (ordinary dynamic Julia);
# replay is a type-stable fold over a concretely-typed tape, suitable for
# `juliac --trim=safe`. Design notes: docs/design/recorder.md.

# ── Tracer IR ─────────────────────────────────────────────────────────────────

"""
    TracerValue{T}

Abstract supertype for record-time stand-ins for values that only become
available at replay time (fields of the data named tuple and computations on
them). `T` is the concrete type the value will have at replay time.
"""
abstract type TracerValue{T} end

@inline replay_type(::Type{<:TracerValue{T}}) where {T} = T
@inline replay_type(::TracerValue{T}) where {T} = T
@inline replay_type(::Type{T}) where {T} = T   # plain types stand for themselves
@inline replay_type(x) = typeof(x)             # plain values stand for themselves

"""
    DataField{T, name} <: TracerValue{T}

Tracer referencing field `name` of the data named tuple, of replay-time type
`T`. Created by `getproperty` on a [`DataTracer`](@ref).
"""
struct DataField{T, name} <: TracerValue{T} end

"""
    TracerExpr{T, F, A} <: TracerValue{T}

Tracer for a deferred computation `f(args...)` over tracer values and plain
constants, with replay-time result type `T`.
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
instead of a value. Passed to the user's build function by [`record`](@ref).
"""
struct DataTracer{NT} end
DataTracer(::NT) where {NT <: NamedTuple} = DataTracer{NT}()

@inline Base.getproperty(::DataTracer{NT}, s::Symbol) where {NT} =
    DataField{fieldtype(NT, s), s}()

"""
    resolve(x, data)

Evaluate a tracer value against the actual data named tuple. Plain (non-tracer)
values resolve to themselves.
"""
@inline resolve(x, data) = x
@inline resolve(::DataField{T, name}, data) where {T, name} = getfield(data, name)::T
@inline resolve(t::TracerExpr{T}, data) where {T} =
    (t.f(map(a -> resolve(a, data), t.args)...))::T
# Multi-dimensional generators iterate a ProductIterator whose component
# ranges may be traced; rebuild it with the components resolved.
@inline resolve(p::Iterators.ProductIterator, data) =
    Iterators.product(map(r -> resolve(r, data), p.iterators)...)

@inline function _record_op(f::F, args...) where {F}
    T = Base.promote_op(f, map(replay_type, args)...)
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
# collect to replay time; the body may itself capture tracers (`k / data.nh`),
# in which case each element is a tracer expression that gets resolved.
"""
    DeferredCollect{ET, F, I} <: TracerValue{Vector{ET}}

Tracer for a comprehension whose range (and possibly body) is traced. At
replay time the range is resolved, the body runs per element, and any tracer
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
        replay_type(typeof(g.f(one(eltype(replay_type(I))))))
    catch
        replay_type(Base.promote_op(g.f, eltype(replay_type(I))))
    end
    DeferredCollect{ET, typeof(g.f), I}(g.f, g.iter)
end

@inline resolve(d::DeferredCollect{ET}, data) where {ET} =
    ET[resolve(d.f(x), data) for x in resolve(d.iter, data)]

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
Compute structural constants outside `record`, or extend the tracer op set if \
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
        "at replay time.",
    ),
)

# ── Tape handles ──────────────────────────────────────────────────────────────

"""
    TapeVar{V}

Record-time stand-in for a [`Variable`](@ref), returned by `add_var` on an
[`ExaTape`](@ref). Holds an initially-empty `Ref` that [`replay`](@ref) binds
to the real `Variable`; indexing delegates to the bound variable, so closures
that captured a `TapeVar` trace correctly inside the real `add_con`/`add_obj`.
"""
struct TapeVar{V}
    ref::Base.RefValue{V}
end
@inline Base.getindex(v::TapeVar, i...) = getindex(v.ref[], i...)

"""
    TapePar{P}

Record-time stand-in for a [`Parameter`](@ref), returned by `add_par` on an
[`ExaTape`](@ref). Same `Ref`-binding mechanism as [`TapeVar`](@ref).
"""
struct TapePar{P}
    ref::Base.RefValue{P}
end
@inline Base.getindex(p::TapePar, i...) = getindex(p.ref[], i...)

"""
    TapeCon{K}

Record-time stand-in for a [`Constraint`](@ref) created by the `K`-th tape
entry. Constraint handles are only ever used as explicit arguments to
`add_con!` (never inside traced closures), so instead of a `Ref` — whose
concrete type would need the traced `SIMDFunction` type, unknowable at record
time — replay threads a positional tuple of realized handles and `add_con!`
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

"""
    ExaTape()

Record-time stand-in for an [`ExaCore`](@ref). `add_var`, `add_con`, and
`add_obj` (and their macro forms) dispatch on it and record their arguments
into a concretely-typed entry tuple instead of building a model, threading the
tape through `(c, x) = add_var(c, ...)` exactly like the real API. Produce one
with [`record`](@ref) and turn it into a real core with [`replay`](@ref).
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
    V = Variable{Tuple{map(replay_type, ns)...}, Int, typeof(tag)}
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
    P = Parameter{Tuple{map(replay_type, ns)...}, Int, typeof(tag)}
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

# ── record ────────────────────────────────────────────────────────────────────

"""
    record(build, template::NamedTuple; minimize = true) -> ExaTape

Run `build(tape::ExaTape, data::DataTracer)` once and return the resulting
tape. `template` supplies only the *schema* (field names and types) of the
data; its values are never read. `build` must thread the tape exactly as it
would thread an `ExaCore` and return it. Core configuration that the direct
API takes on the `ExaCore` constructor (`minimize`) is recorded on the tape;
element type and backend, by contrast, are chosen at [`replay`](@ref) time.

## Example

```julia
tape = record((; N = 4)) do c, data
    @add_var(c, x, data.N; start = ((i % 2 == 1 ? -1.2 : 1.0) for i = 1:data.N))
    @add_con(c, 3x[i+1]^3 + 2x[i+2] - 5 for i = 1:data.N-2)
    @add_obj(c, 100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:data.N)
    c
end
core = replay(tape, (; N = 1000))
model = ExaModel(core)
```
"""
function record(build, template::NamedTuple; minimize = true)
    tape = build(ExaTape(; minimize = minimize), DataTracer(template))
    tape isa ExaTape || throw(
        ArgumentError(
            "the build function passed to `record` must return the tape it was " *
            "given (got $(typeof(tape)))",
        ),
    )
    return tape
end

# ── replay ────────────────────────────────────────────────────────────────────

"""
    replay(tape::ExaTape, data::NamedTuple; T = Float64, backend = nothing) -> ExaCore

Rebuild a real [`ExaCore`](@ref) by folding over the recorded entries and
making the real `add_var`/`add_con`/`add_obj` calls with all tracer values
resolved against `data`. Element type and backend are chosen here, not at
record time. The fold is fully type-inferrable (gated by `@inferred` in
`test/RecorderTest`).

A tape may be replayed any number of times, but not concurrently from multiple
threads: replay binds each recorded variable handle by mutating its `Ref`.
"""
function replay(
    tape::ExaTape,
    data::NamedTuple;
    T::Type{<:AbstractFloat} = Float64,
    backend = nothing,
)
    c = ExaCore(T; backend = backend, minimize = tape.config.minimize, concrete = Val(true))
    return _replay(c, (), data, tape.entries...)
end

# The fold threads a positional tuple of realized handles (one slot per entry;
# `nothing` for entries whose handles bind through Refs) so that ConAugEntry{K}
# can look up its target Constraint type-stably.
@inline _replay(c::ExaCore, handles, data) = c
@inline function _replay(c::ExaCore, handles, data, entry, rest...)
    c, h = _replay_entry(c, handles, entry, data)
    return _replay(c, (handles..., h), data, rest...)
end

# Optional-kwarg resolution: `nothing` means "not given at record time" and
# falls back to the real API's default; generators get their (possibly traced)
# iterable resolved.
@inline _kw(::Nothing, data, default) = default
@inline _kw(x, data, default) = _resolve_arg(x, data)
@inline _resolve_arg(g::Base.Generator, data) = Base.Generator(g.f, resolve(g.iter, data))
@inline _resolve_arg(x, data) = resolve(x, data)

@inline function _replay_entry(c::ExaCore{T}, handles, e::VarEntry, data) where {T}
    dims = map(d -> resolve(d, data), e.dims)
    c, v = add_var(
        c,
        dims...;
        tag = e.tag,
        name = e.name,
        start = _kw(e.start, data, zero(T)),
        lvar = _kw(e.lvar, data, T(-Inf)),
        uvar = _kw(e.uvar, data, T(Inf)),
    )
    e.var.ref[] = v
    return (c, nothing)
end

@inline function _replay_entry(c::ExaCore{T}, handles, e::ParEntry, data) where {T}
    dims = map(d -> resolve(d, data), e.dims)
    c, p = add_par(
        c,
        dims...;
        tag = e.tag,
        name = e.name,
        value = _kw(e.value, data, zero(T)),
    )
    e.par.ref[] = p
    return (c, nothing)
end

@inline function _replay_entry(c::ExaCore{T}, handles, e::ConEntry, data) where {T}
    gen = Base.Generator(e.f, resolve(e.itr, data))
    c, con = add_con(
        c,
        gen;
        tag = e.tag,
        name = e.name,
        start = _kw(e.start, data, zero(T)),
        lcon = _kw(e.lcon, data, zero(T)),
        ucon = _kw(e.ucon, data, zero(T)),
    )
    return (c, con)
end

@inline function _replay_entry(c::ExaCore{T}, handles, e::ConAugEntry{K}, data) where {T, K}
    gen = Base.Generator(e.f, resolve(e.itr, data))
    c, _ = add_con!(c, handles[K], gen; tag = e.tag)
    return (c, nothing)
end

@inline function _replay_entry(c::ExaCore{T}, handles, e::ObjEntry, data) where {T}
    gen = Base.Generator(e.f, resolve(e.itr, data))
    c, _ = add_obj(c, gen; name = e.name)
    return (c, nothing)
end
