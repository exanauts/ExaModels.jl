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
@inline replay_type(x) = typeof(x)   # plain values stand for themselves

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

@inline function _record_op(f::F, args...) where {F}
    T = Base.promote_op(f, map(replay_type, args)...)
    TracerExpr{T}(f, args)
end

for op in (:+, :-, :*)
    @eval begin
        @inline Base.$op(a::TracerValue, b::TracerValue) = _record_op($op, a, b)
        @inline Base.$op(a::TracerValue, b::Number) = _record_op($op, a, b)
        @inline Base.$op(a::Number, b::TracerValue) = _record_op($op, a, b)
    end
end
@inline Base.:-(a::TracerValue) = _record_op(-, a)

@inline Base.:(:)(a::TracerValue{<:Integer}, b::TracerValue{<:Integer}) = _record_op(:, a, b)
@inline Base.:(:)(a::Integer, b::TracerValue{<:Integer}) = _record_op(:, a, b)
@inline Base.:(:)(a::TracerValue{<:Integer}, b::Integer) = _record_op(:, a, b)

@inline Base.length(t::TracerValue{<:Union{AbstractArray, AbstractRange}}) =
    _record_op(length, t)

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

struct TapeCon end
struct TapeObj end

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

struct ConEntry{F, I, St, Lc, Uc, Nm, Tg}
    f::F
    itr::I
    start::St
    lcon::Lc
    ucon::Uc
    name::Nm
    tag::Tg
end

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
struct ExaTape{E}
    entries::E
end
ExaTape() = ExaTape(())

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
    (ExaTape((tape.entries..., entry)), var)
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
    (ExaTape((tape.entries..., entry)), TapeCon())
end

@inline function add_obj(tape::ExaTape, gen::Base.Generator; name = nothing)
    entry = ObjEntry(gen.f, gen.iter, name)
    (ExaTape((tape.entries..., entry)), TapeObj())
end

# ── record ────────────────────────────────────────────────────────────────────

"""
    record(build, template::NamedTuple) -> ExaTape

Run `build(tape::ExaTape, data::DataTracer)` once and return the resulting
tape. `template` supplies only the *schema* (field names and types) of the
data; its values are never read. `build` must thread the tape exactly as it
would thread an `ExaCore` and return it.

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
function record(build, template::NamedTuple)
    tape = build(ExaTape(), DataTracer(template))
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
    c = ExaCore(T; backend = backend, concrete = Val(true))
    return _replay(c, data, tape.entries...)
end

@inline _replay(c::ExaCore, data) = c
@inline function _replay(c::ExaCore, data, entry, rest...)
    return _replay(_replay_entry(c, entry, data), data, rest...)
end

# Optional-kwarg resolution: `nothing` means "not given at record time" and
# falls back to the real API's default; generators get their (possibly traced)
# iterable resolved.
@inline _kw(::Nothing, data, default) = default
@inline _kw(x, data, default) = _resolve_arg(x, data)
@inline _resolve_arg(g::Base.Generator, data) = Base.Generator(g.f, resolve(g.iter, data))
@inline _resolve_arg(x, data) = resolve(x, data)

@inline function _replay_entry(c::ExaCore{T}, e::VarEntry, data) where {T}
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
    return c
end

@inline function _replay_entry(c::ExaCore{T}, e::ConEntry, data) where {T}
    gen = Base.Generator(e.f, resolve(e.itr, data))
    c, _ = add_con(
        c,
        gen;
        tag = e.tag,
        name = e.name,
        start = _kw(e.start, data, zero(T)),
        lcon = _kw(e.lcon, data, zero(T)),
        ucon = _kw(e.ucon, data, zero(T)),
    )
    return c
end

@inline function _replay_entry(c::ExaCore{T}, e::ObjEntry, data) where {T}
    gen = Base.Generator(e.f, resolve(e.itr, data))
    c, _ = add_obj(c, gen; name = e.name)
    return c
end
