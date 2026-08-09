# ── The args sentinel ─────────────────────────────────────────────────────────
#
# `DataTracer(template)` produces typed symbolic stand-ins for instance data:
# `data.N` is a value the model's structure may not branch on, but which may
# occupy any *value* slot of the core — dimensions, ranges, bounds, starts,
# parameter values — with resolution deferred to `ExaModel(core, args)`.
# Ported from the ss/recorder tape (the vocabulary survives; the tape's
# entry/handle machinery does not — value slots live in the core itself).

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

