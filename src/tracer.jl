# ── The args sentinel ─────────────────────────────────────────────────────────
#
# `ArgTracer` mirrors `DataSource`: a bare node standing for the whole
# instantiation argument, usable directly or through `getproperty`/`getindex`
# (which produce `ArgIndexed` nodes, the analogue of `DataIndexed`).
# Arithmetic on arg nodes composes through the ordinary `Node1`/`Node2`
# machinery, so an expression like `args.N - 2` is a concretely-typed node
# tree. Where `DataSource` trees evaluate per element as `node(i, x, θ)`,
# arg trees evaluate once per instantiation as `resolve(node, args)`.
#
# Model *structure* may not depend on arg values: comparisons and iteration
# on arg nodes throw `RecorderStructureError` at construction.

"""
    ArgTracer <: AbstractNode

The instantiation-argument sentinel — the analogue of [`DataSource`](@ref)
for values that arrive at `ExaModel(core, args)` rather than per element.
Use it directly (the whole argument, e.g. a bare size) or index into it
(`args.N`, `args.branch`); either form may occupy any *value* slot of the
core — dimensions, ranges, bounds, starts, parameter values.
"""
struct ArgTracer <: AbstractNode end

"""
    ArgIndexed{I, J} <: AbstractNode

The lookup `inner.J` (or `inner[J]`) on an [`ArgTracer`](@ref) or another
`ArgIndexed` — the analogue of [`DataIndexed`](@ref). `J` is a type
parameter, so resolution is compile-time-addressed.
"""
struct ArgIndexed{I, J} <: AbstractNode
    inner::I
end

@inline ArgIndexed(inner::I, n) where {I} = ArgIndexed{I, n}(inner)
@inline ArgIndexed(inner::I, s::Constant{n}) where {I, n} = ArgIndexed{I, n}(inner)

@inline Base.getproperty(n::ArgTracer, s::Symbol) = ArgIndexed(n, s)
@inline Base.getindex(n::ArgTracer, i) = ArgIndexed(n, i)
@inline Base.indexed_iterate(n::ArgTracer, idx, start = 1) = (ArgIndexed(n, idx), idx + 1)
@inline Base.getproperty(v::ArgIndexed{I, n}, s::Symbol) where {I, n} = ArgIndexed(v, s)
@inline Base.getindex(v::ArgIndexed{I, n}, i) where {I, n} = ArgIndexed(v, i)
@inline Base.indexed_iterate(v::ArgIndexed{I, n}, idx, start = 1) where {I, n} =
    (ArgIndexed(v, idx), idx + 1)

# An arg node is a constant with respect to (i, x, θ): the sparsity/compressor
# pass probes trees by evaluating them, and an unresolved arg leaf behaves
# there exactly as `Constant` does (its value is irrelevant to structure —
# kernels only ever see materialized trees, where these leaves are numbers).
@inline (::ArgTracer)(i, x, θ) = 1
@inline (::ArgIndexed)(i, x, θ) = 1

# ── Resolution: evaluate an arg tree against the instantiation args ──────────

@inline _lookup(x, ::Val{J}) where {J} = J isa Symbol ? getproperty(x, J) : getindex(x, J)
@inline _lookup(x::NamedTuple, ::Val{J}) where {J} = getfield(x, J)

"""
    resolve(x, args)

Evaluate an arg-node tree against the instantiation `args`. Plain values
resolve to themselves; `ArgTracer` resolves to `args` (a NamedTuple binds
fields by name; a bare value binds the tracer used directly).
"""
@inline resolve(x, args) = x
@inline resolve(::ArgTracer, args) = args
@noinline _args_error(J) = throw(ArgumentError(
    "this core references instantiation argument `$J`: materialize with " *
    "`args` — a NamedTuple binding fields by name, or the bare value for a " *
    "tracer used directly"))
@inline resolve(::ArgTracer, ::Nothing) = _args_error(:args)
@inline resolve(v::ArgIndexed{I, J}, args) where {I, J} =
    _lookup(resolve(getfield(v, :inner), args), Val(J))
@inline resolve(v::ArgIndexed{I, J}, ::Nothing) where {I, J} = _args_error(J)
# An indexed reference (`args.N`) binds only by name: a bare value satisfies
# only the tracer used directly (`add_var(c, args)`), never a field lookup.
@inline resolve(v::ArgIndexed{I, J}, args::Number) where {I, J} = throw(ArgumentError(
    "this core references field `$J` of the instantiation argument; a bare " *
    "value binds only a tracer used directly — pass (; $J = ...)"))
@inline resolve(n::Node1{F, I}, args) where {F, I} = F.instance(resolve(n.inner, args))
@inline resolve(n::Node2{F, I1, I2}, args) where {F, I1, I2} =
    F.instance(resolve(n.inner1, args), resolve(n.inner2, args))
@inline resolve(::Constant{T}, args) where {T} = T
# Generators defer through their (possibly deferred) iterable.
@inline resolve(g::Base.Generator, args) = Base.Generator(g.f, resolve(g.iter, args))
# Multi-dimensional index sets resolve componentwise.
@inline resolve(p::Iterators.ProductIterator, args) =
    Iterators.product(map(r -> resolve(r, args), p.iterators)...)

# ── Deferred value forms: ranges, fill, comprehensions ───────────────────────
# These are value-level (not nodes): a range or an array is a *slot value*,
# never part of an expression tree.

const ArgNode = Union{ArgTracer, ArgIndexed}
const _ArgLike = Union{ArgTracer, ArgIndexed, Node1, Node2}

struct ArgRange{A, S, B}
    lo::A
    step::S
    hi::B
end
@inline resolve(r::ArgRange{A, Nothing, B}, args) where {A, B} =
    resolve(r.lo, args):resolve(r.hi, args)
@inline resolve(r::ArgRange, args) =
    resolve(r.lo, args):resolve(r.step, args):resolve(r.hi, args)

@inline Base.:(:)(a::_ArgLike, b::_ArgLike) = ArgRange(a, nothing, b)
@inline Base.:(:)(a::Integer, b::_ArgLike) = ArgRange(a, nothing, b)
@inline Base.:(:)(a::_ArgLike, b::Integer) = ArgRange(a, nothing, b)
for (A, S, B) in (
    (:_ArgLike, :_ArgLike, :_ArgLike), (:_ArgLike, :_ArgLike, :Integer),
    (:_ArgLike, :Integer, :_ArgLike), (:Integer, :_ArgLike, :_ArgLike),
    (:_ArgLike, :Integer, :Integer), (:Integer, :_ArgLike, :Integer),
    (:Integer, :Integer, :_ArgLike),
)
    @eval @inline Base.:(:)(a::$A, s::$S, b::$B) = ArgRange(a, s, b)
end

@inline Base.length(t::ArgNode) = Node1(length, t)
@inline Base.length(r::ArgRange) = Node1(length, r)

struct DeferredFill{V, N}
    value::V
    n::N
end
@inline Base.fill(v, t::_ArgLike) = DeferredFill(v, t)
@inline resolve(f::DeferredFill, args) = fill(resolve(f.value, args), resolve(f.n, args))

# A comprehension over a deferred range reaches `collect` with a generator
# whose iterable is an `ArgRange`; the whole collect defers, and the body may
# itself produce arg trees, which resolve per element.
struct DeferredCollect{F, I}
    f::F
    iter::I
end
@inline Base.collect(g::Base.Generator{<:ArgRange}) = DeferredCollect(g.f, g.iter)
@inline resolve(d::DeferredCollect, args) =
    [resolve(d.f(x), args) for x in resolve(d.iter, args)]

# ── Structure guardrails ─────────────────────────────────────────────────────

"""
    RecorderStructureError

Thrown at construction when model code attempts an operation whose result
would freeze a construction-time value into the model's structure (e.g.
branching on an instantiation argument, or iterating one outside a
generator handed to an `add_*` call).
"""
struct RecorderStructureError <: Exception
    msg::String
end
Base.showerror(io::IO, e::RecorderStructureError) =
    print(io, "RecorderStructureError: ", e.msg)

const _STRUCTURE_MSG = "model structure cannot depend on instantiation \
arguments: this operation would freeze the construction-time result into \
the model. Compute structural constants outside the model, or extend the \
arg-node vocabulary if this operation should be deferrable."

for op in (:(==), :(<), :(<=), :(>), :(>=), :isless)
    @eval begin
        Base.$op(::ArgNode, ::Number) = throw(RecorderStructureError(_STRUCTURE_MSG))
        Base.$op(::Number, ::ArgNode) = throw(RecorderStructureError(_STRUCTURE_MSG))
        Base.$op(::ArgNode, ::ArgNode) = throw(RecorderStructureError(_STRUCTURE_MSG))
    end
end
Base.iterate(::ArgNode, state...) = throw(
    RecorderStructureError(
        "cannot iterate an instantiation argument at construction time. " *
        "Pass generators over deferred ranges directly to " *
        "add_var/add_con/add_obj — they are iterated at materialization.",
    ),
)
