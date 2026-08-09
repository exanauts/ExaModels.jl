# ── Symbolic build-time arguments ─────────────────────────────────────────────
#
# `ArgSource` mirrors [`DataSource`](@ref), one level up in the pipeline.
#
#   DataSource  — a placeholder for one *data record*, resolved per iteration
#                 when the SIMD kernel runs.  Lives inside the expression graph.
#   ArgSource   — a placeholder for the *model arguments* (sizes, initial
#                 values, bounds), resolved once when `ExaModel(core)` is built.
#                 Lives in the slots that the expression graph is built *from*.
#
# The two share a shape: a sentinel singleton whose field/index access returns a
# node encoding the access path in a type parameter, so the lookup compiles away.
# They differ in what they are called with — `DataSource` is evaluated as
# `node(i, x, θ)` on every kernel iteration, `ArgSource` as
# `instantiate(node, arg)` exactly once.
#
# This lets an `ExaCore` be written against arguments that are not yet known:
#
#     @add_var(c, x, arg.N; start = arg.v)
#     @add_con(c, x[i]^2 for i in 1:arg.N; lcon = 0.0, ucon = arg.nh * ones(10))
#
# and instantiated later.  Anything with no `ArgSource` in it is passed through
# by `instantiate` untouched, so ordinary (non-symbolic) models are unaffected.

"""
    AbstractArgNode

Root type for the symbolic *argument* expression tree.  Nodes are built by
ordinary Julia operations on [`arg`](@ref) (`arg.N`, `length(arg.v)`,
`arg.nh * zeros(10)`, `1:arg.N`, `arg.v .+ 1`, …) and are resolved to concrete
values by [`instantiate`](@ref).
"""
abstract type AbstractArgNode end

"""
    ArgSource <: AbstractArgNode

Sentinel node standing for the whole argument object supplied at instantiation
time.  Field access or indexing returns an [`ArgIndexed`](@ref) node.  The
exported singleton [`arg`](@ref) is the one instance ever needed.
"""
struct ArgSource <: AbstractArgNode end

"""
    arg

The singleton [`ArgSource`](@ref).  Write `arg.N`, `arg.v`, `arg[2]` in model
building code to refer to fields of the arguments that will be supplied at
instantiation time.
"""
const arg = ArgSource()

"""
    ArgIndexed{I, J} <: AbstractArgNode

The lookup `inner.J` (or `inner[J]`) where `inner` is an [`ArgSource`](@ref) or
another `ArgIndexed`.  `J` is a type parameter, so `getproperty` / `getindex`
resolves at compile time — the same encoding [`DataIndexed`](@ref) uses.
"""
struct ArgIndexed{I, J} <: AbstractArgNode
    inner::I
end
@inline ArgIndexed(inner::I, j) where {I} = ArgIndexed{I, j}(inner)

"""
    ArgNode1{F, I} <: AbstractArgNode

A unary operation `f(inner)` deferred until instantiation.  `f` is stored as a
field; for the singleton function types this always holds in practice, that
field is zero-size and the call is statically resolved.
"""
struct ArgNode1{F, I} <: AbstractArgNode
    f::F
    inner::I
end

"""
    ArgNode2{F, I1, I2} <: AbstractArgNode

A binary operation `f(inner1, inner2)` deferred until instantiation.  Either
child may be a plain value (a `Number`, an `AbstractArray`, …) so that mixed
expressions such as `arg.nh * zeros(10)` are representable.
"""
struct ArgNode2{F, I1, I2} <: AbstractArgNode
    f::F
    inner1::I1
    inner2::I2
end

"""
    ArgCall{F, A} <: AbstractArgNode

A deferred call `f(args...)` of arbitrary arity.  `ArgNode1` / `ArgNode2` cover
the operator cases that users write and print nicely; `ArgCall` is what the
model-building internals use when an existing helper simply needs to run later
against instantiated inputs.

`args` is always a `Tuple`, and is written out at every call site.  A varargs
constructor would be shadowed by the default two-field one for the arity-1 case
— `ArgCall(f, x)` would quietly store `x` itself rather than `(x,)` — so there
deliberately isn't one.
"""
struct ArgCall{F, A <: Tuple} <: AbstractArgNode
    f::F
    args::A
end

"""
    ArgBroadcasted{F, A} <: AbstractArgNode

A deferred broadcast `f.(args...)`, produced when a fused broadcast expression
contains an argument node (e.g. `arg.v .+ 1`).
"""
struct ArgBroadcasted{F, A} <: AbstractArgNode
    f::F
    args::A
end

# ── Access path construction ──────────────────────────────────────────────────
#
# Defined on the abstract type so every node stays symbolic under further
# access.  All internal field reads below therefore go through `getfield`.

@inline Base.getproperty(n::AbstractArgNode, s::Symbol) = ArgIndexed(n, s)
@inline Base.getindex(n::AbstractArgNode, j) = ArgIndexed(n, j)
@inline Base.indexed_iterate(n::AbstractArgNode, idx, state = 1) =
    (ArgIndexed(n, idx), idx + 1)

# ── Instantiation ─────────────────────────────────────────────────────────────

"""
    instantiate(x, arg)

Resolve every [`AbstractArgNode`](@ref) inside `x` against the concrete
argument object `arg`, and return the resulting value.

`instantiate` is the identity on anything that carries no `ArgSource`
dependency — including containers, which are returned unchanged (`===`) rather
than rebuilt — so it is safe to apply to every slot of a model unconditionally.

## Example
```jldoctest
julia> using ExaModels

julia> ExaModels.instantiate(arg.nh * ones(3), (nh = 2,))
3-element Vector{Float64}:
 2.0
 2.0
 2.0

julia> x = [1.0, 2.0];

julia> ExaModels.instantiate(x, (nh = 2,)) === x     # identity, no arg dependency
true
```
"""
@inline instantiate(x, a) = x
@inline instantiate(::ArgSource, a) = a
@inline instantiate(n::ArgIndexed{I, J}, a) where {I, J} =
    _arg_access(instantiate(getfield(n, :inner), a), J)
@inline instantiate(n::ArgNode1, a) =
    getfield(n, :f)(instantiate(getfield(n, :inner), a))
@inline instantiate(n::ArgNode2, a) = getfield(n, :f)(
    instantiate(getfield(n, :inner1), a),
    instantiate(getfield(n, :inner2), a),
)
@inline instantiate(n::ArgCall, a) =
    getfield(n, :f)(map(x -> instantiate(x, a), getfield(n, :args))...)
@inline instantiate(n::ArgBroadcasted, a) =
    broadcast(getfield(n, :f), map(x -> instantiate(x, a), getfield(n, :args))...)

@inline _arg_access(x, j::Symbol) = getproperty(x, j)
@inline _arg_access(x, j) = getindex(x, j)

# Containers map unconditionally — never guarded by a "does this contain an
# argument node?" predicate.  Such a predicate can only see what is *directly*
# in the tuple, so `core.var` — a tuple of `Variable` structs each holding
# placeholders — reads as clean and is handed back untouched, leaving a core
# that looks fully instantiated and is not.  Mapping costs nothing observable:
# an immutable tuple rebuilds `===` to itself, and each element is passed
# through by identity when it has no dependency of its own.
@inline instantiate(t::Tuple, a) = map(x -> instantiate(x, a), t)
@inline instantiate(t::NamedTuple, a) = map(x -> instantiate(x, a), t)

"""
    _anyarg(xs...)

`Val(true)` if any of `xs` is an [`AbstractArgNode`](@ref), `Val(false)`
otherwise — used to pick between the eager and the deferred form of a
build-time operation.

Resolved entirely by dispatch: each method returns a literal `Val`, and the
recursion is over the argument list, so the answer is part of the method
signature rather than a `Bool` that inference has to fold.  That matters for
`juliac --trim=safe`, where a `Val(::Bool)` built from a runtime value would
leave the choice — and both branches' return types — unresolved.
"""
@inline _anyarg() = Val(false)
@inline _anyarg(::AbstractArgNode, xs...) = Val(true)
@inline _anyarg(x, xs...) = _anyarg(xs...)

# ── Algebra ───────────────────────────────────────────────────────────────────
#
# Operands that may sit opposite an argument node.  Deliberately a closed union
# rather than `Any`: it keeps these methods from claiming dispatch they have no
# business in, and avoids ambiguities with Base's own array/number arithmetic.
const ArgOperand = Union{Number, AbstractArray, Tuple}

# Binary operations. Each gets three methods (node⊗node, node⊗value,
# value⊗node) so either or both sides may be symbolic.
#
# NOTE: the *stored* function must be spelled `Base.$op` too, not the bare
# symbol.  ExaModels shadows a few Base names in its own namespace (`size` in
# nlp.jl, `sort!` in templates.jl, `append!`), so a bare `$op` inside this
# module would capture the ExaModels method and `instantiate` would then call
# the wrong function on a perfectly ordinary array.
for op in (:+, :-, :*, :/, :\, :^, :%, :÷, :fld, :cld, :mod, :min, :max)
    @eval begin
        @inline Base.$op(a::AbstractArgNode, b::AbstractArgNode) = ArgNode2(Base.$op, a, b)
        @inline Base.$op(a::AbstractArgNode, b::ArgOperand) = ArgNode2(Base.$op, a, b)
        @inline Base.$op(a::ArgOperand, b::AbstractArgNode) = ArgNode2(Base.$op, a, b)
    end
end

# `x^2` lowers to `Base.literal_pow(^, x, Val(2))`, which would otherwise miss
# the `^` methods above.
@inline Base.literal_pow(::typeof(^), a::AbstractArgNode, ::Val{p}) where {p} =
    ArgNode2(^, a, p)

# Unary operations, and the queries that make sizes usable symbolically
# (`length(arg.v)`, `size(arg.A, 2)` via the binary table below, …).
for op in (
    :+, :-, :abs, :abs2, :sqrt, :cbrt, :inv, :exp, :log, :log2, :log10,
    :sin, :cos, :tan, :floor, :ceil, :round, :trunc, :sign,
    :length, :size, :ndims, :eltype, :first, :last, :axes, :collect,
    :sum, :prod, :maximum, :minimum, :zeros, :ones, :zero, :one, :transpose,
)
    @eval @inline Base.$op(a::AbstractArgNode) = ArgNode1(Base.$op, a)
end

# Two-argument forms of the same queries / constructors.  `getindex` is absent
# deliberately: indexing *into* an argument node is the access-path case above
# and must stay an `ArgIndexed`, so only the "concrete container, symbolic
# index" direction (`v[arg.i]`) is added here.
for op in (:size, :axes, :reshape, :fill, :repeat)
    @eval begin
        @inline Base.$op(a::AbstractArgNode, b::AbstractArgNode) = ArgNode2(Base.$op, a, b)
        @inline Base.$op(a::AbstractArgNode, b::ArgOperand) = ArgNode2(Base.$op, a, b)
        @inline Base.$op(a::ArgOperand, b::AbstractArgNode) = ArgNode2(Base.$op, a, b)
    end
end
# Split by concrete operand type rather than reusing `ArgOperand`: Base already
# has `getindex(::AbstractArray, I...)`, so the union form is ambiguous with it.
@inline Base.getindex(a::AbstractArray, b::AbstractArgNode) = ArgNode2(Base.getindex, a, b)
@inline Base.getindex(a::Tuple, b::AbstractArgNode) = ArgNode2(Base.getindex, a, b)
@inline Base.getindex(a::Number, b::AbstractArgNode) = ArgNode2(Base.getindex, a, b)

# Type-directed conversions: `floor(Int, arg.x)`, `zeros(Float64, arg.N)`.
# `Base.Fix1` keeps the type parameter in the node's type instead of capturing a
# closure, so the call stays statically resolvable.
for op in (:floor, :ceil, :round, :trunc, :zeros, :ones, :convert)
    @eval @inline Base.$op(::Type{T}, a::AbstractArgNode) where {T} =
        ArgNode1(Base.Fix1(Base.$op, T), a)
end

# Ranges: `1:arg.N`, `arg.lo:arg.hi`.
@inline Base.:(:)(a::AbstractArgNode, b::AbstractArgNode) = ArgNode2(Colon(), a, b)
@inline Base.:(:)(a::AbstractArgNode, b::Number) = ArgNode2(Colon(), a, b)
@inline Base.:(:)(a::Number, b::AbstractArgNode) = ArgNode2(Colon(), a, b)

# ── Broadcasting ──────────────────────────────────────────────────────────────
#
# Any fused broadcast containing an argument node becomes a single deferred
# `ArgBroadcasted`.  `Broadcast.flatten` collapses the nesting first, so the
# stored args are leaves and `instantiate` needs no Broadcasted handling.

struct ArgBroadcastStyle <: Base.BroadcastStyle end
Base.BroadcastStyle(::Type{<:AbstractArgNode}) = ArgBroadcastStyle()
Base.BroadcastStyle(s::ArgBroadcastStyle, ::Base.BroadcastStyle) = s
Base.BroadcastStyle(s::ArgBroadcastStyle, ::ArgBroadcastStyle) = s

# An argument node enters a broadcast as itself.  Without this the fallback
# `broadcastable(x) = collect(x)` would fire and wrap it in a `collect` node.
Base.broadcastable(a::AbstractArgNode) = a

# Skip axis computation: the shapes are not known until instantiation, and the
# result is built by `broadcast` there anyway.
Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{ArgBroadcastStyle}) = bc

function Base.copy(bc::Base.Broadcast.Broadcasted{ArgBroadcastStyle})
    flat = Base.Broadcast.flatten(bc)
    return ArgBroadcasted(flat.f, flat.args)
end

# ── Display ───────────────────────────────────────────────────────────────────

const _ARG_INFIX = Dict{Any, String}(
    (+) => " + ", (-) => " - ", (*) => " * ", (/) => " / ", (\) => " \\ ",
    (^) => "^", (%) => " % ", (÷) => " ÷ ", Colon() => ":",
)

_arg_string(x) = repr(x)
_arg_string(::ArgSource) = "arg"
function _arg_string(n::ArgIndexed{I, J}) where {I, J}
    inner = _arg_string(getfield(n, :inner))
    return J isa Symbol ? "$inner.$J" : "$inner[$(repr(J))]"
end
_arg_string(n::ArgNode1) =
    "$(_arg_fname(getfield(n, :f)))($(_arg_string(getfield(n, :inner))))"
function _arg_string(n::ArgNode2)
    f = getfield(n, :f)
    l = _arg_string(getfield(n, :inner1))
    r = _arg_string(getfield(n, :inner2))
    infix = get(_ARG_INFIX, f, nothing)
    return infix === nothing ? "$(_arg_fname(f))($l, $r)" : "($l$infix$r)"
end
_arg_string(n::ArgCall) =
    "$(_arg_fname(getfield(n, :f)))($(join(map(_arg_string, getfield(n, :args)), ", ")))"
_arg_string(n::ArgBroadcasted) =
    "$(_arg_fname(getfield(n, :f))).($(join(map(_arg_string, getfield(n, :args)), ", ")))"

_arg_fname(f) = string(f)
_arg_fname(f::Base.Fix1) = "$(string(f.f))($(string(f.x)), ·)"

Base.show(io::IO, n::AbstractArgNode) = print(io, _arg_string(n))
