"""
    Const{T} <: AbstractNode

A type-stable constant wrapper for use in ExaModels expressions. The value is
stored as a field, NOT in the type parameter, so `Const{Float64}` is the same
type regardless of the numeric value. This prevents recompilation when the value
changes (unlike `Val{V}` which bakes `V` into the type).

Use `Const(v)` when a numeric value is a model parameter that varies across
calls but should not trigger recompilation of the expression tree.

# Example
```julia
function mymodel(N)
    c = ExaCore()
    @var(c, x, 10)
    @obj(c, Const(N) * x[i] for i in 1:10)  # no recompilation when N changes
    return ExaModel(c)
end
```

Since `Const <: AbstractNode`, it participates in all existing operator dispatch
(`AbstractNode × AbstractNode`, etc.) without any additional method definitions.
"""
struct Const{T} <: AbstractNode
    value::T
end
@inline (c::Const)(i, x, θ) = c.value
@inline (c::Const)(::Identity, x, θ) = c.value
@inline Const(v::Const) = v  # idempotent constructor

"""
    _maybe_const(v)

Runtime gatekeeper for auto-Const wrapping in macros. Only wraps plain `Real`
values in `Const`; everything else (ExaModels Variables, Parameters, Expressions,
AbstractNodes, etc.) passes through unchanged.
"""
@inline _maybe_const(v::Real) = Const(v)
@inline _maybe_const(v) = v

# ── Auto-Const AST helpers (used by @obj, @con, @con!, @expr macros) ─────────
# These walk generator expressions at macro-expansion time and wrap free symbols
# (not iterator vars, not function names, not array/dot bases) via _maybe_const().

const _MAYBE_CONST_REF = GlobalRef(@__MODULE__, :_maybe_const)
const _EXA_SUM_REF    = GlobalRef(@__MODULE__, :exa_sum)
const _EXA_PROD_REF   = GlobalRef(@__MODULE__, :exa_prod)

# Collect iterator variable names from the LHS of a for-clause
function _collect_iter_vars!(vars::Set{Symbol}, lhs)
    if lhs isa Symbol
        push!(vars, lhs)
    elseif lhs isa Expr && lhs.head == :tuple
        for arg in lhs.args
            _collect_iter_vars!(vars, arg)
        end
    end
end

# Entry point: find generators in an expression and wrap free symbols in their bodies
function _auto_const_gen(expr)
    expr isa Expr || return expr
    if expr.head == :generator
        return _process_generator(expr, Set{Symbol}())
    elseif expr.head == :flatten
        return Expr(:flatten, _auto_const_gen(expr.args[1]))
    elseif expr.head == :call
        # e.g. sum(generator) — recurse into call arguments but keep function name
        new_args = Any[expr.args[1]]
        for i in 2:length(expr.args)
            push!(new_args, _auto_const_gen(expr.args[i]))
        end
        return Expr(:call, new_args...)
    else
        return expr
    end
end

function _process_generator(gen, outer_vars::Set{Symbol})
    iter_vars = copy(outer_vars)
    for spec in gen.args[2:end]
        _collect_iter_vars!(iter_vars, spec.args[1])
    end

    # hoisted: accumulates (sym => val_expr) pairs that must be computed OUTSIDE
    # the generator closure for juliac constant propagation to resolve Val{N}.
    hoisted = Pair{Symbol,Any}[]

    body = gen.args[1]
    if body isa Expr && body.head == :generator
        new_body = _process_generator(body, iter_vars)
    elseif body isa Expr && body.head == :flatten
        new_body = Expr(:flatten, _process_generator(body.args[1], iter_vars))
    else
        new_body = _wrap_free_symbols(body, iter_vars, hoisted)
    end

    result = Expr(:generator, new_body, gen.args[2:end]...)

    # Wrap in a let-block so hoisted Val bindings are evaluated once, outside the
    # generator closure.  Julia's inference can then propagate Val{N} concretely.
    if !isempty(hoisted)
        binds = Expr(:block, [Expr(:(=), p.first, p.second) for p in hoisted]...)
        result = Expr(:let, binds, result)
    end

    return result
end

# Walk an expression and wrap free symbols in Const().
# "Free" means: not an iterator var, not in function-call position,
# not an array base (first arg of ref[]), not a dot-access target.
# `hoisted` accumulates (sym => expr) pairs for Val bindings that must be
# evaluated outside the generator closure (for juliac constant propagation).
function _wrap_free_symbols(expr, iter_vars::Set{Symbol},
                            hoisted::Vector{Pair{Symbol,Any}} = Pair{Symbol,Any}[])
    if expr isa Symbol
        return expr in iter_vars ? expr : Expr(:call, _MAYBE_CONST_REF, expr)
    elseif !(expr isa Expr)
        return expr  # literals, LineNumberNode, etc.
    end

    h = expr.head
    if h == :call
        fn = expr.args[1]
        # sum(body for j in range) / prod(body for j in range) — single-spec generator only.
        # Transform to exa_sum(j -> wrapped_body, lo, val_sym) where val_sym is a
        # hoisted Val(length(range)) binding.  Hoisting outside the generator closure
        # lets juliac --trim=safe resolve Val{N} concretely via constant propagation.
        if length(expr.args) == 2 &&
           expr.args[2] isa Expr && expr.args[2].head == :generator &&
           length(expr.args[2].args) == 2 &&   # body + exactly one for-spec
           (fn === :sum || fn === :prod ||
            fn isa GlobalRef && (fn.name === :sum || fn.name === :prod))
            gen        = expr.args[2]
            body       = gen.args[1]
            spec       = gen.args[2]            # Expr(:(=), iter_var_lhs, range_expr)
            iter_lhs   = spec.args[1]
            range_expr = spec.args[2]
            nested_vars = copy(iter_vars)
            _collect_iter_vars!(nested_vars, iter_lhs)
            wrapped_body = _wrap_free_symbols(body, nested_vars, hoisted)
            ref = (fn === :sum || fn isa GlobalRef && fn.name === :sum) ?
                  _EXA_SUM_REF : _EXA_PROD_REF
            # Hoist Val(range) outside the generator closure so juliac can
            # resolve the concrete type (e.g. Val{1:3}).
            val_sym = gensym(:exa_sum_val)
            push!(hoisted, val_sym => Expr(:call, :Val, range_expr))
            return Expr(:call, ref, Expr(:->, iter_lhs, wrapped_body), val_sym)
        end
        new_args = Any[expr.args[1]]  # keep function name as-is
        for i in 2:length(expr.args)
            push!(new_args, _wrap_free_symbols(expr.args[i], iter_vars, hoisted))
        end
        return Expr(:call, new_args...)
    elseif h == :ref
        new_args = Any[expr.args[1]]  # keep array base as-is
        for i in 2:length(expr.args)
            push!(new_args, _wrap_free_symbols(expr.args[i], iter_vars, hoisted))
        end
        return Expr(:ref, new_args...)
    elseif h == :generator
        nested_vars = copy(iter_vars)
        for spec in expr.args[2:end]
            _collect_iter_vars!(nested_vars, spec.args[1])
        end
        new_body = _wrap_free_symbols(expr.args[1], nested_vars, hoisted)
        return Expr(:generator, new_body, expr.args[2:end]...)
    elseif h == :->
        lambda_vars = copy(iter_vars)
        _collect_iter_vars!(lambda_vars, expr.args[1])
        new_body = _wrap_free_symbols(expr.args[2], lambda_vars, hoisted)
        return Expr(:->, expr.args[1], new_body)
    elseif h == :. || h == :quote || h == :macrocall
        return expr
    else
        return Expr(h, [_wrap_free_symbols(a, iter_vars, hoisted) for a in expr.args]...)
    end
end

# Operator dispatch for AbstractNode.
#
# All arithmetic constants are stored as Const{T} in the expression tree.
# Const{T} is always a concrete type for any T, which is required by juliac --trim=safe
# to resolve dispatch through the Node2 constructor without abstract Val{<:Real}.
#
# Val{V} (value-encoded type parameter) is still used for literal integer exponents
# via literal_pow and _pow_val, where V is always a compile-time constant.

# ── ^ with integer exponent ──────────────────────────────────────────────────
# Intercept AbstractNode ^ Integer before _register_biv can add the generic version.
# Override literal_pow (Julia's x^N fast-path) so the exponent stays as a type
# parameter Val{P} all the way through — juliac --trim=safe needs this to trace the
# specific Node2{^, ..., Val{P}} constructor without abstract dispatch.
@inline Base.literal_pow(::typeof(^), d1::AbstractNode, ::Val{P}) where {P} =
    _pow_val(d1, Val{P}())
# Runtime integer exponent: use Const so the type is concrete for juliac.
@inline Base.:^(d1::AbstractNode, d2::Integer) = Node2(^, d1, Const(d2))
# Runtime Real (e.g. Float64) exponent: Val(d2::Real) would infer abstract Val{<:Real}
# in the _register_biv generic body; use Const{T} instead (always concrete for any T).
@inline Base.:^(d1::AbstractNode, d2::Real) = Node2(^, d1, Const(d2))
@inline _pow_val(d1::AbstractNode, ::Val{1}) = d1
@inline _pow_val(d1::AbstractNode, ::Val{2}) = Node1(abs2, d1)
@inline _pow_val(d1::AbstractNode, ::Val{V}) where {V} = Node2(^, d1, Val{V}())
# Val{V}() used as inner2 in Node2(^, d1, Val{V}()) — make it callable in the eval context.
@inline (::Val{V})(i, x, θ) where {V} = V

# ── node OP real / real OP node ───────────────────────────────────────────────
# Use Const(d) instead of Val(d): Const{T} is always a concrete type for any T,
# whereas Val(d::Real) produces abstract Val{<:Real} when d is a runtime value,
# preventing juliac --trim=safe from resolving the Node2 constructor dispatch.
# Note: the identity-element shortcuts (x+0=x, x*1=x) required Val and only
# worked for compile-time constants anyway; they are omitted here for simplicity.
@inline Base.:+(d1::AbstractNode, d2::Real) = Node2(+, d1, Const(d2))
@inline Base.:-(d1::AbstractNode, d2::Real) = Node2(-, d1, Const(d2))
@inline Base.:*(d1::AbstractNode, d2::Real) = Node2(*, d1, Const(d2))
@inline Base.:/(d1::AbstractNode, d2::Real) = Node2(/, d1, Const(d2))

@inline Base.:+(d1::Real, d2::AbstractNode) = Node2(+, Const(d1), d2)
@inline Base.:*(d1::Real, d2::AbstractNode) = Node2(*, Const(d1), d2)
@inline Base.:-(d1::Real, d2::AbstractNode) = Node2(-, Const(d1), d2)

# ── exa_sum / exa_prod ────────────────────────────────────────────────────────
#
# Type-stable sum/prod for use inside @obj / @con / @expr generators.
#
# The macro transformation rewrites:
#   sum(body for j in range)  →  exa_sum(j -> wrapped_body, range)
#   prod(body for j in range) →  exa_prod(j -> wrapped_body, range)
#
# The function-based form is type-stable for:
#   • Tuple iterators  — tail-recursive _exa_map builds a concrete NTuple
#   • UnitRange{Int}   — ntuple(f, Val{N}()) is concrete when N is a
#                        compile-time constant (e.g. literal range 1:3 or
#                        a Val-encoded length). Runtime-variable lengths are
#                        not supported for juliac --trim=safe.

# Helper: type-stable map over a tuple → returns a concrete tuple of results.
@inline _exa_map(f, ::Tuple{}) = ()
@inline _exa_map(f, t::Tuple) = (f(t[1]), _exa_map(f, Base.tail(t))...)

# ── Tuple iterator ────────────────────────────────────────────────────────────

"""
    exa_sum(f, itr)
    exa_sum(f, ::Val{range})
    exa_sum(gen::Base.Generator)

Build a [`SumNode`](@ref) representing `∑ f(k)` for `k ∈ itr`. Inside `@obj`,
`@con`, and `@expr` macros, `sum(body for k in range)` is automatically
rewritten to `exa_sum(k -> body, Val(range))` with the `Val` hoisted outside
the generator closure.

# Supported iterators
- `Tuple`: type-stable via tail recursion.
- `UnitRange{Int}`: type-stable when the length is a compile-time constant.
- `Val{range}` (preferred for juliac): `exa_sum(f, Val(1:nc))` embeds the range
  in the type parameter, ensuring `juliac --trim=safe` can resolve `Val{N}`.

# juliac / AOT usage
Julia's inference cannot propagate constants like `nc = 3` through a generator
closure boundary.  When calling `add_con`/`add_obj` programmatically (not via
macro), hoist `Val(range)` outside the generator:

```julia
v = Val(1:nc)                                          # outside generator
c, con = add_con(c, (exa_sum(j -> x[j], v) for i in 1:nh))  # v captured
```
"""
@inline exa_sum(f, ::Tuple{}) = SumNode(())
@inline exa_sum(f, t::Tuple) = SumNode(_exa_map(f, t))

"""
    exa_prod(f, itr)
    exa_prod(f, ::Val{range})
    exa_prod(gen::Base.Generator)

Build a [`ProdNode`](@ref) representing `∏ f(k)` for `k ∈ itr`. Inside `@obj`,
`@con`, and `@expr` macros, `prod(body for k in range)` is automatically
rewritten to `exa_prod(k -> body, Val(range))` with the `Val` hoisted outside
the generator closure.

See [`exa_sum`](@ref) for supported iterators and juliac usage notes.
"""
@inline exa_prod(f, ::Tuple{}) = ProdNode(())
@inline exa_prod(f, t::Tuple) = ProdNode(_exa_map(f, t))

# ── UnitRange{Int}: Val{N}-based recursion ────────────────────────────────────

# UnitRange form (for direct use / Generator fallback — NOT type-stable inside
# generator closures for juliac; prefer the Val form below).
@inline exa_sum(f, r::UnitRange{Int}) = _exa_sum_range(f, first(r), Val(length(r)))
@inline exa_prod(f, r::UnitRange{Int}) = _exa_prod_range(f, first(r), Val(length(r)))

# Val-wrapped range form: Val(1:nc) embeds the range in the type parameter.
# The macro hoists `Val(range)` outside the generator closure so juliac can
# resolve the concrete Val{UnitRange{Int64}(1,3)} type.
# Users can also call this directly: `exa_sum(j -> x[j], Val(1:nc))`
@inline function exa_sum(f, ::Val{r}) where {r}
    _exa_sum_range(f, first(r), Val(length(r)))
end
@inline function exa_prod(f, ::Val{r}) where {r}
    _exa_prod_range(f, first(r), Val(length(r)))
end

@inline _exa_sum_range(f, lo::Int, ::Val{N}) where {N} =
    SumNode(ntuple(i -> f(lo + i - 1), Val{N}()))

@inline _exa_prod_range(f, lo::Int, ::Val{N}) where {N} =
    ProdNode(ntuple(i -> f(lo + i - 1), Val{N}()))

# ── Generator form (direct use outside macro) ─────────────────────────────────

@inline exa_sum(gen::Base.Generator) = exa_sum(gen.f, gen.iter)
@inline exa_prod(gen::Base.Generator) = exa_prod(gen.f, gen.iter)
