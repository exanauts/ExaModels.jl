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
const _EXA_SUM_REF    = GlobalRef(@__MODULE__, :_exa_sum)
const _EXA_PROD_REF   = GlobalRef(@__MODULE__, :_exa_prod)

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

    body = gen.args[1]
    if body isa Expr && body.head == :generator
        new_body = _process_generator(body, iter_vars)
    elseif body isa Expr && body.head == :flatten
        new_body = Expr(:flatten, _process_generator(body.args[1], iter_vars))
    else
        new_body = _wrap_free_symbols(body, iter_vars)
    end

    return Expr(:generator, new_body, gen.args[2:end]...)
end

# Walk an expression and wrap free symbols in Const().
# "Free" means: not an iterator var, not in function-call position,
# not an array base (first arg of ref[]), not a dot-access target.
function _wrap_free_symbols(expr, iter_vars::Set{Symbol})
    if expr isa Symbol
        return expr in iter_vars ? expr : Expr(:call, _MAYBE_CONST_REF, expr)
    elseif !(expr isa Expr)
        return expr  # literals, LineNumberNode, etc.
    end

    h = expr.head
    if h == :call
        fn = expr.args[1]
        # sum(body for j in range) / prod(body for j in range) — single-spec generator only.
        # Transform to _exa_sum(j -> wrapped_body, range) so the fold is type-stable for juliac.
        # Only the body is auto-const-wrapped; the range is kept as plain Julia (it's not an NLP expr).
        if length(expr.args) == 2 &&
           expr.args[2] isa Expr && expr.args[2].head == :generator &&
           length(expr.args[2].args) == 2 &&   # body + exactly one for-spec
           (fn === :sum || fn === :prod ||
            fn isa GlobalRef && (fn.name === :sum || fn.name === :prod))
            gen  = expr.args[2]
            body = gen.args[1]
            spec = gen.args[2]   # Expr(:(=), iter_var_lhs, range_expr)
            iter_lhs   = spec.args[1]
            range_expr = spec.args[2]
            nested_vars = copy(iter_vars)
            _collect_iter_vars!(nested_vars, iter_lhs)
            wrapped_body = _wrap_free_symbols(body, nested_vars)
            ref = (fn === :sum || fn isa GlobalRef && fn.name === :sum) ? _EXA_SUM_REF : _EXA_PROD_REF
            return Expr(:call, ref, Expr(:->, iter_lhs, wrapped_body), range_expr)
        end
        new_args = Any[expr.args[1]]  # keep function name as-is
        for i in 2:length(expr.args)
            push!(new_args, _wrap_free_symbols(expr.args[i], iter_vars))
        end
        return Expr(:call, new_args...)
    elseif h == :ref
        new_args = Any[expr.args[1]]  # keep array base as-is
        for i in 2:length(expr.args)
            push!(new_args, _wrap_free_symbols(expr.args[i], iter_vars))
        end
        return Expr(:ref, new_args...)
    elseif h == :generator
        # Nested generator (e.g. sum(f(x) for x in 1:n) appearing in the body of an
        # outer generator).  We must collect the inner iterator variables and only
        # wrap free symbols in the generator *body* — not in the for-specs.
        # Wrapping the for-spec LHS (the iterator variable) turns `j = 1:n` into
        # `_maybe_const(j) = 1:n`, which Julia lowering reads as a global method
        # definition and raises a lowering error.
        nested_vars = copy(iter_vars)
        for spec in expr.args[2:end]
            _collect_iter_vars!(nested_vars, spec.args[1])
        end
        new_body = _wrap_free_symbols(expr.args[1], nested_vars)
        return Expr(:generator, new_body, expr.args[2:end]...)
    elseif h == :. || h == :quote || h == :macrocall
        return expr  # leave dot access, quotes, and nested macros alone
    else
        return Expr(h, [_wrap_free_symbols(a, iter_vars) for a in expr.args]...)
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

# ── Type-stable sum/prod for use inside @obj / @con generators ───────────────
#
# Julia's Base.sum/prod over a UnitRange uses a loop whose accumulator type
# changes each iteration (T → Node2{+,T,T} → Node2{+,Node2{...},T} → …),
# which is not type-stable and breaks juliac --trim=safe.
#
# _exa_sum / _exa_prod use:
#   • Tuple iterator  → Base.tail recursion (each dispatch level has a
#                        different concrete Tuple type → concrete return type)
#   • UnitRange{Int}  → Val{N}-based recursion; Val{length(r)} is concrete when
#                        the range length is a compile-time constant (e.g. nc=3).
#                        For truly runtime-length ranges the type is not inferrable
#                        by juliac, which is the intended restriction: only
#                        "concretely typed" iterators are supported inside @obj/@con.
#
# These helpers are injected by the @obj / @con macro transformation when it sees
#   sum(body for j in range)  →  _exa_sum(j -> body, range)
# and similarly for prod.

# Tuple iterator — recursion on the tail
@inline _exa_sum(f, t::Tuple{T}) where {T}  = f(t[1])
@inline _exa_sum(f, t::Tuple)               = f(t[1]) + _exa_sum(f, Base.tail(t))

@inline _exa_prod(f, t::Tuple{T}) where {T} = f(t[1])
@inline _exa_prod(f, t::Tuple)              = f(t[1]) * _exa_prod(f, Base.tail(t))

# UnitRange{Int} — Val{N}-based recursion (N = length of range, concrete when constant)
@inline _exa_sum(f, r::UnitRange{Int})  = _exa_sum_val(f,  first(r), Val(length(r)))
@inline _exa_prod(f, r::UnitRange{Int}) = _exa_prod_val(f, first(r), Val(length(r)))

@inline _exa_sum_val(f, lo, ::Val{1})       = f(lo)
@inline _exa_sum_val(f, lo, ::Val{N}) where {N} = f(lo) + _exa_sum_val(f, lo + 1, Val{N-1}())

@inline _exa_prod_val(f, lo, ::Val{1})       = f(lo)
@inline _exa_prod_val(f, lo, ::Val{N}) where {N} = f(lo) * _exa_prod_val(f, lo + 1, Val{N-1}())
