# ── Materialization ──────────────────────────────────────────────────────────
# Resolve every deferred value slot of a core against the instantiation
# `args`, producing the fully-eager core. On a sentinel-free core every
# resolve is the identity and this is a field-for-field copy.

@inline _resolve_dim(d, args) = d
@inline _resolve_dim(d::_ArgLike, args) = resolve(d, args)
@inline _resolve_dim(d::ArgRange, args) = resolve(d, args)

@inline _resolve_handle(v::Variable, args) = Variable(
    map(d -> _resolve_dim(d, args), v.size),
    resolve(v.length, args)::Int,
    resolve(v.offset, args)::Int,
    v.name,
    v.tag,
)
@inline _resolve_handle(p::Parameter, args) = Parameter(
    map(d -> _resolve_dim(d, args), p.size),
    resolve(p.length, args)::Int,
    resolve(p.offset, args)::Int,
    p.tag,
)

@inline _resolve_pending(backend, a, args) = a
function _resolve_pending(backend, p::PendingVec, args)
    a = _pending_base(backend, p.base, args)
    return append!(backend, a, resolve(p.spec, args), resolve(p.len, args)::Int)
end
# The chain's leaf is the live core's array and append! mutates: detach with
# a copy exactly once, at the bottom, so one core materializes any number of
# times.
@inline _pending_base(backend, a::AbstractVector, args) = copy(a)
@inline _pending_base(backend, p::PendingVec, args) = _resolve_pending(backend, p, args)

@inline _deferred(c::ExaCore) = !(
    c.nvar isa Int && c.npar isa Int && c.ncon isa Int && c.nconaug isa Int &&
    c.nobj isa Int && c.nnzc isa Int && c.nnzg isa Int && c.nnzj isa Int &&
    c.nnzh isa Int && c.x0 isa AbstractVector && c.θ isa AbstractVector &&
    c.lvar isa AbstractVector && c.uvar isa AbstractVector &&
    c.y0 isa AbstractVector && c.lcon isa AbstractVector &&
    c.ucon isa AbstractVector
)

# ── Structure-preserving tree rewrite ────────────────────────────────────────
# Replace arg leaves in an expression tree with their resolved values, using
# the RAW node constructors (never the operator overloads, which simplify and
# would change the structure the compressors were computed over).

@inline _rt(x, args) = x
@inline _rt(n::ArgTracer, args) = resolve(n, args)
@inline _rt(n::ArgIndexed, args) = resolve(n, args)
# Value-only ops are pure arg subtrees by construction (length, integer
# division, rounding never wrap variables): they resolve to their value
# rather than being rebuilt — a rebuilt one would survive as an unevaluated
# node that the sparsity probe then misreads as a constant 1.
@inline _rt(n::Node1{typeof(length), I}, args) where {I} = resolve(n, args)
@inline _rt(n::Node1{<:_RoundTo, I}, args) where {I} = resolve(n, args)
@inline _rt(n::Node2{typeof(div), I1, I2}, args) where {I1, I2} = resolve(n, args)
@inline _rt(n::Node1{F, I}, args) where {F, I} = Node1(F.instance, _rt(n.inner, args))
@inline _rt(n::Node2{F, I1, I2}, args) where {F, I1, I2} =
    Node2(F.instance, _rt(n.inner1, args), _rt(n.inner2, args))
@inline _rt(v::Var{I}, args) where {I} = Var(_rt(v.i, args))
@inline _rt(p::ParameterNode{I}, args) where {I} = ParameterNode(_rt(p.i, args))
@inline _rt(n::SumNode, args) = SumNode(map(c -> _rt(c, args), n.inners))
@inline _rt(n::ProdNode, args) = ProdNode(map(c -> _rt(c, args), n.inners))
@inline _rt(p::Pair, args) = _rt(p.first, args) => _rt(p.second, args)

@inline _resolve_sf(f::SIMDFunction, args) = SIMDFunction(
    _rt(f.f, args), f.comp1, f.comp2,
    resolve(f.o0, args)::Int, resolve(f.o1, args)::Int, resolve(f.o2, args)::Int,
    f.o1step, f.o2step,
)

@inline _resolve_itr(itr, args) = itr
@inline _resolve_itr(r::ArgRange, args) = resolve(r, args)
@inline _resolve_itr(n::_ArgLike, args) = resolve(n, args)
@inline _resolve_itr(d::DeferredCollect, args) = resolve(d, args)

@inline _resolve_entry(cn::Constraint, args) = Constraint(
    _resolve_sf(cn.f, args),
    _resolve_itr(cn.itr, args),
    resolve(cn.offset, args)::Int,
    map(d -> _resolve_dim(d, args), cn.size),
    cn.tag,
)
@inline _resolve_entry(a::ConstraintAugmentation, args) = ConstraintAugmentation(
    _resolve_sf(a.f, args),
    _resolve_itr(a.itr, args),
    resolve(a.oa, args)::Int,
    map(d -> _resolve_dim(d, args), a.dims),
    a.tag,
)
@inline _resolve_entry(o::Objective, args) =
    Objective(_resolve_sf(o.f, args), _resolve_itr(o.itr, args))

"""
    materialize(core, args = nothing) -> ExaCore

Resolve the core's deferred value slots (counters, offsets, dimensions,
array segments) against `args`. A sentinel-free core is returned as-is
(`===`); a deferred core materialized without args names the missing
argument. Internal — the public surface is `ExaModel(core, args)`.
"""
@inline materialize(c::ExaCore, ::Nothing = nothing) = _deferred(c) ? _materialize(c, nothing) : c
@inline materialize(c::ExaCore, args) = _materialize(c, args)

function _materialize(c::ExaCore, args)
    return _exa_core(;
        name = c.name,
        backend = c.backend,
        var = map(v -> _resolve_handle(v, args), c.var),
        par = map(p -> _resolve_handle(p, args), c.par),
        obj = map(o -> _resolve_entry(o, args), c.obj),
        cons = map(cn -> _resolve_entry(cn, args), c.cons),
        nvar = resolve(c.nvar, args)::Int,
        npar = resolve(c.npar, args)::Int,
        ncon = resolve(c.ncon, args)::Int,
        nconaug = resolve(c.nconaug, args)::Int,
        nobj = resolve(c.nobj, args)::Int,
        nnzc = resolve(c.nnzc, args)::Int,
        nnzg = resolve(c.nnzg, args)::Int,
        nnzj = resolve(c.nnzj, args)::Int,
        nnzh = resolve(c.nnzh, args)::Int,
        x0 = _resolve_pending(c.backend, c.x0, args),
        θ = _resolve_pending(c.backend, c.θ, args),
        lvar = _resolve_pending(c.backend, c.lvar, args),
        uvar = _resolve_pending(c.backend, c.uvar, args),
        y0 = _resolve_pending(c.backend, c.y0, args),
        lcon = _resolve_pending(c.backend, c.lcon, args),
        ucon = _resolve_pending(c.backend, c.ucon, args),
        minimize = c.minimize,
        tag = c.tag,
        refs = map(v -> _resolve_handle(v, args), c.refs),
        oracles = c.oracles,
        scalar_oracles = c.scalar_oracles,
        evals = c.evals,
    )
end

"""
    ExaModel(core, args; kwargs...)

Materialize `core` at `args` and build the model — `ExaModel(core)` for a
sentinel-free core is unchanged. `args` binds a NamedTuple by name, a bare
value for a single-field schema, or the tracer's direct value.
"""
@inline ExaModel(c::ExaCore, args; kwargs...) = ExaModel(materialize(c, args); kwargs...)
