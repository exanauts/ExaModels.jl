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
    a = _resolve_pending(backend, p.base, args)
    return append!(backend, a, resolve(p.spec, args), resolve(p.len, args)::Int)
end

"""
    materialize(core, args = nothing) -> ExaCore

Resolve the core's deferred value slots (counters, offsets, dimensions,
array segments) against `args`. Sentinel-free cores pass through unchanged.
"""
function materialize(c::ExaCore, args = nothing)
    return _exa_core(;
        name = c.name,
        backend = c.backend,
        var = map(v -> _resolve_handle(v, args), c.var),
        par = map(p -> _resolve_handle(p, args), c.par),
        obj = c.obj,
        cons = c.cons,
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
