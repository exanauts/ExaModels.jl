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

"""
    ExaModel(core, args; kwargs...)

Materialize `core` at `args` and build the model — `ExaModel(core)` for a
sentinel-free core is unchanged. `args` binds a NamedTuple by name, a bare
value for a single-field schema, or the tracer's direct value.
"""
@inline ExaModel(c::ExaCore, args; kwargs...) = ExaModel(materialize(c, args); kwargs...)
