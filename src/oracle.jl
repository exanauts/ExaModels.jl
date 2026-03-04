# oracle.jl
#
# VectorNonlinearOracle: lets users plug arbitrary nonlinear constraint blocks
# (e.g. GPU simulation residuals, external ODE solvers) into an ExaModel by
# supplying three Julia callbacks:
#
#   f!   (c, x)      — fills c[1:ncon] with residuals
#   jac! (vals, x)   — fills vals[1:nnzj] with Jacobian nonzeros
#   hess!(vals, x, y)— fills vals[1:nnzh] with Lagrangian Hessian nonzeros
#                       (y[1:ncon] = constraint multipliers)
#
# Sparsity patterns (jac_rows/jac_cols, hess_rows/hess_cols) are declared
# once at construction time and never change.
#
# Usage:
#
#   oracle = VectorNonlinearOracle(
#       nvar      = nvar,
#       ncon      = ncon,
#       nnzj      = nnzj,
#       nnzh      = nnzh,
#       jac_rows  = [...],   # length nnzj, 1-based row indices (1:ncon)
#       jac_cols  = [...],   # length nnzj, 1-based col indices (1:nvar)
#       hess_rows = [...],   # length nnzh, 1-based row (lower-triangle)
#       hess_cols = [...],   # length nnzh, 1-based col
#       lcon      = [...],   # length ncon, lower bound
#       ucon      = [...],   # length ncon, upper bound
#       f!        = (c, x)     -> ...,
#       jac!      = (vals, x)  -> ...,
#       hess!     = (vals, x, y) -> ...,
#   )
#   constraint(core, oracle)   # registers oracle; updates ncon/nnzj/nnzh in ExaCore
#   model = ExaModel(core)     # returns ExaModelWithOracle automatically

"""
    VectorNonlinearOracle

Encapsulates a user-supplied nonlinear constraint block that can be registered
into an `ExaCore` with [`constraint`](@ref). The block is characterised by:

- `nvar`: number of decision variables (= total problem `nvar`)
- `ncon`: number of constraints this oracle contributes
- `nnzj`: number of Jacobian nonzeros
- `nnzh`: number of Lagrangian Hessian nonzeros (set to 0 to use L-BFGS / finite differences)
- `jac_rows`, `jac_cols`: sparsity pattern of the Jacobian (1-based, length `nnzj`)
- `hess_rows`, `hess_cols`: sparsity pattern of the (lower-triangular) Hessian (1-based, length `nnzh`)
- `lcon`, `ucon`: constraint lower/upper bounds (length `ncon`)
- `f!`: `f!(c, x)` — writes residuals into `c[1:ncon]`
- `jac!`: `jac!(vals, x)` — writes Jacobian values into `vals[1:nnzj]` in the declared sparsity order
- `hess!`: `hess!(vals, x, y)` — writes Hessian values into `vals[1:nnzh]` (y = constraint multipliers)
"""
struct VectorNonlinearOracle{F, J, H, VT <: AbstractVector}
    nvar::Int
    ncon::Int
    nnzj::Int
    nnzh::Int
    jac_rows::Vector{Int}    # length nnzj, 1-based row indices into the oracle's own constraint block
    jac_cols::Vector{Int}    # length nnzj, 1-based column indices into x
    hess_rows::Vector{Int}   # length nnzh, 1-based row indices into x (lower triangle)
    hess_cols::Vector{Int}   # length nnzh, 1-based column indices into x
    lcon::VT
    ucon::VT
    f!::F
    jac!::J
    hess!::H
end

function VectorNonlinearOracle(;
    nvar::Int,
    ncon::Int,
    nnzj::Int,
    nnzh::Int,
    jac_rows::Vector{Int},
    jac_cols::Vector{Int},
    hess_rows::Vector{Int} = Int[],
    hess_cols::Vector{Int} = Int[],
    lcon::AbstractVector    = zeros(ncon),
    ucon::AbstractVector    = zeros(ncon),
    f!,
    jac!,
    hess! = (vals, x, y) -> nothing,
)
    @assert length(jac_rows)  == nnzj "jac_rows length must equal nnzj"
    @assert length(jac_cols)  == nnzj "jac_cols length must equal nnzj"
    @assert length(hess_rows) == nnzh "hess_rows length must equal nnzh"
    @assert length(hess_cols) == nnzh "hess_cols length must equal nnzh"
    @assert length(lcon)      == ncon "lcon length must equal ncon"
    @assert length(ucon)      == ncon "ucon length must equal ncon"
    return VectorNonlinearOracle(
        nvar, ncon, nnzj, nnzh,
        jac_rows, jac_cols,
        hess_rows, hess_cols,
        lcon, ucon,
        f!, jac!, hess!,
    )
end

Base.show(io::IO, o::VectorNonlinearOracle) = print(
    io,
    """
VectorNonlinearOracle

  ncon: $(o.ncon)   nnzj: $(o.nnzj)   nnzh: $(o.nnzh)
""",
)

# ── Registration ──────────────────────────────────────────────────────────────

"""
    constraint(core::ExaCore, oracle::VectorNonlinearOracle)

Register a `VectorNonlinearOracle` with an `ExaCore`.  The oracle's constraint
bounds are appended to `core.lcon`/`core.ucon`, and the constraint/Jacobian/
Hessian counts are updated.  When `ExaModel(core)` is called afterwards, the
result is an [`ExaModelWithOracle`](@ref) that evaluates both the SIMD symbolic
constraints and all registered oracles.

Returns the oracle (for indexing convenience).
"""
function constraint(c::ExaCore, oracle::VectorNonlinearOracle)
    push!(c.oracles, oracle)
    c.ncon += oracle.ncon
    c.nnzj += oracle.nnzj
    c.nnzh += oracle.nnzh
    c.y0   = append!(c.backend, c.y0,   zero(eltype(c.θ)),  oracle.ncon)
    c.lcon = append!(c.backend, c.lcon, oracle.lcon, oracle.ncon)
    c.ucon = append!(c.backend, c.ucon, oracle.ucon, oracle.ncon)
    return oracle
end

# ── Model type ────────────────────────────────────────────────────────────────

"""
    ExaModelWithOracle

An `AbstractExaModel` that augments an `ExaModel`'s SIMD symbolic constraints
with one or more [`VectorNonlinearOracle`](@ref) callback blocks.  Constructed
automatically by `ExaModel(core)` when `core` has registered oracles.

The constraint ordering is:
  1. SIMD symbolic constraints  (indices  1 : n_simd_con)
  2. Oracle blocks, in registration order

and analogously for the Jacobian and Hessian nonzero arrays.
"""
struct ExaModelWithOracle{T, VT, E, O, C, S, R} <: AbstractExaModel{T, VT, E}
    objs::O                          # same as ExaModel
    cons::C                          # same as ExaModel (SIMD constraints only)
    θ::VT                            # same as ExaModel
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    ext::E
    tags::S
    oracles::R                       # Tuple of VectorNonlinearOracle
    oracle_con_offsets::Vector{Int}  # g-index offset (0-based) for each oracle
    oracle_jac_offsets::Vector{Int}  # jac NNZ offset (0-based) for each oracle
    oracle_hess_offsets::Vector{Int} # hess NNZ offset (0-based) for each oracle
end

function Base.show(io::IO, m::ExaModelWithOracle{T, VT}) where {T, VT}
    println(io, "An ExaModelWithOracle{$T, $VT, ...}\n")
    Base.show(io, m.meta)
end

# ── Internal constructor ───────────────────────────────────────────────────────

function _build_with_oracle(c::ExaCore; kwargs...)
    oracles = c.oracles                          # Vector{Any} of VectorNonlinearOracle

    # Total oracle contributions
    total_oracle_ncon = sum(o.ncon for o in oracles)
    total_oracle_nnzj = sum(o.nnzj for o in oracles)
    total_oracle_nnzh = sum(o.nnzh for o in oracles)

    # SIMD-only counts (c.ncon etc. already include oracle contributions)
    n_simd_ncon = c.ncon - total_oracle_ncon
    n_simd_nnzj = c.nnzj - total_oracle_nnzj
    n_simd_nnzh = c.nnzh - total_oracle_nnzh

    # Compute per-oracle offsets (0-based, relative to the full arrays)
    oracle_con_offsets  = Vector{Int}(undef, length(oracles))
    oracle_jac_offsets  = Vector{Int}(undef, length(oracles))
    oracle_hess_offsets = Vector{Int}(undef, length(oracles))

    con_off  = n_simd_ncon
    jac_off  = n_simd_nnzj
    hess_off = n_simd_nnzh
    for (i, o) in enumerate(oracles)
        oracle_con_offsets[i]  = con_off
        oracle_jac_offsets[i]  = jac_off
        oracle_hess_offsets[i] = hess_off
        con_off  += o.ncon
        jac_off  += o.nnzj
        hess_off += o.nnzh
    end

    meta = NLPModels.NLPModelMeta(
        c.nvar,
        ncon    = c.ncon,
        nnzj    = c.nnzj,
        nnzh    = c.nnzh,
        x0      = c.x0,
        lvar    = c.lvar,
        uvar    = c.uvar,
        y0      = c.y0,
        lcon    = c.lcon,
        ucon    = c.ucon,
        minimize = c.minimize,
    )

    return ExaModelWithOracle(
        c.obj,
        c.con,
        c.θ,
        meta,
        NLPModels.Counters(),
        build_extension(c; kwargs...),
        c.tags,
        Tuple(oracles),
        oracle_con_offsets,
        oracle_jac_offsets,
        oracle_hess_offsets,
    )
end

# ── NLPModels methods ──────────────────────────────────────────────────────────

# --- objective and gradient: oracles are constraints only, delegate to SIMD ---

function obj(m::ExaModelWithOracle, x::AbstractVector)
    return _obj(m.objs, x, m.θ)
end

function grad!(m::ExaModelWithOracle, x::AbstractVector, f::AbstractVector)
    fill!(f, zero(eltype(f)))
    _grad!(m.objs, x, m.θ, f)
    return f
end

# --- constraint residuals ---

function cons_nln!(m::ExaModelWithOracle, x::AbstractVector, g::AbstractVector)
    fill!(g, zero(eltype(g)))
    # SIMD symbolic constraints (indices 1 : n_simd_con)
    _cons_nln!(m.cons, x, m.θ, g)
    # Oracle constraint blocks
    for (i, oracle) in enumerate(m.oracles)
        off = m.oracle_con_offsets[i]
        oracle.f!(@view(g[off+1 : off+oracle.ncon]), x)
    end
    return g
end

# --- Jacobian ---

function jac_structure!(m::ExaModelWithOracle, rows::AbstractVector, cols::AbstractVector)
    # SIMD Jacobian sparsity
    _jac_structure!(m.cons, rows, cols)
    # Oracle Jacobian sparsity (row indices shifted by oracle's constraint offset)
    for (i, oracle) in enumerate(m.oracles)
        off_j = m.oracle_jac_offsets[i]
        off_c = m.oracle_con_offsets[i]
        for k in 1:oracle.nnzj
            rows[off_j + k] = oracle.jac_rows[k] + off_c
            cols[off_j + k] = oracle.jac_cols[k]
        end
    end
    return rows, cols
end

function jac_coord!(m::ExaModelWithOracle, x::AbstractVector, jac::AbstractVector)
    fill!(jac, zero(eltype(jac)))
    # SIMD Jacobian values
    _jac_coord!(m.cons, x, m.θ, jac)
    # Oracle Jacobian values
    for (i, oracle) in enumerate(m.oracles)
        off_j = m.oracle_jac_offsets[i]
        oracle.jac!(@view(jac[off_j+1 : off_j+oracle.nnzj]), x)
    end
    return jac
end

# Jacobian-vector products via sparse accumulation (CPU path; GPU users should
# override jprod_nln! / jtprod_nln! directly on their model type if needed).

function jprod_nln!(
    m::ExaModelWithOracle,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    fill!(Jv, zero(eltype(Jv)))
    _jprod_nln!(m.cons, x, m.θ, v, Jv)
    for (i, oracle) in enumerate(m.oracles)
        off_j = m.oracle_jac_offsets[i]
        off_c = m.oracle_con_offsets[i]
        jac_vals = similar(x, oracle.nnzj)
        oracle.jac!(jac_vals, x)
        for k in 1:oracle.nnzj
            Jv[oracle.jac_rows[k] + off_c] += jac_vals[k] * v[oracle.jac_cols[k]]
        end
    end
    return Jv
end

function jtprod_nln!(
    m::ExaModelWithOracle,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    fill!(Jtv, zero(eltype(Jtv)))
    _jtprod_nln!(m.cons, x, m.θ, v, Jtv)
    for (i, oracle) in enumerate(m.oracles)
        off_j = m.oracle_jac_offsets[i]
        off_c = m.oracle_con_offsets[i]
        jac_vals = similar(x, oracle.nnzj)
        oracle.jac!(jac_vals, x)
        for k in 1:oracle.nnzj
            Jtv[oracle.jac_cols[k]] += jac_vals[k] * v[oracle.jac_rows[k] + off_c]
        end
    end
    return Jtv
end

# --- Hessian ---

function hess_structure!(m::ExaModelWithOracle, rows::AbstractVector, cols::AbstractVector)
    _obj_hess_structure!(m.objs, rows, cols)
    _con_hess_structure!(m.cons, rows, cols)
    for (i, oracle) in enumerate(m.oracles)
        off_h = m.oracle_hess_offsets[i]
        for k in 1:oracle.nnzh
            rows[off_h + k] = oracle.hess_rows[k]
            cols[off_h + k] = oracle.hess_cols[k]
        end
    end
    return rows, cols
end

function hess_coord!(
    m::ExaModelWithOracle,
    x::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.objs, x, m.θ, hess, obj_weight)
    return hess
end

function hess_coord!(
    m::ExaModelWithOracle,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.objs, x, m.θ, hess, obj_weight)
    _con_hess_coord!(m.cons, x, m.θ, y, hess, obj_weight)
    for (i, oracle) in enumerate(m.oracles)
        off_h = m.oracle_hess_offsets[i]
        off_c = m.oracle_con_offsets[i]
        oracle.hess!(
            @view(hess[off_h+1 : off_h+oracle.nnzh]),
            x,
            @view(y[off_c+1 : off_c+oracle.ncon]),
        )
    end
    return hess
end

function hprod!(
    m::ExaModelWithOracle,
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(Hv, zero(eltype(Hv)))
    _obj_hprod!(m.objs, x, m.θ, v, Hv, obj_weight)
    return Hv
end

function hprod!(
    m::ExaModelWithOracle,
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(Hv, zero(eltype(Hv)))
    _obj_hprod!(m.objs, x, m.θ, v, Hv, obj_weight)
    _con_hprod!(m.cons, x, m.θ, y, v, Hv, obj_weight)
    # Oracle Hessian-vector product via sparse accumulation
    for (i, oracle) in enumerate(m.oracles)
        off_c = m.oracle_con_offsets[i]
        off_h = m.oracle_hess_offsets[i]
        hess_vals = similar(x, oracle.nnzh)
        oracle.hess!(hess_vals, x, @view(y[off_c+1 : off_c+oracle.ncon]))
        for k in 1:oracle.nnzh
            r, c_ = oracle.hess_rows[k], oracle.hess_cols[k]
            Hv[r] += hess_vals[k] * v[c_]
            if r != c_
                Hv[c_] += hess_vals[k] * v[r]
            end
        end
    end
    return Hv
end
