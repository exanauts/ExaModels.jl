# oracle.jl
#
# ScalarNonlinearOracle: a scalar objective term with callbacks for value,
# gradient, and (optionally) Hessian-vector product.  Registered via
# `objective(core, oracle)`.  No Jacobian is needed — only ∇f and ∇²f·v.
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
# Alternatively, matrix-free callbacks can be provided:
#
#   jvp! (Jv, x, v)     — fills Jv[1:ncon] = J(x) * v
#   vjp! (Jtv, x, w)    — fills Jtv[1:nvar] = J(x)' * w
#   hvp! (Hv, x, w, v)  — fills Hv[1:nvar] = (Σ wᵢ ∇²fᵢ) * v
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
- `nnzj`: number of Jacobian nonzeros (0 if matrix-free only; dense structure declared automatically)
- `nnzh`: number of Lagrangian Hessian nonzeros (0 to use L-BFGS / finite differences, or matrix-free)
- `jac_rows`, `jac_cols`: sparsity pattern of the Jacobian (1-based, length `nnzj`)
- `hess_rows`, `hess_cols`: sparsity pattern of the (lower-triangular) Hessian (1-based, length `nnzh`)
- `lcon`, `ucon`: constraint lower/upper bounds (length `ncon`)
- `f!`: `f!(c, x)` — writes residuals into `c[1:ncon]`
- `jac!`: `jac!(vals, x)` — writes Jacobian values into `vals[1:nnzj]` (nothing if matrix-free only)
- `hess!`: `hess!(vals, x, y)` — writes Hessian values into `vals[1:nnzh]` (nothing if matrix-free only)
- `jvp!`: `jvp!(Jv, x, v)` — Jacobian-vector product: `Jv[1:ncon] = J(x) * v` (nothing if explicit only)
- `vjp!`: `vjp!(Jtv, x, w)` — transpose Jacobian-vector product: `Jtv[1:nvar] = J(x)' * w` (nothing if explicit only)
- `hvp!`: `hvp!(Hv, x, w, v)` — Hessian-vector product: `Hv[1:nvar] = (Σ wᵢ ∇²fᵢ) * v` (nothing if explicit only)
- `gpu`: if `true`, callbacks receive device arrays (e.g. `CuArray`) directly — no CPU round-trip.
  Use this when your callbacks are already GPU-capable (CUDA kernels, CuDSS, etc.).
  Default is `false` (CPU bridge: arrays are `Array`-copied before every call).
"""
struct VectorNonlinearOracle{F, J, H, JVP, VJP, HVP, VT <: AbstractVector}
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
    jac!::J                  # nothing if matrix-free only
    hess!::H                 # nothing if matrix-free only
    jvp!::JVP                # nothing if explicit only
    vjp!::VJP                # nothing if explicit only
    hvp!::HVP                # nothing if explicit only
    gpu::Bool                # true ⟹ callbacks accept device arrays directly
end

function VectorNonlinearOracle(;
        nvar::Int,
        ncon::Int,
        nnzj::Int = -1,    # -1 = auto-detect from jac_rows/jvp!
        nnzh::Int = -1,    # -1 = auto-detect from hess_rows/hvp!
        jac_rows::Vector{Int} = Int[],
        jac_cols::Vector{Int} = Int[],
        hess_rows::Vector{Int} = Int[],
        hess_cols::Vector{Int} = Int[],
        lcon::AbstractVector = zeros(ncon),
        ucon::AbstractVector = zeros(ncon),
        f!,
        jac! = nothing,
        hess! = nothing,
        jvp! = nothing,
        vjp! = nothing,
        hvp! = nothing,
        gpu::Bool = false,
    )
    # Validate: at least one Jacobian path must be provided
    has_explicit_jac = jac! !== nothing
    has_matfree_jac = jvp! !== nothing && vjp! !== nothing
    @assert has_explicit_jac || has_matfree_jac (
        "At least one of `jac!` or (`jvp!` + `vjp!`) must be provided"
    )

    # Auto-detect nnzj/nnzh when not explicitly provided
    if nnzj == -1
        nnzj = length(jac_rows)
        # Matrix-free only: declare dense Jacobian structure for NLPModelMeta
        if !has_explicit_jac && nnzj == 0
            nnzj = ncon * nvar
            jac_rows = vec([i for i in 1:ncon, _ in 1:nvar])
            jac_cols = vec([j for _ in 1:ncon, j in 1:nvar])
        end
    end
    if nnzh == -1
        nnzh = length(hess_rows)
        # Matrix-free only with hvp!: declare dense lower-triangular Hessian
        if jac! === nothing && hess! === nothing && hvp! !== nothing && nnzh == 0
            nnzh = nvar * (nvar + 1) ÷ 2
            hess_rows = Int[]
            hess_cols = Int[]
            sizehint!(hess_rows, nnzh)
            sizehint!(hess_cols, nnzh)
            for i in 1:nvar, j in 1:i
                push!(hess_rows, i)
                push!(hess_cols, j)
            end
        end
    end

    # Default hess! callback when not provided and no hvp! either
    if hess! === nothing && hvp! === nothing
        hess! = (vals, x, y) -> nothing
    end

    @assert length(jac_rows) == nnzj "jac_rows length must equal nnzj"
    @assert length(jac_cols) == nnzj "jac_cols length must equal nnzj"
    @assert length(hess_rows) == nnzh "hess_rows length must equal nnzh"
    @assert length(hess_cols) == nnzh "hess_cols length must equal nnzh"
    @assert length(lcon) == ncon "lcon length must equal ncon"
    @assert length(ucon) == ncon "ucon length must equal ncon"
    return VectorNonlinearOracle(
        nvar, ncon, nnzj, nnzh,
        jac_rows, jac_cols,
        hess_rows, hess_cols,
        lcon, ucon,
        f!, jac!, hess!,
        jvp!, vjp!, hvp!,
        gpu,
    )
end

Base.show(io::IO, o::VectorNonlinearOracle) = print(
    io,
    """
    VectorNonlinearOracle

      ncon: $(o.ncon)   nnzj: $(o.nnzj)   nnzh: $(o.nnzh)   gpu: $(o.gpu)
      jac!: $(o.jac! !== nothing)   jvp!: $(o.jvp! !== nothing)   vjp!: $(o.vjp! !== nothing)   hvp!: $(o.hvp! !== nothing)
    """,
)

"""
    has_matfree_jac(oracle::VectorNonlinearOracle)

Return `true` if the oracle provides matrix-free Jacobian-vector products
(`jvp!` and `vjp!`).
"""
has_matfree_jac(o::VectorNonlinearOracle) =
    o.jvp! !== nothing && o.vjp! !== nothing

"""
    has_matfree_hess(oracle::VectorNonlinearOracle)

Return `true` if the oracle provides a matrix-free Hessian-vector product (`hvp!`).
"""
has_matfree_hess(o::VectorNonlinearOracle) = o.hvp! !== nothing


# ── ScalarNonlinearOracle ─────────────────────────────────────────────────────

"""
    ScalarNonlinearOracle

A scalar objective oracle: adds `f(x)` to the objective via callbacks.
No output variables, no constraints, no Jacobian — only gradient and
optionally Hessian-vector product.

Callbacks:
- `f(x) -> Float64`:           objective value
- `grad!(g, x)`:              fills g[1:nvar] = ∇f(x)
- `hvp!(Hv, x, v) = nothing`: fills Hv[1:nvar] = ∇²f(x)·v  (optional; L-BFGS if absent)

Register via `objective(core, oracle)`.
"""
struct ScalarNonlinearOracle{F, G, H}
    nvar::Int
    f::F
    grad!::G
    hvp!::H
    nnzh::Int
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
    gpu::Bool
end

function ScalarNonlinearOracle(;
        nvar::Int,
        f,
        grad!,
        hvp! = nothing,
        nnzh::Int = -1,
        hess_rows::Vector{Int} = Int[],
        hess_cols::Vector{Int} = Int[],
        gpu::Bool = false,
    )
    if nnzh == -1
        nnzh = length(hess_rows)
        # Matrix-free with hvp!: declare dense lower-triangular Hessian
        if hvp! !== nothing && nnzh == 0
            nnzh = nvar * (nvar + 1) ÷ 2
            hess_rows = Int[]
            hess_cols = Int[]
            sizehint!(hess_rows, nnzh)
            sizehint!(hess_cols, nnzh)
            for i in 1:nvar, j in 1:i
                push!(hess_rows, i)
                push!(hess_cols, j)
            end
        end
    end
    return ScalarNonlinearOracle(nvar, f, grad!, hvp!, nnzh, hess_rows, hess_cols, gpu)
end

Base.show(io::IO, o::ScalarNonlinearOracle) = print(
    io,
    """
    ScalarNonlinearOracle

      nvar: $(o.nvar)   nnzh: $(o.nnzh)   gpu: $(o.gpu)
      hvp!: $(o.hvp! !== nothing)
    """,
)

_oracle_input(oracle::ScalarNonlinearOracle, x) =
    oracle.gpu ? x : adapt(Array, x)

"""
    objective(core::ExaCore, oracle::ScalarNonlinearOracle)

Register a scalar objective oracle with `core`.
"""
function objective(c::ExaCore, oracle::ScalarNonlinearOracle)
    push!(c.scalar_oracles, oracle)
    return oracle
end


# ── Array-routing helper ─────────────────────────────────────────────────────
# When oracle.gpu == false (default): adapt device arrays to CPU before calling
# the callback.  When oracle.gpu == true: pass arrays through unchanged.
_oracle_input(oracle::VectorNonlinearOracle, x) =
    oracle.gpu ? x : adapt(Array, x)


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
    # All count updates (ncon, nnzj, nnzh) are deferred to _build_with_oracle
    # so that SIMD constraints added *after* the oracle still get contiguous
    # offsets starting from 0.  Bounds are also deferred and appended in
    # _build_with_oracle (oracle constraints always come after SIMD constraints).
    return oracle
end

# ── Model type ────────────────────────────────────────────────────────────────

struct OracleIndexCache{VI <: AbstractVector{Int}, VF <: AbstractVector}
    con_idx::VI             # indices into g/y for this oracle's constraints
    jac_idx::VI             # indices into jac for this oracle's Jacobian entries
    hess_idx::VI            # indices into hess for this oracle's Hessian entries
    # Shifted sparsity for jac/hess_structure!
    jac_rows_shifted::VI    # jac_rows .+ con_offset
    jac_cols_copy::VI       # copy of jac_cols
    hess_rows_copy::VI      # copy of hess_rows
    hess_cols_copy::VI      # copy of hess_cols
    # Per-column reconstruction data (stays CPU — used in scalar loops)
    jac_recon_cols::Vector{Int}
    jac_recon_row_idx::Vector{Vector{Int}}
    jac_recon_pos_idx::Vector{Vector{Int}}
    hess_recon_cols::Vector{Int}
    hess_recon_row_idx::Vector{Vector{Int}}
    hess_recon_pos_idx::Vector{Vector{Int}}
    # Preallocated work buffers (same device as solver arrays)
    buf_ncon::VF            # length ncon — for f!, jvp! results
    buf_nvar::VF            # length nvar — for vjp!, hvp! results, e_col vectors
    buf_nvar2::VF           # length nvar — second nvar buffer (for hess reconstruct)
    buf_nnzj::VF            # length nnzj — for jac! results
    buf_nnzh::VF            # length nnzh — for hess! results
    # CPU-side buffers for gpu=false callbacks on GPU backends
    cpu_ncon::Vector{Float64}
    cpu_nvar::Vector{Float64}
    cpu_nvar2::Vector{Float64}
end

function _build_oracle_index_cache(oracle, con_off, jac_off, hess_off)
    con_idx = collect((con_off + 1):(con_off + oracle.ncon))
    jac_idx = collect((jac_off + 1):(jac_off + oracle.nnzj))
    hess_idx = collect((hess_off + 1):(hess_off + oracle.nnzh))
    jac_rows_shifted = oracle.jac_rows .+ con_off
    jac_cols_copy = copy(oracle.jac_cols)
    hess_rows_copy = copy(oracle.hess_rows)
    hess_cols_copy = copy(oracle.hess_cols)

    # Jac reconstruction (for matrix-free oracles)
    jac_col_map = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
    for k in 1:oracle.nnzj
        col = oracle.jac_cols[k]
        rows, pos = get!(jac_col_map, col) do
            (Int[], Int[])
        end
        push!(rows, oracle.jac_rows[k])
        push!(pos, jac_off + k)
    end
    jac_recon_cols = collect(keys(jac_col_map))
    jac_recon_row_idx = [jac_col_map[c][1] for c in jac_recon_cols]
    jac_recon_pos_idx = [jac_col_map[c][2] for c in jac_recon_cols]

    # Hess reconstruction (for matrix-free oracles)
    hess_col_map = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
    for k in 1:oracle.nnzh
        col = oracle.hess_cols[k]
        rows, pos = get!(hess_col_map, col) do
            (Int[], Int[])
        end
        push!(rows, oracle.hess_rows[k])
        push!(pos, hess_off + k)
    end
    hess_recon_cols = collect(keys(hess_col_map))
    hess_recon_row_idx = [hess_col_map[c][1] for c in hess_recon_cols]
    hess_recon_pos_idx = [hess_col_map[c][2] for c in hess_recon_cols]

    return OracleIndexCache(
        con_idx, jac_idx, hess_idx,
        jac_rows_shifted, jac_cols_copy,
        hess_rows_copy, hess_cols_copy,
        jac_recon_cols, jac_recon_row_idx, jac_recon_pos_idx,
        hess_recon_cols, hess_recon_row_idx, hess_recon_pos_idx,
        zeros(oracle.ncon),   # buf_ncon
        zeros(oracle.nvar),   # buf_nvar
        zeros(oracle.nvar),   # buf_nvar2
        zeros(oracle.nnzj),   # buf_nnzj
        zeros(oracle.nnzh),   # buf_nnzh
        zeros(oracle.ncon),   # cpu_ncon
        zeros(oracle.nvar),   # cpu_nvar
        zeros(oracle.nvar),   # cpu_nvar2
    )
end

# Adapt all cache arrays to the target backend (e.g. CuArray for GPU).
_adapt_cache(c::OracleIndexCache, ::Nothing) = c  # CPU: no conversion needed
function _adapt_cache(c::OracleIndexCache, backend)
    cv = v -> convert_array(v, backend)
    return OracleIndexCache(
        cv(c.con_idx), cv(c.jac_idx), cv(c.hess_idx),
        cv(c.jac_rows_shifted), cv(c.jac_cols_copy),
        cv(c.hess_rows_copy), cv(c.hess_cols_copy),
        c.jac_recon_cols,     # column indices stay CPU (loop iteration only)
        c.jac_recon_row_idx,  # per-column row/pos arrays stay CPU (small, loop-indexed)
        c.jac_recon_pos_idx,
        c.hess_recon_cols,
        c.hess_recon_row_idx,
        c.hess_recon_pos_idx,
        cv(c.buf_ncon), cv(c.buf_nvar), cv(c.buf_nvar2),
        cv(c.buf_nnzj), cv(c.buf_nnzh),
        c.cpu_ncon, c.cpu_nvar, c.cpu_nvar2,  # stay on CPU
    )
end

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
struct ExaModelWithOracle{T, VT, E, O, C, S, R, IC, SO} <: AbstractExaModel{T, VT, E}
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
    oracle_caches::IC                # precomputed index arrays (Vector{OracleIndexCache{VI}})
    scalar_oracles::SO               # Tuple of ScalarNonlinearOracle
end

function Base.show(io::IO, m::ExaModelWithOracle{T, VT}) where {T, VT}
    println(io, "An ExaModelWithOracle{$T, $VT, ...}\n")
    return Base.show(io, m.meta)
end

# ── Internal constructor ───────────────────────────────────────────────────────

function _build_with_oracle(c::ExaCore; kwargs...)
    oracles = c.oracles                          # Vector{Any} of VectorNonlinearOracle
    s_oracles = c.scalar_oracles                 # Vector{Any} of ScalarNonlinearOracle

    # SIMD-only counts (oracle contributions deferred to here)
    n_simd_ncon = c.ncon
    n_simd_nnzj = c.nnzj
    n_simd_nnzh = c.nnzh

    # Total oracle contributions
    total_oracle_ncon = isempty(oracles) ? 0 : sum(o.ncon for o in oracles)
    total_oracle_nnzj = isempty(oracles) ? 0 : sum(o.nnzj for o in oracles)
    total_oracle_nnzh = (isempty(oracles) ? 0 : sum(o.nnzh for o in oracles)) +
                        (isempty(s_oracles) ? 0 : sum(o.nnzh for o in s_oracles))

    total_ncon = n_simd_ncon + total_oracle_ncon
    total_nnzj = n_simd_nnzj + total_oracle_nnzj
    total_nnzh = n_simd_nnzh + total_oracle_nnzh

    # Build extended constraint bound arrays without mutating c
    y0   = c.y0
    lcon = c.lcon
    ucon = c.ucon
    for o in oracles
        y0   = append!(c.backend, y0,   zero(eltype(c.θ)), o.ncon)
        lcon = append!(c.backend, lcon, o.lcon, o.ncon)
        ucon = append!(c.backend, ucon, o.ucon, o.ncon)
    end

    # Compute per-oracle offsets (0-based, relative to the full arrays)
    oracle_con_offsets = Vector{Int}(undef, length(oracles))
    oracle_jac_offsets = Vector{Int}(undef, length(oracles))
    oracle_hess_offsets = Vector{Int}(undef, length(oracles))

    con_off = n_simd_ncon
    jac_off = n_simd_nnzj
    hess_off = n_simd_nnzh
    for (i, o) in enumerate(oracles)
        oracle_con_offsets[i] = con_off
        oracle_jac_offsets[i] = jac_off
        oracle_hess_offsets[i] = hess_off
        con_off += o.ncon
        jac_off += o.nnzj
        hess_off += o.nnzh
    end

    meta = NLPModels.NLPModelMeta(
        c.nvar,
        ncon = total_ncon,
        nnzj = total_nnzj,
        nnzh = total_nnzh,
        x0 = c.x0,
        lvar = c.lvar,
        uvar = c.uvar,
        y0 = y0,
        lcon = lcon,
        ucon = ucon,
        minimize = c.minimize,
    )

    # Precompute index caches for zero-allocation NLPModels evaluations.
    oracle_caches = [
        _adapt_cache(
            _build_oracle_index_cache(
                oracles[i], oracle_con_offsets[i],
                oracle_jac_offsets[i], oracle_hess_offsets[i],
            ),
            c.backend,
        )
        for i in eachindex(oracles)
    ]

    return ExaModelWithOracle(
        c.obj,
        c.cons,
        c.θ,
        meta,
        NLPModels.Counters(),
        build_extension(c; kwargs...),
        c.tag,
        Tuple(oracles),
        oracle_con_offsets,
        oracle_jac_offsets,
        oracle_hess_offsets,
        oracle_caches,
        Tuple(s_oracles),
    )
end

# ── NLPModels methods ──────────────────────────────────────────────────────────

# --- objective (with scalar oracle contributions) ---

function obj(m::ExaModelWithOracle, x::AbstractVector)
    val = _obj(m.objs, x, m.θ)
    for oracle in m.scalar_oracles
        xin = _oracle_input(oracle, x)
        val += oracle.f(xin)
    end
    return val
end

function grad!(m::ExaModelWithOracle, x::AbstractVector, f::AbstractVector)
    fill!(f, zero(eltype(f)))
    _grad!(m.objs, x, m.θ, f)
    for oracle in m.scalar_oracles
        xin = _oracle_input(oracle, x)
        g_buf = zeros(eltype(f), oracle.nvar)
        oracle.grad!(g_buf, xin)
        f .+= g_buf
    end
    return f
end

# --- constraint residuals ---

function cons_nln!(m::ExaModelWithOracle, x::AbstractVector, g::AbstractVector)
    fill!(g, zero(eltype(g)))
    _cons_nln!(m.cons, x, m.θ, g)
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        xin = _oracle_input(oracle, x)
        cv = cache.buf_ncon
        oracle.f!(cv, xin)
        g[cache.con_idx] .= cv
    end
    return g
end

# --- Jacobian ---

function jac_structure!(m::ExaModelWithOracle{T}, rows::AbstractVector, cols::AbstractVector) where {T}
    _jac_structure!(T, m.cons, rows, cols)
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        if oracle.nnzj > 0
            rows[cache.jac_idx] .= cache.jac_rows_shifted
            cols[cache.jac_idx] .= cache.jac_cols_copy
        end
    end
    return rows, cols
end

function jac_coord!(m::ExaModelWithOracle, x::AbstractVector, jac::AbstractVector)
    fill!(jac, zero(eltype(jac)))
    _jac_coord!(m.cons, x, m.θ, jac)
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        oracle.nnzj == 0 && continue
        if oracle.jac! !== nothing
            xin = _oracle_input(oracle, x)
            jv = cache.buf_nnzj
            oracle.jac!(jv, xin)
            jac[cache.jac_idx] .= jv
        else
            _jac_reconstruct_via_jvp!(oracle, x, jac, cache)
        end
    end
    return jac
end

"""
    _jac_reconstruct_via_jvp!(oracle, x, jac, cache, backend=nothing)

Reconstruct the Jacobian coordinate values using `nvar` calls to `oracle.jvp!`.
All buffers are preallocated in `cache`. For `gpu=false` on a GPU backend,
CPU scratch arrays are used for the callback and results are copied to device.
"""
function _jac_reconstruct_via_jvp!(oracle, x, jac, cache::OracleIndexCache, backend=nothing)
    T = eltype(x)
    xin = _oracle_input(oracle, x)
    # Pick buffers: device buffers if gpu=true or CPU backend, else CPU scratch.
    use_cpu = !oracle.gpu && backend !== nothing
    v  = use_cpu ? cache.cpu_nvar : cache.buf_nvar
    Jv = use_cpu ? cache.cpu_ncon : cache.buf_ncon
    for (ci, col) in enumerate(cache.jac_recon_cols)
        fill!(v, zero(T))
        v[col:col] .= one(T)
        oracle.jvp!(Jv, xin, v)
        if use_cpu
            copyto!(cache.buf_ncon, Jv)
            jac[cache.jac_recon_pos_idx[ci]] .= cache.buf_ncon[cache.jac_recon_row_idx[ci]]
        else
            jac[cache.jac_recon_pos_idx[ci]] .= Jv[cache.jac_recon_row_idx[ci]]
        end
    end
    return nothing
end

# Jacobian-vector products: prefer matrix-free jvp!/vjp! when available,
# otherwise fall back to sparse accumulation from jac!.

function jprod_nln!(
        m::ExaModelWithOracle,
        x::AbstractVector,
        v::AbstractVector,
        Jv::AbstractVector,
    )
    fill!(Jv, zero(eltype(Jv)))
    _jprod_nln!(m.cons, x, m.θ, v, Jv)
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        xin = _oracle_input(oracle, x)
        vin = _oracle_input(oracle, v)
        if has_matfree_jac(oracle)
            Jv_oracle = cache.buf_ncon
            oracle.jvp!(Jv_oracle, xin, vin)
            Jv[cache.con_idx] .+= Jv_oracle
        else
            off_c = m.oracle_con_offsets[i]
            jac_buf = cache.buf_nnzj
            oracle.jac!(jac_buf, xin)
            jac_cpu = adapt(Array, jac_buf)
            v_cpu = adapt(Array, vin)
            delta = zeros(eltype(jac_cpu), length(Jv))
            for k in 1:oracle.nnzj
                delta[oracle.jac_rows[k] + off_c] += jac_cpu[k] * v_cpu[oracle.jac_cols[k]]
            end
            buf = similar(Jv)
            copyto!(buf, delta)
            Jv .+= buf
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
        cache = m.oracle_caches[i]
        xin = _oracle_input(oracle, x)
        if has_matfree_jac(oracle)
            w = _oracle_input(oracle, v[cache.con_idx])
            Jtv_oracle = cache.buf_nvar
            oracle.vjp!(Jtv_oracle, xin, w)
            Jtv .+= Jtv_oracle
        else
            off_c = m.oracle_con_offsets[i]
            vin = _oracle_input(oracle, v)
            jac_buf = cache.buf_nnzj
            oracle.jac!(jac_buf, xin)
            jac_cpu = adapt(Array, jac_buf)
            v_cpu = adapt(Array, vin)
            delta = zeros(eltype(jac_cpu), length(Jtv))
            for k in 1:oracle.nnzj
                delta[oracle.jac_cols[k]] += jac_cpu[k] * v_cpu[oracle.jac_rows[k] + off_c]
            end
            buf = similar(Jtv)
            copyto!(buf, delta)
            Jtv .+= buf
        end
    end
    return Jtv
end

# --- Hessian ---

function hess_structure!(m::ExaModelWithOracle{T}, rows::AbstractVector, cols::AbstractVector) where {T}
    _obj_hess_structure!(T, m.objs, rows, cols)
    _con_hess_structure!(T, m.cons, rows, cols)
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        if oracle.nnzh > 0
            rows[cache.hess_idx] .= cache.hess_rows_copy
            cols[cache.hess_idx] .= cache.hess_cols_copy
        end
    end
    return rows, cols
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
        cache = m.oracle_caches[i]
        oracle.nnzh == 0 && continue
        if oracle.hess! !== nothing
            xin = _oracle_input(oracle, x)
            win = _oracle_input(oracle, y[cache.con_idx])
            hv = cache.buf_nnzh
            oracle.hess!(hv, xin, win)
            hess[cache.hess_idx] .= hv
        else
            _hess_reconstruct_via_hvp!(oracle, x, y, hess, cache)
        end
    end
    return hess
end

"""
    _hess_reconstruct_via_hvp!(oracle, x, y, hess, cache, backend=nothing)

Reconstruct the lower-triangular Hessian coordinate values using `nvar` calls
to `oracle.hvp!`. All buffers are preallocated in `cache`.
"""
function _hess_reconstruct_via_hvp!(oracle, x, y, hess, cache::OracleIndexCache, backend=nothing)
    T = eltype(x)
    xin = _oracle_input(oracle, x)
    win = _oracle_input(oracle, y[cache.con_idx])
    use_cpu = !oracle.gpu && backend !== nothing
    v  = use_cpu ? cache.cpu_nvar  : cache.buf_nvar
    Hv = use_cpu ? cache.cpu_nvar2 : cache.buf_nvar2
    for (ci, col) in enumerate(cache.hess_recon_cols)
        fill!(v, zero(T))
        v[col:col] .= one(T)
        oracle.hvp!(Hv, xin, win, v)
        if use_cpu
            copyto!(cache.buf_nvar2, Hv)
            hess[cache.hess_recon_pos_idx[ci]] .= cache.buf_nvar2[cache.hess_recon_row_idx[ci]]
        else
            hess[cache.hess_recon_pos_idx[ci]] .= Hv[cache.hess_recon_row_idx[ci]]
        end
    end
    return nothing
end

# hprod! without y (obj-only) is handled by the KA extension for GPU;
# for CPU the nlp.jl AbstractExaModel fallback is used.

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
    for (i, oracle) in enumerate(m.oracles)
        cache = m.oracle_caches[i]
        oracle.nnzh == 0 && continue
        xin = _oracle_input(oracle, x)
        win = _oracle_input(oracle, y[cache.con_idx])
        vin = _oracle_input(oracle, v)
        if has_matfree_hess(oracle)
            Hv_oracle = cache.buf_nvar
            oracle.hvp!(Hv_oracle, xin, win, vin)
            Hv .+= Hv_oracle
        else
            hess_buf = cache.buf_nnzh
            oracle.hess!(hess_buf, xin, win)
            hess_cpu = adapt(Array, hess_buf)
            v_cpu = adapt(Array, vin)
            delta = zeros(eltype(hess_cpu), length(Hv))
            for k in 1:oracle.nnzh
                r, c_ = oracle.hess_rows[k], oracle.hess_cols[k]
                delta[r] += hess_cpu[k] * v_cpu[c_]
                if r != c_
                    delta[c_] += hess_cpu[k] * v_cpu[r]
                end
            end
            buf = similar(Hv)
            copyto!(buf, delta)
            Hv .+= buf
        end
    end
    return Hv
end

# ── High-level embedding API ─────────────────────────────────────────────────

"""
    embed_oracle(core, x, output_dim; f!, jvp!, vjp!, hvp! = nothing, gpu = false)

Embed an opaque vector function `f: ℝⁿ → ℝᵐ` into an `ExaCore` using the
full-space pattern.  Creates auxiliary variables `z ∈ ℝᵐ`, registers an oracle
constraint `z − f(x) = 0`, and returns `z` as a regular `Variable` that can be
used in any SIMD objective or constraint.

The callbacks operate on **local** input/output vectors — no global offsets:

- `f!(y, x)`:         `y[1:m] = f(x[1:n])`
- `jvp!(Jv, x, v)`:   `Jv[1:m] = J_f(x) * v[1:n]`
- `vjp!(Jtv, x, w)`:  `Jtv[1:n] = J_f(x)' * w[1:m]`
- `hvp!(Hv, x, w, v)`: `Hv[1:n] = (Σ wᵢ ∇²fᵢ(x)) * v[1:n]`  (optional)

Returns `(z, oracle)` where `z` is the output `Variable`.

## Example

```julia
core = ExaCore()
x = variable(core, 2; start = [1.0, 1.0])

z, _ = embed_oracle(core, x, 1;
    f!   = (y, xv) -> (y[1] = xv[1]^2 + xv[2]^2; nothing),
    jvp! = (Jv, xv, v) -> (Jv[1] = 2xv[1]*v[1] + 2xv[2]*v[2]; nothing),
    vjp! = (Jtv, xv, w) -> (Jtv[1] = 2xv[1]*w[1]; Jtv[2] = 2xv[2]*w[1]; nothing),
    hvp! = (Hv, xv, w, v) -> (Hv[1] = 2w[1]*v[1]; Hv[2] = 2w[1]*v[2]; nothing),
)

objective(core, z[1])                                    # min f(x)
constraint(core, x[1] + x[2]; lcon = 2.0, ucon = Inf)   # x₁+x₂ ≥ 2
model = ExaModel(core)
```
"""
function embed_oracle(
        core::ExaCore,
        x::AbstractVariable,
        output_dim::Int;
        f!,
        jvp!,
        vjp!,
        hvp! = nothing,
        gpu::Bool = false,
    )
    input_dim = x.length
    x_off = x.offset
    z = variable(core, output_dim)
    z_off = z.offset
    nvar = core.nvar

    # Precompute integer index arrays for GPU-safe gather/scatter.
    x_idx = collect((x_off + 1):(x_off + input_dim))
    z_idx = collect((z_off + 1):(z_off + output_dim))

    # Preallocate scratch buffers (one set per closure group).
    # For GPU, these will be adapted to device arrays at first call via similar().

    _f! = let x_idx = x_idx, z_idx = z_idx
        (c, xv) -> begin
            f!(c, xv[x_idx])
            c .= xv[z_idx] .- c
            nothing
        end
    end

    _jvp! = let x_idx = x_idx, z_idx = z_idx
        (Jv, xv, v) -> begin
            jvp!(Jv, xv[x_idx], v[x_idx])
            Jv .= v[z_idx] .- Jv
            nothing
        end
    end

    _vjp! = let x_idx = x_idx, z_idx = z_idx, input_dim = input_dim
        (Jtv, xv, w) -> begin
            Jtv_x = similar(xv, input_dim)
            vjp!(Jtv_x, xv[x_idx], w)
            fill!(Jtv, zero(eltype(Jtv)))
            Jtv[x_idx] .= .-Jtv_x
            Jtv[z_idx] .= w
            nothing
        end
    end

    _hvp! = if hvp! !== nothing
        let x_idx = x_idx, input_dim = input_dim
            (Hv, xv, w, v) -> begin
                Hv_x = similar(xv, input_dim)
                hvp!(Hv_x, xv[x_idx], w, v[x_idx])
                fill!(Hv, zero(eltype(Hv)))
                Hv[x_idx] .= .-Hv_x
                nothing
            end
        end
    else
        nothing
    end

    oracle = VectorNonlinearOracle(
        nvar = nvar,
        ncon = output_dim,
        f! = _f!,
        jvp! = _jvp!,
        vjp! = _vjp!,
        hvp! = _hvp!,
        gpu = gpu,
    )
    constraint(core, oracle)
    return z, oracle
end
