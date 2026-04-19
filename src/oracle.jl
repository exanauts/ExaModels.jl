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
- `adapt`: `Val(false)` (default) — callbacks receive arrays as-is (device arrays on GPU backends).
  `Val(true)` — arrays are copied to CPU (`Array`) before every call.
  Use `Val(true)` when your callbacks are CPU-only; use `Val(false)` when they are GPU-capable.
"""
struct VectorNonlinearOracle{F, J, H, JVP, VJP, HVP, VT <: AbstractVector, A}
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
    adapt::A                 # Val(false) ⟹ pass arrays through; Val(true) ⟹ copy to CPU
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
        adapt::Val = Val(false),
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
        adapt,
    )
end

Base.show(io::IO, o::VectorNonlinearOracle) = print(
    io,
    """
    VectorNonlinearOracle

      ncon: $(o.ncon)   nnzj: $(o.nnzj)   nnzh: $(o.nnzh)   adapt: $(o.adapt)
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
struct ScalarNonlinearOracle{F, G, H, A}
    nvar::Int
    f::F
    grad!::G
    hvp!::H
    nnzh::Int
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
    adapt::A
end

function ScalarNonlinearOracle(;
        nvar::Int,
        f,
        grad!,
        hvp! = nothing,
        nnzh::Int = -1,
        hess_rows::Vector{Int} = Int[],
        hess_cols::Vector{Int} = Int[],
        adapt::Val = Val(false),
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
    return ScalarNonlinearOracle(nvar, f, grad!, hvp!, nnzh, hess_rows, hess_cols, adapt)
end

Base.show(io::IO, o::ScalarNonlinearOracle) = print(
    io,
    """
    ScalarNonlinearOracle

      nvar: $(o.nvar)   nnzh: $(o.nnzh)   adapt: $(o.adapt)
      hvp!: $(o.hvp! !== nothing)
    """,
)

_oracle_input(oracle::ScalarNonlinearOracle, x) = _do_adapt(oracle.adapt, x)

"""
    objective(core::ExaCore, oracle::ScalarNonlinearOracle)

Register a scalar objective oracle with `core`.
"""
function objective(c::ExaCore, oracle::ScalarNonlinearOracle)
    return ExaCore(c; scalar_oracles = (c.scalar_oracles..., oracle))
end


# ── Array-routing helpers ────────────────────────────────────────────────────
# Val(false) (default): pass arrays through unchanged (GPU-capable callbacks).
# Val(true): copy to CPU Array before calling the callback.
_do_adapt(::Val{false}, x) = x
_do_adapt(::Val{true},  x) = adapt(Array, x)

_oracle_input(oracle::VectorNonlinearOracle, x) = _do_adapt(oracle.adapt, x)


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
    # Immutable functional style: return a new ExaCore with the oracle appended.
    # Count updates (ncon, nnzj, nnzh) are deferred to _build_with_oracle so that
    # SIMD constraints added *after* the oracle still get contiguous offsets.
    return ExaCore(c; oracles = (c.oracles..., oracle))
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
    # CPU-side buffers for adapt=Val(true) callbacks on GPU backends
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

# ── OracleEvaluator: augment pre-existing constraint rows via callbacks ───────

"""
    OracleEvaluator

Encapsulates callbacks that augment a set of pre-existing constraint rows
(declared via [`add_con`](@ref)) with values from an opaque function.
Registered via [`add_eval`](@ref) / [`@add_con!`](@ref) (oracle form).

Unlike [`VectorNonlinearOracle`](@ref), an `OracleEvaluator` does **not**
introduce new constraint rows; it fills rows that already exist.

Callbacks operate on **local** concatenated vectors:
- `f!(res, x_local)`:            `res[1:ncon_total] = f(x_local[1:nvar_total])`
- `jac!(vals, x_local)`:         fills `vals[1:nnzj]` with Jacobian values
- `hess!(vals, x_local, y_local)`:fills `vals[1:nnzh]` with Lagrangian Hessian values
- `jvp!(Jv, x_local, v_local)`:  `Jv[1:ncon_total] = J(x) * v_local`
- `vjp!(Jtv, x_local, w_local)`: `Jtv[1:nvar_total] = J(x)' * w_local`
- `hvp!(Hv, x_local, w_local, v_local)`: `Hv[1:nvar_total] = (∑ wᵢ ∇²gᵢ) * v_local`
"""
struct OracleEvaluator{F, J, H, JVP, VJP, HVP, A}
    nvar_total::Int        # total local variables  (sum of v.length for each v)
    ncon_total::Int        # total local constraints (sum of length(c.itr) for each c)
    nnzj::Int
    nnzh::Int
    jac_rows::Vector{Int}  # local 1-based row indices (1:ncon_total)
    jac_cols::Vector{Int}  # local 1-based col indices (1:nvar_total)
    hess_rows::Vector{Int} # local 1-based row indices, lower triangle
    hess_cols::Vector{Int} # local 1-based col indices
    con_global_idx::Vector{Int}  # 1-based global g-indices (length ncon_total)
    var_global_idx::Vector{Int}  # 1-based global x-indices (length nvar_total)
    f!::F
    jac!::J
    hess!::H
    jvp!::JVP
    vjp!::VJP
    hvp!::HVP
    adapt::A
end

struct EvalIndexCache{VI <: AbstractVector{Int}, VF <: AbstractVector}
    con_global_idx::VI       # device: 1-based global g-indices
    var_global_idx::VI       # device: 1-based global x-indices
    jac_idx::VI              # positions in global jac NNZ array
    hess_idx::VI             # positions in global hess NNZ array
    jac_rows_global::VI      # global row indices for jac entries
    jac_cols_global::VI      # global col indices for jac entries
    hess_rows_global::VI     # global row indices for hess entries
    hess_cols_global::VI     # global col indices for hess entries
    buf_ncon::VF
    buf_nvar::VF
    buf_nnzj::VF
    buf_nnzh::VF
end

function _build_eval_index_cache(ev::OracleEvaluator, jac_off::Int, hess_off::Int)
    jac_idx  = isempty(ev.jac_rows)  ? Int[] : collect((jac_off  + 1):(jac_off  + ev.nnzj))
    hess_idx = isempty(ev.hess_rows) ? Int[] : collect((hess_off + 1):(hess_off + ev.nnzh))
    jac_rows_global  = isempty(ev.jac_rows)  ? Int[] : ev.con_global_idx[ev.jac_rows]
    jac_cols_global  = isempty(ev.jac_cols)  ? Int[] : ev.var_global_idx[ev.jac_cols]
    hess_rows_global = isempty(ev.hess_rows) ? Int[] : ev.var_global_idx[ev.hess_rows]
    hess_cols_global = isempty(ev.hess_cols) ? Int[] : ev.var_global_idx[ev.hess_cols]
    return EvalIndexCache(
        copy(ev.con_global_idx),
        copy(ev.var_global_idx),
        jac_idx, hess_idx,
        jac_rows_global, jac_cols_global,
        hess_rows_global, hess_cols_global,
        zeros(ev.ncon_total),
        zeros(ev.nvar_total),
        zeros(ev.nnzj),
        zeros(ev.nnzh),
    )
end

_adapt_eval_cache(c::EvalIndexCache, ::Nothing) = c
function _adapt_eval_cache(c::EvalIndexCache, backend)
    cv = v -> convert_array(v, backend)
    return EvalIndexCache(
        cv(c.con_global_idx), cv(c.var_global_idx),
        cv(c.jac_idx), cv(c.hess_idx),
        cv(c.jac_rows_global), cv(c.jac_cols_global),
        cv(c.hess_rows_global), cv(c.hess_cols_global),
        cv(c.buf_ncon), cv(c.buf_nvar),
        cv(c.buf_nnzj), cv(c.buf_nnzh),
    )
end

_eval_input(ev::OracleEvaluator, x) = _do_adapt(ev.adapt, x)

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
struct ExaModelWithOracle{T, VT, E, O, C, S, R, IC, SO, EV, EC} <: AbstractExaModel{T, VT, E}
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
    evals::EV                        # Tuple of OracleEvaluator
    eval_caches::EC                  # Vector{EvalIndexCache}
end

function Base.show(io::IO, m::ExaModelWithOracle{T, VT}) where {T, VT}
    println(io, "An ExaModelWithOracle{$T, $VT, ...}\n")
    return Base.show(io, m.meta)
end

# ── Internal constructor ───────────────────────────────────────────────────────

function _build_with_oracle(c::ExaCore; kwargs...)
    oracles = c.oracles                          # Tuple of VectorNonlinearOracle
    s_oracles = c.scalar_oracles                 # Tuple of ScalarNonlinearOracle
    evaluators = c.evals                         # Tuple of OracleEvaluator

    # SIMD-only counts (oracle contributions deferred to here)
    n_simd_ncon = c.ncon
    n_simd_nnzj = c.nnzj
    n_simd_nnzh = c.nnzh

    # Total oracle contributions (oracles add rows; evaluators add NNZ only)
    total_oracle_ncon = isempty(oracles) ? 0 : sum(o.ncon for o in oracles)
    total_oracle_nnzj = isempty(oracles) ? 0 : sum(o.nnzj for o in oracles)
    total_oracle_nnzh = (isempty(oracles) ? 0 : sum(o.nnzh for o in oracles)) +
                        (isempty(s_oracles) ? 0 : sum(o.nnzh for o in s_oracles))
    total_eval_nnzj = isempty(evaluators) ? 0 : sum(e.nnzj for e in evaluators)
    total_eval_nnzh = isempty(evaluators) ? 0 : sum(e.nnzh for e in evaluators)

    total_ncon = n_simd_ncon + total_oracle_ncon
    total_nnzj = n_simd_nnzj + total_oracle_nnzj + total_eval_nnzj
    total_nnzh = n_simd_nnzh + total_oracle_nnzh + total_eval_nnzh

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

    # Evaluator NNZ offsets (after oracle NNZ)
    eval_jac_off  = n_simd_nnzj + total_oracle_nnzj
    eval_hess_off = n_simd_nnzh + total_oracle_nnzh

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

    # Precompute eval caches
    eval_jac_cursor  = eval_jac_off
    eval_hess_cursor = eval_hess_off
    eval_caches = [
        begin
            cache = _adapt_eval_cache(
                _build_eval_index_cache(evaluators[i], eval_jac_cursor, eval_hess_cursor),
                c.backend,
            )
            eval_jac_cursor  += evaluators[i].nnzj
            eval_hess_cursor += evaluators[i].nnzh
            cache
        end
        for i in eachindex(evaluators)
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
        Tuple(evaluators),
        eval_caches,
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
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        xin = _eval_input(ev, x[cache.var_global_idx])
        ev.f!(cache.buf_ncon, xin)
        g[cache.con_global_idx] .+= cache.buf_ncon
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
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        if ev.nnzj > 0
            rows[cache.jac_idx] .= cache.jac_rows_global
            cols[cache.jac_idx] .= cache.jac_cols_global
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
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        ev.nnzj == 0 && continue
        xin = _eval_input(ev, x[cache.var_global_idx])
        if ev.jac! !== nothing
            ev.jac!(cache.buf_nnzj, xin)
            jac[cache.jac_idx] .= cache.buf_nnzj
        elseif ev.jvp! !== nothing
            _eval_jac_reconstruct_via_jvp!(ev, xin, jac, cache)
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
    # Pick buffers: device buffers if adapt=Val(false) or CPU backend, else CPU scratch.
    use_cpu = oracle.adapt === Val(true) && backend !== nothing
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
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        xin = _eval_input(ev, x[cache.var_global_idx])
        vin_local = _eval_input(ev, v[cache.var_global_idx])
        if ev.jvp! !== nothing
            ev.jvp!(cache.buf_ncon, xin, vin_local)
            Jv[cache.con_global_idx] .+= cache.buf_ncon
        elseif ev.jac! !== nothing
            ev.jac!(cache.buf_nnzj, xin)
            jac_cpu = adapt(Array, cache.buf_nnzj)
            v_cpu   = adapt(Array, vin_local)
            delta   = zeros(eltype(jac_cpu), ev.ncon_total)
            for k in 1:ev.nnzj
                delta[ev.jac_rows[k]] += jac_cpu[k] * v_cpu[ev.jac_cols[k]]
            end
            con_cpu = adapt(Array, cache.con_global_idx)
            for k in 1:ev.ncon_total
                Jv[con_cpu[k]] += delta[k]
            end
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
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        xin   = _eval_input(ev, x[cache.var_global_idx])
        w_local = _eval_input(ev, v[cache.con_global_idx])
        if ev.vjp! !== nothing
            ev.vjp!(cache.buf_nvar, xin, w_local)
            Jtv[cache.var_global_idx] .+= cache.buf_nvar
        elseif ev.jac! !== nothing
            ev.jac!(cache.buf_nnzj, xin)
            jac_cpu = adapt(Array, cache.buf_nnzj)
            w_cpu   = adapt(Array, w_local)
            delta   = zeros(eltype(jac_cpu), ev.nvar_total)
            for k in 1:ev.nnzj
                delta[ev.jac_cols[k]] += jac_cpu[k] * w_cpu[ev.jac_rows[k]]
            end
            var_cpu = adapt(Array, cache.var_global_idx)
            for k in 1:ev.nvar_total
                Jtv[var_cpu[k]] += delta[k]
            end
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
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        if ev.nnzh > 0
            rows[cache.hess_idx] .= cache.hess_rows_global
            cols[cache.hess_idx] .= cache.hess_cols_global
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
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        ev.nnzh == 0 && continue
        xin = _eval_input(ev, x[cache.var_global_idx])
        win = _eval_input(ev, y[cache.con_global_idx])
        if ev.hess! !== nothing
            ev.hess!(cache.buf_nnzh, xin, win)
            hess[cache.hess_idx] .= cache.buf_nnzh
        elseif ev.hvp! !== nothing
            _eval_hess_reconstruct_via_hvp!(ev, xin, win, hess, cache)
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
    use_cpu = oracle.adapt === Val(true) && backend !== nothing
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

function _eval_jac_reconstruct_via_jvp!(ev::OracleEvaluator, xin, jac, cache::EvalIndexCache)
    T = eltype(xin)
    v  = cache.buf_nvar
    Jv = cache.buf_ncon
    jac_rows_cpu = ev.jac_rows
    jac_cols_cpu = ev.jac_cols
    jac_idx_cpu  = adapt(Array, cache.jac_idx)
    for col in unique(jac_cols_cpu)
        fill!(v, zero(T))
        v[col:col] .= one(T)
        ev.jvp!(Jv, xin, v)
        for k in 1:ev.nnzj
            if jac_cols_cpu[k] == col
                jac[jac_idx_cpu[k]:jac_idx_cpu[k]] .= Jv[jac_rows_cpu[k]:jac_rows_cpu[k]]
            end
        end
    end
    return nothing
end

function _eval_hess_reconstruct_via_hvp!(ev::OracleEvaluator, xin, win, hess, cache::EvalIndexCache)
    T = eltype(xin)
    v  = cache.buf_nvar
    Hv = cache.buf_nvar  # reuse buf — overwritten per column
    hess_rows_cpu = ev.hess_rows
    hess_cols_cpu = ev.hess_cols
    hess_idx_cpu  = adapt(Array, cache.hess_idx)
    for col in unique(hess_cols_cpu)
        fill!(v, zero(T))
        v[col:col] .= one(T)
        ev.hvp!(Hv, xin, win, v)
        for k in 1:ev.nnzh
            if hess_cols_cpu[k] == col
                hess[hess_idx_cpu[k]:hess_idx_cpu[k]] .= Hv[hess_rows_cpu[k]:hess_rows_cpu[k]]
            end
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
    for (i, ev) in enumerate(m.evals)
        cache = m.eval_caches[i]
        ev.nnzh == 0 && continue
        xin   = _eval_input(ev, x[cache.var_global_idx])
        win   = _eval_input(ev, y[cache.con_global_idx])
        vin   = _eval_input(ev, v[cache.var_global_idx])
        if ev.hvp! !== nothing
            ev.hvp!(cache.buf_nvar, xin, win, vin)
            Hv[cache.var_global_idx] .+= cache.buf_nvar
        elseif ev.hess! !== nothing
            ev.hess!(cache.buf_nnzh, xin, win)
            hess_cpu = adapt(Array, cache.buf_nnzh)
            v_cpu    = adapt(Array, vin)
            delta    = zeros(eltype(hess_cpu), ev.nvar_total)
            for k in 1:ev.nnzh
                r, c_ = ev.hess_rows[k], ev.hess_cols[k]
                delta[r] += hess_cpu[k] * v_cpu[c_]
                if r != c_
                    delta[c_] += hess_cpu[k] * v_cpu[r]
                end
            end
            var_cpu = adapt(Array, cache.var_global_idx)
            for k in 1:ev.nvar_total
                Hv[var_cpu[k]] += delta[k]
            end
        end
    end
    return Hv
end

"""
    add_eval(core, cons, vars, f!; jac!, hess!, jvp!, vjp!, hvp!,
             jac_structure!, hess_structure!, nnzj=-1, nnzh=-1, adapt=Val(false))

Register an oracle evaluator that augments pre-existing constraint rows.

- `cons`: a tuple of [`Constraint`](@ref) handles whose rows will be filled.
- `vars`: a tuple of [`Variable`](@ref) handles whose values are passed to the callbacks.
- `f!(res, x_local)`: fills `res[1:ncon_total]` given `x_local[1:nvar_total]`.

Sparsity is declared via `jac_structure!(rows, cols)` / `hess_structure!(rows, cols)`,
called once at construction with pre-allocated local-index vectors; or pass `nnzj`/`nnzh`
directly.  Passing `nnzj=-1` with no `jac_structure!` triggers a dense fallback.

Returns `(core, evaluator)`.

## Example

```julia
core = ExaCore(concrete = Val(true))
@add_var(core, x, 4)
c = @add_con(core, 0.0 * x[i] for i in 1:4; lcon = 0.0, ucon = 0.0)

@add_con!(core, (c,), (x,), (res, xv) -> (res .= xv .^ 2; nothing);
    jac! = (vals, xv) -> (vals .= 2 .* xv; nothing),
    nnzj = 4,
    jac_structure! = (rows, cols) -> begin
        append!(rows, 1:4); append!(cols, 1:4)
    end,
)
```
"""
function add_eval(
        core::ExaCore,
        cons::Tuple,
        vars::Tuple,
        f!;
        jac! = nothing,
        hess! = nothing,
        jvp! = nothing,
        vjp! = nothing,
        hvp! = nothing,
        jac_structure! = nothing,
        hess_structure! = nothing,
        nnzj::Int = -1,
        nnzh::Int = -1,
        adapt::Val = Val(false),
    )
    ncon_total = sum(length(c.itr) for c in cons)
    nvar_total = sum(v.length for v in vars)

    con_global_idx = Int[]
    for c in cons
        Base.append!(con_global_idx, (c.offset + 1):(c.offset + length(c.itr)))
    end
    var_global_idx = Int[]
    for v in vars
        Base.append!(var_global_idx, (v.offset + 1):(v.offset + v.length))
    end

    # Resolve Jacobian sparsity
    if nnzj == -1
        if jac_structure! !== nothing
            jac_rows = Int[]; jac_cols = Int[]
            jac_structure!(jac_rows, jac_cols)
            nnzj = length(jac_rows)
        else
            nnzj = ncon_total * nvar_total
            jac_rows = vec([i for i in 1:ncon_total, _ in 1:nvar_total])
            jac_cols = vec([j for _ in 1:ncon_total, j in 1:nvar_total])
        end
    else
        if jac_structure! !== nothing
            jac_rows = Int[]; jac_cols = Int[]
            jac_structure!(jac_rows, jac_cols)
        else
            jac_rows = vec([i for i in 1:ncon_total, _ in 1:nvar_total])
            jac_cols = vec([j for _ in 1:ncon_total, j in 1:nvar_total])
            resize!(jac_rows, nnzj); resize!(jac_cols, nnzj)
        end
    end

    # Resolve Hessian sparsity
    if nnzh == -1
        if hess_structure! !== nothing
            hess_rows = Int[]; hess_cols = Int[]
            hess_structure!(hess_rows, hess_cols)
            nnzh = length(hess_rows)
        elseif hess! !== nothing || hvp! !== nothing
            nnzh = nvar_total * (nvar_total + 1) ÷ 2
            hess_rows = Int[]; hess_cols = Int[]
            sizehint!(hess_rows, nnzh); sizehint!(hess_cols, nnzh)
            for i in 1:nvar_total, j in 1:i
                push!(hess_rows, i); push!(hess_cols, j)
            end
        else
            nnzh = 0
            hess_rows = Int[]; hess_cols = Int[]
        end
    else
        if hess_structure! !== nothing
            hess_rows = Int[]; hess_cols = Int[]
            hess_structure!(hess_rows, hess_cols)
        else
            hess_rows = Int[]; hess_cols = Int[]
            if nnzh > 0
                for i in 1:nvar_total, j in 1:i
                    push!(hess_rows, i); push!(hess_cols, j)
                    length(hess_rows) == nnzh && break
                end
            end
        end
    end

    evaluator = OracleEvaluator(
        nvar_total, ncon_total, nnzj, nnzh,
        jac_rows, jac_cols, hess_rows, hess_cols,
        con_global_idx, var_global_idx,
        f!, jac!, hess!, jvp!, vjp!, hvp!, adapt,
    )
    return ExaCore(core; evals = (core.evals..., evaluator)), evaluator
end

# ── High-level embedding API ─────────────────────────────────────────────────

"""
    embed_oracle(core, x, output_dim; f!, jvp!, vjp!, hvp! = nothing, adapt = Val(false))

Embed an opaque vector function `f: ℝⁿ → ℝᵐ` into an `ExaCore` using the
full-space pattern.  Creates auxiliary variables `z ∈ ℝᵐ`, registers an oracle
constraint `z − f(x) = 0`, and returns the updated core together with `z`.

The callbacks operate on **local** input/output vectors — no global offsets:

- `f!(y, x)`:         `y[1:m] = f(x[1:n])`
- `jvp!(Jv, x, v)`:   `Jv[1:m] = J_f(x) * v[1:n]`
- `vjp!(Jtv, x, w)`:  `Jtv[1:n] = J_f(x)' * w[1:m]`
- `hvp!(Hv, x, w, v)`: `Hv[1:n] = (Σ wᵢ ∇²fᵢ(x)) * v[1:n]`  (optional)

Returns `(core, z, oracle)` where `core` is the updated core and `z` is the
output `Variable`.

## Example

```julia
core = ExaCore(concrete = Val(true))
@add_var(core, x, 2; start = [1.0, 1.0])

core, z, _ = embed_oracle(core, x, 1;
    f!   = (y, xv) -> (y[1] = xv[1]^2 + xv[2]^2; nothing),
    jvp! = (Jv, xv, v) -> (Jv[1] = 2xv[1]*v[1] + 2xv[2]*v[2]; nothing),
    vjp! = (Jtv, xv, w) -> (Jtv[1] = 2xv[1]*w[1]; Jtv[2] = 2xv[2]*w[1]; nothing),
    hvp! = (Hv, xv, w, v) -> (Hv[1] = 2w[1]*v[1]; Hv[2] = 2w[1]*v[2]; nothing),
)

@add_obj(core, z[1])                                          # min f(x)
@add_con(core, x[1] + x[2]; lcon = 2.0, ucon = Inf)          # x₁+x₂ ≥ 2
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
        adapt::Val = Val(false),
    )
    input_dim = x.length
    x_off = x.offset
    core, z = add_var(core, output_dim)
    z_off = z.offset
    nvar = core.nvar

    # Precompute integer index arrays for GPU-safe gather/scatter.
    x_idx = collect((x_off + 1):(x_off + input_dim))
    z_idx = collect((z_off + 1):(z_off + output_dim))

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
        adapt = adapt,
    )
    core = constraint(core, oracle)
    return core, z, oracle
end
