"""
    Stochastic optimization support for ExaModels.

This module provides block-structured evaluation for two-stage optimization
problems with block Jacobian and Hessian structure.

All scenarios share ONE compiled expression pattern, achieving true SIMD parallelism.
This requires scenarios to have identical structure. The builder receives all
variables/parameters and must use generators that iterate over scenario data.

# Variable Structure
- `ns`: Number of scenarios
- `nv`: Number of recourse variables per scenario (second-stage)
- `nd`: Number of design variables (shared/first-stage)
- Total variables: `ns * nv + nd`

# Constraint Structure
- `nc`: Number of constraints per scenario
- Each scenario's constraints depend on its recourse variables (nv) and design variables (nd)
- Total constraints: `ns * nc`

# Global Indexing Convention
- Variables: [v₁; v₂; ...; vₛ; d]
- Constraints: [c₁; c₂; ...; cₛ]
"""

"""
    TwoStageExaModel{T, VT, M}

A two-stage optimization model where all scenarios are fused into a single ExaModel.
Evaluates all scenarios in ONE kernel launch.

# Fields
- `model::M`: Single fused ExaModel containing all scenarios
- `ns::Int`: Number of scenarios
- `nv::Int`: Recourse variables per scenario
- `nd::Int`: Design (shared) variables
- `nc::Int`: Constraints per scenario
- `nθ::Int`: Parameters per scenario
- `nnzj_per_scenario::Int`: Jacobian nonzeros per scenario (approximate)
- `nnzh_per_scenario::Int`: Hessian nonzeros per scenario (approximate)

# Structure
- Total variables: ns*nv + nd
- Total constraints: ns*nc
- Global variable layout: [v₁; v₂; ...; vₛ; d]
- Global constraint layout: [c₁; c₂; ...; cₛ]
"""
struct TwoStageExaModel{T, VT <: AbstractVector{T}, M <: ExaModel{T, VT}}
    model::M
    ns::Int
    nv::Int
    nd::Int
    nc::Int
    nθ::Int
    nnzj_per_scenario::Int
    nnzh_per_scenario::Int
end

# Accessors
num_scenarios(m::TwoStageExaModel) = m.ns
num_recourse_vars(m::TwoStageExaModel) = m.nv
num_design_vars(m::TwoStageExaModel) = m.nd
num_constraints_per_scenario(m::TwoStageExaModel) = m.nc
total_vars(m::TwoStageExaModel) = m.ns * m.nv + m.nd
total_cons(m::TwoStageExaModel) = m.ns * m.nc

function Base.show(io::IO, m::TwoStageExaModel{T, VT}) where {T, VT}
    println(io, "TwoStageExaModel{$T, $VT}")
    println(io, "  Scenarios: $(m.ns)")
    println(io, "  Recourse vars per scenario: $(m.nv)")
    println(io, "  Design vars (shared): $(m.nd)")
    println(io, "  Constraints per scenario: $(m.nc)")
    println(io, "  Total variables: $(total_vars(m))")
    println(io, "  Total constraints: $(total_cons(m))")
    println(io, "  Jacobian nnz per scenario: $(m.nnzj_per_scenario)")
    return println(io, "  Hessian nnz per scenario: $(m.nnzh_per_scenario)")
end

# ============================================================================
# Constructor
# ============================================================================

"""
    TwoStageExaModel(build, nd, nv, ns, θ_sets; backend=nothing)

Build a two-stage model where all scenarios are fused into a single ExaModel.

All scenarios share ONE compiled expression pattern, achieving maximum GPU efficiency.
This requires scenarios to have identical structure.

# Arguments
- `build::Function`: Function `(c, d, v, θ, ns, nv, nθ) -> nothing`
  - `c`: ExaCore
  - `d`: Variable handle for design variables (indices 1:nd)
  - `v`: Variable handle for ALL recourse variables (indices 1:ns*nv)
        Scenario i's vars are at indices (i-1)*nv+1 : i*nv
  - `θ`: Parameter handle for ALL parameters (length ns*nθ)
        Scenario i's params are at indices (i-1)*nθ+1 : i*nθ
  - `ns, nv, nθ`: dimensions for building iteration data
- `nd::Int`: Number of design variables
- `nv::Int`: Number of recourse variables per scenario
- `ns::Int`: Number of scenarios
- `θ_sets::Vector{<:AbstractVector}`: Parameter vectors for each scenario

# Keyword Arguments
- `backend`: Backend for computation (default: `nothing`)
- `d_start`: Initial values for design variables (scalar or vector of length `nd`, default: `0.0`)
- `d_lvar`: Lower bounds for design variables (scalar or vector of length `nd`, default: `-Inf`)
- `d_uvar`: Upper bounds for design variables (scalar or vector of length `nd`, default: `Inf`)
- `v_start`: Initial values for recourse variables (scalar or vector of length `ns*nv`, default: `0.0`)
- `v_lvar`: Lower bounds for recourse variables (scalar or vector of length `ns*nv`, default: `-Inf`)
- `v_uvar`: Upper bounds for recourse variables (scalar or vector of length `ns*nv`, default: `Inf`)

# Example
```julia
ns, nv, nd, nθ = 100, 5, 2, 3
θ_sets = [rand(nθ) for _ in 1:ns]

model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
    obj_data = [(i, j, (i-1)*nv + j, (i-1)*nθ) for i in 1:ns for j in 1:nv]
    objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

    con_data = [(i, j, (i-1)*nv + j, (i-1)*nθ) for i in 1:ns for j in 1:nv]
    constraint(c, v[v_idx] + d[1] - θ[θ_off + 3] for (i, j, v_idx, θ_off) in con_data)
end
```
"""
function TwoStageExaModel(
        build::Function,
        nd::Int,
        nv::Int,
        ns::Int,
        θ_sets::Vector{<:AbstractVector};
        backend = nothing,
        d_start = 0.0,
        d_lvar = -Inf,
        d_uvar = Inf,
        v_start = 0.0,
        v_lvar = -Inf,
        v_uvar = Inf
    )
    length(θ_sets) == ns || throw(ArgumentError("θ_sets must have length ns=$ns"))
    nθ = length(θ_sets[1])
    all(length(θ) == nθ for θ in θ_sets) || throw(ArgumentError("All θ_sets must have same length"))

    c = ExaCore(; backend = backend)

    # All recourse vars as one block, all params as one vector
    v = variable(c, ns * nv; start = v_start, lvar = v_lvar, uvar = v_uvar)
    d = variable(c, nd; start = d_start, lvar = d_lvar, uvar = d_uvar)
    θ_flat = reduce(vcat, θ_sets)
    θ = parameter(c, θ_flat)

    nc_before = c.ncon
    build(c, d, v, θ, ns, nv, nθ)

    # Determine nc per scenario
    nc_total = c.ncon - nc_before
    nc = nc_total ÷ ns
    if nc * ns != nc_total
        throw(
            DimensionMismatch(
                "Total constraints ($nc_total) not evenly divisible by ns ($ns)"
            )
        )
    end

    model = ExaModel(c)

    # Calculate nnz per scenario
    total_nnzj = NLPModels.get_nnzj(model)
    total_nnzh = NLPModels.get_nnzh(model)
    nnzj_per_scenario = total_nnzj ÷ ns
    nnzh_per_scenario = total_nnzh ÷ ns

    T = eltype(c.x0)
    VT = typeof(c.x0)

    return TwoStageExaModel{T, VT, typeof(model)}(
        model, ns, nv, nd, nc, nθ, nnzj_per_scenario, nnzh_per_scenario
    )
end

# ============================================================================
# Full Evaluation (Single Kernel Launch)
# ============================================================================

"""
    obj(model::TwoStageExaModel, x_global)

Evaluate total objective (sum over all scenarios).
"""
function obj(model::TwoStageExaModel, x_global::AbstractVector)
    return obj(model.model, x_global)
end

"""
    cons!(model::TwoStageExaModel, x_global, c_global)

Evaluate all constraints.
Output: c_global ∈ ℝ^{ns*nc}
"""
function cons!(
        model::TwoStageExaModel,
        x_global::AbstractVector,
        c_global::AbstractVector
    )
    cons!(model.model, x_global, c_global)
    return c_global
end

"""
    grad!(model::TwoStageExaModel, x_global, g_global)

Evaluate total gradient.
Output: g_global ∈ ℝ^{ns*nv + nd}
"""
function grad!(
        model::TwoStageExaModel,
        x_global::AbstractVector,
        g_global::AbstractVector
    )
    grad!(model.model, x_global, g_global)
    return g_global
end

"""
    jac_coord!(model::TwoStageExaModel, x_global, jac_global)

Evaluate full Jacobian (COO values).
"""
function jac_coord!(
        model::TwoStageExaModel,
        x_global::AbstractVector,
        jac_global::AbstractVector
    )
    jac_coord!(model.model, x_global, jac_global)
    return jac_global
end

"""
    jac_structure!(model::TwoStageExaModel, rows, cols)

Get full Jacobian sparsity structure.
"""
function jac_structure!(
        model::TwoStageExaModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer}
    )
    jac_structure!(model.model, rows, cols)
    return rows, cols
end

"""
    hess_coord!(model::TwoStageExaModel, x_global, y_global, hess_global; obj_weight=1.0)

Evaluate full Hessian of Lagrangian (COO values).
"""
function hess_coord!(
        model::TwoStageExaModel,
        x_global::AbstractVector,
        y_global::AbstractVector,
        hess_global::AbstractVector;
        obj_weight = one(eltype(x_global))
    )
    hess_coord!(model.model, x_global, y_global, hess_global; obj_weight = obj_weight)
    return hess_global
end

"""
    hess_structure!(model::TwoStageExaModel, rows, cols)

Get full Hessian sparsity structure.
"""
function hess_structure!(
        model::TwoStageExaModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer}
    )
    hess_structure!(model.model, rows, cols)
    return rows, cols
end

# ============================================================================
# NLPModels Interface
# ============================================================================

"""
    get_nnzj(model::TwoStageExaModel)

Total number of Jacobian nonzeros.
"""
NLPModels.get_nnzj(model::TwoStageExaModel) = NLPModels.get_nnzj(model.model)

"""
    get_nnzh(model::TwoStageExaModel)

Total number of Hessian nonzeros.
"""
NLPModels.get_nnzh(model::TwoStageExaModel) = NLPModels.get_nnzh(model.model)

"""
    get_nvar(model::TwoStageExaModel)

Total number of variables.
"""
NLPModels.get_nvar(model::TwoStageExaModel) = NLPModels.get_nvar(model.model)

"""
    get_ncon(model::TwoStageExaModel)

Total number of constraints.
"""
NLPModels.get_ncon(model::TwoStageExaModel) = NLPModels.get_ncon(model.model)

# ============================================================================
# Index Ranges (GPU-friendly, zero allocation)
# ============================================================================

"""
    recourse_var_indices(model::TwoStageExaModel, i) -> UnitRange

Get the index range for recourse variables of scenario i in the global variable vector.
Use as: `x_global[recourse_var_indices(model, i)]`
"""
function recourse_var_indices(model::TwoStageExaModel, i::Int)
    nv = model.nv
    return ((i - 1) * nv + 1):(i * nv)
end

"""
    design_var_indices(model::TwoStageExaModel) -> UnitRange

Get the index range for design variables in the global variable vector.
Use as: `x_global[design_var_indices(model)]`
"""
function design_var_indices(model::TwoStageExaModel)
    ns, nv, nd = model.ns, model.nv, model.nd
    return (ns * nv + 1):(ns * nv + nd)
end

"""
    cons_block_indices(model::TwoStageExaModel, i) -> UnitRange

Get the index range for constraints of scenario i in the global constraint vector.
Use as: `c_global[cons_block_indices(model, i)]`
"""
function cons_block_indices(model::TwoStageExaModel, i::Int)
    nc = model.nc
    return ((i - 1) * nc + 1):(i * nc)
end

"""
    grad_recourse_indices(model::TwoStageExaModel, i) -> UnitRange

Get the index range for recourse gradient of scenario i.
Same as `recourse_var_indices` since gradient has same layout as variables.
"""
grad_recourse_indices(model::TwoStageExaModel, i::Int) = recourse_var_indices(model, i)

"""
    grad_design_indices(model::TwoStageExaModel) -> UnitRange

Get the index range for design gradient.
Same as `design_var_indices` since gradient has same layout as variables.
"""
grad_design_indices(model::TwoStageExaModel) = design_var_indices(model)

# ============================================================================
# In-place Extraction (zero allocation, GPU-friendly)
# ============================================================================

"""
    extract_recourse_vars!(dest, model::TwoStageExaModel, i, x_global)

Extract recourse variables for scenario i into pre-allocated `dest`.
Returns `dest` for convenience.
"""
function extract_recourse_vars!(
        dest::AbstractVector,
        model::TwoStageExaModel,
        i::Int,
        x_global::AbstractVector
    )
    copyto!(dest, 1, x_global, first(recourse_var_indices(model, i)), model.nv)
    return dest
end

"""
    extract_design_vars!(dest, model::TwoStageExaModel, x_global)

Extract design variables into pre-allocated `dest`.
Returns `dest` for convenience.
"""
function extract_design_vars!(
        dest::AbstractVector,
        model::TwoStageExaModel,
        x_global::AbstractVector
    )
    copyto!(dest, 1, x_global, first(design_var_indices(model)), model.nd)
    return dest
end

"""
    extract_cons_block!(dest, model::TwoStageExaModel, i, c_global)

Extract constraint block for scenario i into pre-allocated `dest`.
Returns `dest` for convenience.
"""
function extract_cons_block!(
        dest::AbstractVector,
        model::TwoStageExaModel,
        i::Int,
        c_global::AbstractVector
    )
    copyto!(dest, 1, c_global, first(cons_block_indices(model, i)), model.nc)
    return dest
end

"""
    extract_grad_block!(g_v, g_d, model::TwoStageExaModel, i, g_global)

Extract gradient block for scenario i into pre-allocated `g_v` and `g_d`.
Returns `(g_v, g_d)` for convenience.

Note: The design variable gradient accumulates contributions from all scenarios.
"""
function extract_grad_block!(
        g_v::AbstractVector,
        g_d::AbstractVector,
        model::TwoStageExaModel,
        i::Int,
        g_global::AbstractVector
    )
    copyto!(g_v, 1, g_global, first(grad_recourse_indices(model, i)), model.nv)
    copyto!(g_d, 1, g_global, first(grad_design_indices(model)), model.nd)
    return g_v, g_d
end

# ============================================================================
# Index Mapping
# ============================================================================

"""
    global_var_index(model, i, local_idx) -> global_idx

Convert local variable index to global index for scenario i.

Local ordering (for scenario API): [d₁, ..., d_nd, v₁, ..., v_nv]
Global ordering: [v₁¹...v_nv¹, v₁²...v_nv², ..., v₁ⁿˢ...v_nvⁿˢ, d₁...d_nd]
"""
function global_var_index(model::TwoStageExaModel, i::Int, local_idx::Int)
    nd, nv, ns = model.nd, model.nv, model.ns

    if local_idx <= nd
        # Design variable: maps to end of global vector
        return ns * nv + local_idx
    else
        # Recourse variable: maps to scenario i's block
        return (i - 1) * nv + (local_idx - nd)
    end
end

"""
    global_con_index(model, i, local_idx) -> global_idx

Convert local constraint index to global index for scenario i.
"""
function global_con_index(model::TwoStageExaModel, i::Int, local_idx::Int)
    return (i - 1) * model.nc + local_idx
end

"""
    recourse_var_index(model, i, j) -> global_idx

Get global index for recourse variable j of scenario i.
"""
function recourse_var_index(model::TwoStageExaModel, i::Int, j::Int)
    return (i - 1) * model.nv + j
end

"""
    design_var_index(model, j) -> global_idx

Get global index for design variable j.
"""
function design_var_index(model::TwoStageExaModel, j::Int)
    return model.ns * model.nv + j
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    set_scenario_parameters!(model, i, θ_new)

Update parameters for scenario i.
"""
function set_scenario_parameters!(
        model::TwoStageExaModel,
        i::Int,
        θ_new::AbstractVector
    )
    nθ = model.nθ
    if length(θ_new) != nθ
        throw(
            DimensionMismatch(
                "Parameter size mismatch: expected $nθ, got $(length(θ_new))"
            )
        )
    end

    # Parameters are stored consecutively: [θ₁; θ₂; ...; θₛ]
    θ_start = (i - 1) * nθ + 1
    θ_end = i * nθ
    copyto!(view(model.model.θ, θ_start:θ_end), θ_new)
    return nothing
end

"""
    set_all_scenario_parameters!(model, θ_sets)

Update parameters for all scenarios.
"""
function set_all_scenario_parameters!(
        model::TwoStageExaModel,
        θ_sets::Vector{<:AbstractVector}
    )
    length(θ_sets) == model.ns || throw(
        ArgumentError(
            "θ_sets must have length $(model.ns)"
        )
    )
    for i in 1:model.ns
        set_scenario_parameters!(model, i, θ_sets[i])
    end
    return nothing
end

# ============================================================================
# Access underlying ExaModel
# ============================================================================

"""
    get_model(model::TwoStageExaModel)

Get the underlying ExaModel for direct NLPModels interface usage.
"""
get_model(model::TwoStageExaModel) = model.model
