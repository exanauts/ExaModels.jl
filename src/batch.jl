"""
    BatchExaModel{T, VT, M}

Parametric optimization model where multiple fully independent scenarios are fused
into a single ExaModel and evaluated simultaneously using a shared compiled expression
pattern.

All scenarios must share identical sparsity structures for Jacobians and Hessians
independently of the parameter values. The model builder receives all variables
and parameters and should rely on generators iterating over scenario data.

Unlike `TwoStageExaModel`, there are no shared/design variables. Each scenario is
fully independent.

# Dimensions

- `ns`: number of scenarios
- `nv`: number of variables per scenario
- `nc`: number of constraints per scenario
- `nθ`: number of parameters per scenario

# Layout

- Variables:    [v₁; v₂; …; vₙₛ]
- Constraints: [c₁; c₂; …; cₙₛ]

# Fields

- `model::M` : fused ExaModel containing all scenarios
- `ns::Int`  : number of scenarios
- `nv::Int`  : variables per scenario
- `nc::Int`  : constraints per scenario
- `nθ::Int`  : parameters per scenario
- `nnzj_per_scenario::Int` : Jacobian nonzeros per scenario
- `nnzh_per_scenario::Int` : Hessian nonzeros per scenario
"""
struct BatchExaModel{T, VT <: AbstractVector{T}, M <: ExaModel{T, VT}} <: NLPModels.AbstractNLPModel{T,VT}
    model::M
    ns::Int
    nv::Int
    nc::Int
    nθ::Int
    nnzj_per_scenario::Int
    nnzh_per_scenario::Int
end

function Base.show(io::IO, m::BatchExaModel{T, VT}) where {T, VT}
    println(io, "BatchExaModel{$T, $VT}")
    println(io, "  Scenarios: $(m.ns)")
    println(io, "  Variables per scenario: $(m.nv)")
    println(io, "  Constraints per scenario: $(m.nc)")
    println(io, "  Parameters per scenario: $(m.nθ)")
    println(io, "  Total variables: $(total_vars(m))")
    println(io, "  Total constraints: $(total_cons(m))")
    println(io, "  Jacobian nnz per scenario: $(m.nnzj_per_scenario)")
    return println(io, "  Hessian nnz per scenario: $(m.nnzh_per_scenario)")
end

# ============================================================================
# Constructor
# ============================================================================

"""
    BatchExaModel(build, nv, ns, θ_sets; backend=nothing, kwargs...)

Build a batch model where all fully independent scenarios are fused into a single
ExaModel.

All scenarios share ONE compiled expression pattern, achieving maximum GPU efficiency.
This requires scenarios to have identical structure. Unlike `TwoStageExaModel`, there
are no shared/design variables.

# Arguments
- `build::Function`: Function `(c, v, θ, ns, nv, nθ) -> nothing`
  - `c`: ExaCore
  - `v`: Variable handle for ALL variables (indices 1:ns*nv)
        Scenario i's vars are at indices (i-1)*nv+1 : i*nv
  - `θ`: Parameter handle for ALL parameters (length ns*nθ)
        Scenario i's params are at indices (i-1)*nθ+1 : i*nθ
  - `ns, nv, nθ`: dimensions for building iteration data
- `nv::Int`: Number of variables per scenario
- `ns::Int`: Number of scenarios
- `θ_sets::Vector{<:AbstractVector}`: Parameter vectors for each scenario

# Keyword Arguments
- `backend`: Backend for computation (default: `nothing`)
- `v_start`: Initial values for variables (scalar or vector of length `ns*nv`, default: `0.0`)
- `v_lvar`: Lower bounds for variables (scalar or vector of length `ns*nv`, default: `-Inf`)
- `v_uvar`: Upper bounds for variables (scalar or vector of length `ns*nv`, default: `Inf`)

# Example
```julia
ns, nv, nθ = 100, 5, 3
θ_sets = [rand(nθ) for _ in 1:ns]

model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
    obj_data = [(i, j, (i-1)*nv + j, (i-1)*nθ) for i in 1:ns for j in 1:nv]
    objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

    con_data = [(i, j, (i-1)*nv + j, (i-1)*nθ) for i in 1:ns for j in 1:nv]
    constraint(c, v[v_idx] - θ[θ_off + 3] for (i, j, v_idx, θ_off) in con_data)
end
```
"""
function BatchExaModel(
        build::Function,
        nv::Int,
        ns::Int,
        θ_sets::Vector{<:AbstractVector};
        backend = nothing,
        v_start = 0.0,
        v_lvar = -Inf,
        v_uvar = Inf
    )
    length(θ_sets) == ns || throw(ArgumentError("θ_sets must have length ns=$ns"))
    nθ = length(θ_sets[1])
    all(length(θ) == nθ for θ in θ_sets) || throw(ArgumentError("All θ_sets must have same length"))

    c = ExaCore(; backend = backend)

    # All vars as one block, all params as one vector
    v = variable(c, ns * nv; start = v_start, lvar = v_lvar, uvar = v_uvar)
    θ_flat = reduce(vcat, θ_sets)
    θ = parameter(c, θ_flat)

    nc_before = c.ncon
    build(c, v, θ, ns, nv, nθ)

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

    return BatchExaModel{T, VT, typeof(model)}(
        model, ns, nv, nc, nθ, nnzj_per_scenario, nnzh_per_scenario
    )
end

# ============================================================================
# Accessors
# ============================================================================

num_scenarios(m::BatchExaModel) = m.ns
num_vars_per_scenario(m::BatchExaModel) = m.nv
num_constraints_per_scenario(m::BatchExaModel) = m.nc
total_vars(m::BatchExaModel) = m.ns * m.nv
total_cons(m::BatchExaModel) = m.ns * m.nc

# ============================================================================
# Full Evaluation (Single Kernel Launch)
# ============================================================================

"""
    obj(model::BatchExaModel, x_global)

Evaluate total objective (sum over all scenarios).
"""
function obj(model::BatchExaModel, x_global::AbstractVector)
    return obj(model.model, x_global)
end

"""
    cons!(model::BatchExaModel, x_global, c_global)

Evaluate all constraints.
Output: c_global ∈ ℝ^{ns*nc}
"""
function cons!(
        model::BatchExaModel,
        x_global::AbstractVector,
        c_global::AbstractVector
    )
    cons!(model.model, x_global, c_global)
    return c_global
end

"""
    grad!(model::BatchExaModel, x_global, g_global)

Evaluate total gradient.
Output: g_global ∈ ℝ^{ns*nv}
"""
function grad!(
        model::BatchExaModel,
        x_global::AbstractVector,
        g_global::AbstractVector
    )
    grad!(model.model, x_global, g_global)
    return g_global
end

"""
    jac_coord!(model::BatchExaModel, x_global, jac_global)

Evaluate all Jacobians (COO format).
Output: jac_global ∈ ℝ^{ns*nnzj}
"""
function jac_coord!(
        model::BatchExaModel,
        x_global::AbstractVector,
        jac_global::AbstractVector
    )
    jac_coord!(model.model, x_global, jac_global)
    return jac_global
end

"""
    jac_structure!(model::BatchExaModel, jrows, jcols)

Get the sparsity pattern of the Jacobian.
"""
function jac_structure!(
        model::BatchExaModel,
        jrows::AbstractVector{<:Integer},
        jcols::AbstractVector{<:Integer}
    )
    jac_structure!(model.model, jrows, jcols)
    return jrows, jcols
end

"""
    hess_coord!(model::BatchExaModel, x_global, y_global, hess_global; obj_weight=1.0)

Evaluate all Hessians of the Lagrangian (COO format).
"""
function hess_coord!(
        model::BatchExaModel,
        x_global::AbstractVector,
        y_global::AbstractVector,
        hess_global::AbstractVector;
        obj_weight = one(eltype(x_global))
    )
    hess_coord!(model.model, x_global, y_global, hess_global; obj_weight = obj_weight)
    return hess_global
end

"""
    hess_structure!(model::BatchExaModel, hrows, hcols)

Get the sparsity pattern of the Hessian of the Lagrangian.
"""
function hess_structure!(
        model::BatchExaModel,
        hrows::AbstractVector{<:Integer},
        hcols::AbstractVector{<:Integer}
    )
    hess_structure!(model.model, hrows, hcols)
    return hrows, hcols
end

# ============================================================================
# NLPModels Interface
# ============================================================================

NLPModels.get_nnzj(model::BatchExaModel) = NLPModels.get_nnzj(model.model)
NLPModels.get_nnzh(model::BatchExaModel) = NLPModels.get_nnzh(model.model)
NLPModels.get_nvar(model::BatchExaModel) = NLPModels.get_nvar(model.model)
NLPModels.get_ncon(model::BatchExaModel) = NLPModels.get_ncon(model.model)

# ============================================================================
# Index Ranges (GPU-friendly, zero allocation)
# ============================================================================

"""
    var_indices(model::BatchExaModel, i) -> UnitRange

Get the index range for variables of scenario i in the global variable vector.
Use as: `x_global[var_indices(model, i)]`
"""
function var_indices(model::BatchExaModel, i::Int)
    nv = model.nv
    return ((i - 1) * nv + 1):(i * nv)
end

"""
    cons_block_indices(model::BatchExaModel, i) -> UnitRange

Get the index range for constraints of scenario i in the global constraint vector.
Use as: `c_global[cons_block_indices(model, i)]`
"""
function cons_block_indices(model::BatchExaModel, i::Int)
    nc = model.nc
    return ((i - 1) * nc + 1):(i * nc)
end

"""
    grad_indices(model::BatchExaModel, i) -> UnitRange

Get the index range for the gradient of scenario i.
Same as `var_indices` since gradient has same layout as variables.
"""
grad_indices(model::BatchExaModel, i::Int) = var_indices(model, i)

# ============================================================================
# In-place Extraction (zero allocation, GPU-friendly)
# ============================================================================

"""
    extract_vars!(dest, model::BatchExaModel, i, x_global)

Extract variables for scenario i into pre-allocated `dest`.
Returns `dest` for convenience.
"""
function extract_vars!(
        dest::AbstractVector,
        model::BatchExaModel,
        i::Int,
        x_global::AbstractVector
    )
    copyto!(dest, 1, x_global, first(var_indices(model, i)), model.nv)
    return dest
end

"""
    extract_cons_block!(dest, model::BatchExaModel, i, c_global)

Extract constraint block for scenario i into pre-allocated `dest`.
Returns `dest` for convenience.
"""
function extract_cons_block!(
        dest::AbstractVector,
        model::BatchExaModel,
        i::Int,
        c_global::AbstractVector
    )
    copyto!(dest, 1, c_global, first(cons_block_indices(model, i)), model.nc)
    return dest
end

# ============================================================================
# Index Mapping
# ============================================================================

"""
    global_var_index(model::BatchExaModel, i, local_idx) -> global_idx

Convert local variable index to global index for scenario i.
"""
function global_var_index(model::BatchExaModel, i::Int, local_idx::Int)
    return (i - 1) * model.nv + local_idx
end

"""
    global_con_index(model::BatchExaModel, i, local_idx) -> global_idx

Convert local constraint index to global index for scenario i.
"""
function global_con_index(model::BatchExaModel, i::Int, local_idx::Int)
    return (i - 1) * model.nc + local_idx
end

# ============================================================================
# Parameter Updates
# ============================================================================

"""
    set_scenario_parameters!(model::BatchExaModel, i, θ_new)

Update parameters for scenario i.
"""
function set_scenario_parameters!(
        model::BatchExaModel,
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
    set_all_scenario_parameters!(model::BatchExaModel, θ_sets)

Update parameters for all scenarios.
"""
function set_all_scenario_parameters!(
        model::BatchExaModel,
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
    get_model(model::BatchExaModel)

Get the underlying ExaModel for direct NLPModels interface usage.
"""
get_model(model::BatchExaModel) = model.model
