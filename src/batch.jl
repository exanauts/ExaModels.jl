"""
    BatchExaModel{T, VT, M}

Parametric optimization model where multiple scenarios are fused into a single
ExaModel and evaluated simultaneously using a shared compiled expression pattern.

All scenarios must share identical sparsity structures for Jacobians and Hessians
independently of the parameter values. The model builder receives all variables
and parameters and should rely on generators iterating over scenario data.

# Dimensions

- `ns`: number of scenarios
- `nv`: number of variables per scenario
- `nc`: number of constraints per scenario
- `np`: number of parameters per scenario

# Layout

- Variables:    [v₁; v₂; …; vₙₛ]
- Constraints: [c₁; c₂; …; cₙₛ]

# Fields

- `model::M` : fused ExaModel containing all scenarios
- `ns::Int`  : number of scenarios
- `np::Int`  : number of parameters per scenario
"""
struct BatchExaModel{T, VT <: AbstractVector{T}, M <: ExaModel{T, VT}} <: NLPModels.AbstractNLPModel{T,VT}
    model::M
    ns::Int
    np::Int
end

function Base.show(io::IO, m::BatchExaModel{T, VT}) where {T, VT}
    println(io, "BatchExaModel{$T, $VT}")
    println(io, "  Number of scenarios: $(m.ns)")
    println(io, "  Number of parameter per scenario: $(m.np)")
    Base.show(m.model)
    return
end

# ============================================================================
# Constructor
# ============================================================================

"""
    BatchExaModel(build, nd, nv, ns, θ_sets; backend=nothing)

Build a batch model where all scenarios are fused into a single ExaModel.

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

model = BatchExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
    obj_data = [(i, j, (i-1)*nv + j, (i-1)*nθ) for i in 1:ns for j in 1:nv]
    objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

    con_data = [(i, j, (i-1)*nv + j, (i-1)*nθ) for i in 1:ns for j in 1:nv]
    constraint(c, v[v_idx] + d[1] - θ[θ_off + 3] for (i, j, v_idx, θ_off) in con_data)
end
```
"""
function BatchExaModel(
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

    model = ExaModel(c)

    # Calculate nnz per scenario
    total_nnzj = NLPModels.get_nnzj(model)
    total_nnzh = NLPModels.get_nnzh(model)
    nnzj = total_nnzj ÷ ns
    nnzh= total_nnzh ÷ ns

    T = eltype(c.x0)
    VT = typeof(c.x0)

    return BatchExaModel{T, VT, typeof(model)}(
        model, ns, nv, nd, nc, nθ, nnzj_per_scenario, nnzh_per_scenario
    )
end

# ============================================================================
# Full Evaluation (Single Kernel Launch)
# ============================================================================

"""
    obj(model::BatchExaModel, x_global)

Evaluate all objectives.
Output: obj_global ∈ ℝ^{ns}
"""
function obj(model::BatchExaModel, x_global::AbstractVector)
    return obj(model.model, x_global, obj_global)
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

Evaluate all gradients.
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

Get the common sparsity pattern of the Jacobian.
Output: jrows ∈ ℝ^{ns*nnzj} and jcols ∈ ℝ^{ns*nnzj}
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
Output: hess_global ∈ ℝ^{ns*nnzh}
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

Get the common sparsity pattern of the Hessian of the Lagrangian.
Output: hrows ∈ ℝ^{ns*nnzh} and hcols ∈ ℝ^{ns*nnzh}
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

"""
    get_nnzj(model::BatchExaModel)

Total number of Jacobian nonzeros.
"""
NLPModels.get_nnzj(model::BatchExaModel) = NLPModels.get_nnzj(model.model)

"""
    get_nnzh(model::BatchExaModel)

Total number of Hessian nonzeros.
"""
NLPModels.get_nnzh(model::BatchExaModel) = NLPModels.get_nnzh(model.model)

"""
    get_nvar(model::BatchExaModel)

Total number of variables.
"""
NLPModels.get_nvar(model::BatchExaModel) = NLPModels.get_nvar(model.model)

"""
    get_ncon(model::BatchExaModel)

Total number of constraints.
"""
NLPModels.get_ncon(model::BatchExaModel) = NLPModels.get_ncon(model.model)
