# ============================================================================
# BatchExaModel — single-scenario builder + NLPModels matrix batch API
# ============================================================================

"""
    BatchExaModel{T, VT, MT, M} <: NLPModels.AbstractBatchNLPModel{T, MT}

Parametric optimization model where multiple fully independent scenarios are fused
into a single ExaModel and evaluated simultaneously using shared compiled expression
patterns.

All scenarios share identical sparsity structures. The builder defines expressions
for **one scenario**; `BatchExaModel` calls it `ns` times internally with offset
variable/parameter handles.

# Dimensions

- `ns`: number of scenarios (batch size)
- `nv`: number of variables per scenario
- `nc`: number of constraints per scenario
- `nθ`: number of parameters per scenario
"""
struct BatchExaModel{T, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}, M} <:
    NLPModels.AbstractBatchNLPModel{T, MT}
    meta::NLPModels.BatchNLPModelMeta{T, MT}
    model::M
    objbuffer::VT
    hess_perm::Vector{Int}
    hess_buffer::VT
    ns::Int
    nv::Int
    nc::Int
    nθ::Int
    nobj_per::Int
    nnzj_per::Int
    nnzh_per::Int
    nnzh_obj_per::Int
    nnzh_con_per::Int
end

function Base.show(io::IO, m::BatchExaModel{T, VT}) where {T, VT}
    println(io, "BatchExaModel{$T, $VT}")
    println(io, "  Scenarios: $(m.ns)")
    println(io, "  Variables per scenario: $(m.nv)")
    println(io, "  Constraints per scenario: $(m.nc)")
    println(io, "  Parameters per scenario: $(m.nθ)")
    println(io, "  Total variables: $(m.ns * m.nv)")
    println(io, "  Total constraints: $(m.ns * m.nc)")
    println(io, "  Jacobian nnz per scenario: $(m.nnzj_per)")
    return println(io, "  Hessian nnz per scenario: $(m.nnzh_per)")
end

# ============================================================================
# Helpers
# ============================================================================

_count_hess_nnz(::ObjectiveNull) = 0
_count_hess_nnz(::ConstraintNull) = 0
_count_hess_nnz(node) = _count_hess_nnz(node.inner) + node.f.o2step * length(node.itr)

function _build_hess_perm(ns, nnzh_obj_per, nnzh_con_per)
    nnzh_per = nnzh_obj_per + nnzh_con_per
    perm = Vector{Int}(undef, ns * nnzh_per)
    for s in 1:ns
        base = (s - 1) * nnzh_per
        for k in 1:nnzh_obj_per
            perm[base + k] = (s - 1) * nnzh_obj_per + k
        end
        for k in 1:nnzh_con_per
            perm[base + nnzh_obj_per + k] = ns * nnzh_obj_per + (s - 1) * nnzh_con_per + k
        end
    end
    return perm
end

function _to_matrix(v::AbstractVector, nrows::Int, ncols::Int)
    mat = similar(v, nrows, ncols)
    copyto!(vec(mat), v)
    return mat
end

# ============================================================================
# Constructor
# ============================================================================

"""
    BatchExaModel(build, c::ExaCore, ns::Int, θ_data::AbstractMatrix)

Build a batch model from a single-scenario builder function.

The user creates an `ExaCore` and passes parameter data as a matrix of size `(nθ, ns)`.
The builder defines a **single scenario** — creating variables, objectives, and
constraints using standard ExaModels calls. `BatchExaModel` invokes it `ns` times,
each time with a per-scenario parameter handle `θ`. Variable creation via
`variable(c, ...)` works normally and automatically gets the correct offsets.

# Arguments
- `build::Function`: Function `(c, θ) -> nothing`
  - `c`: ExaCore — use `variable(c, ...)`, `objective(c, ...)`, `constraint(c, ...)` as usual
  - `θ`: Parameter handle for this scenario's parameters (indices 1:nθ)
- `c::ExaCore`: ExaCore instance (parameters will be registered internally)
- `ns::Int`: Number of scenarios
- `θ_data::AbstractMatrix`: Parameter matrix of size `(nθ, ns)`

# Example
```julia
ns, nθ = 100, 3
θ_data = rand(nθ, ns)

c = ExaCore()
model = BatchExaModel(c, ns, θ_data) do c, θ
    v = variable(c, 5; start = 1.0, lvar = 0.0, uvar = 10.0)
    objective(c, θ[j] * v[j]^2 for j in 1:5)
    constraint(c, v[j] - θ[j] for j in 1:3)
end
```
"""
function BatchExaModel(
        build::Function,
        c::ExaCore,
        ns::Int,
        θ_data::AbstractMatrix,
    )
    Base.size(θ_data, 2) == ns || throw(
        ArgumentError("θ_data must have ns=$ns columns, got $(Base.size(θ_data, 2))"),
    )
    nθ = Base.size(θ_data, 1)

    # Register parameters as a flat vector (column-major: scenario 1, scenario 2, ...)
    parameter(c, vec(θ_data))

    # Call builder once per scenario with per-scenario θ handle.
    # The builder calls variable(c, ...) itself — offsets are automatic.
    for s in 1:ns
        θ_s = Parameter(nθ, nθ, (s - 1) * nθ)
        build(c, θ_s)
    end

    # Infer per-scenario dimensions
    nv = c.nvar ÷ ns
    nv * ns != c.nvar && throw(
        DimensionMismatch(
            "Total variables ($(c.nvar)) not evenly divisible by ns ($ns)",
        ),
    )

    nc_total = c.ncon
    nc = nc_total ÷ ns
    nc * ns != nc_total && throw(
        DimensionMismatch(
            "Total constraints ($nc_total) not evenly divisible by ns ($ns)",
        ),
    )

    nobj_total = c.nobj
    nobj_per = nobj_total ÷ ns
    nobj_per * ns != nobj_total && throw(
        DimensionMismatch(
            "Total objective entries ($nobj_total) not evenly divisible by ns ($ns)",
        ),
    )

    objbuffer = similar(c.x0, nobj_total)

    model = ExaModel(c)

    # Per-scenario sparsity counts
    total_nnzj = NLPModels.get_nnzj(model)
    total_nnzh = NLPModels.get_nnzh(model)
    nnzj_per = total_nnzj ÷ ns
    nnzh_per = total_nnzh ÷ ns

    # Obj/con hessian split
    nnzh_obj_total = _count_hess_nnz(model.objs)
    nnzh_con_total = _count_hess_nnz(model.cons)
    nnzh_obj_per = nnzh_obj_total ÷ ns
    nnzh_con_per = nnzh_con_total ÷ ns

    # Hessian permutation
    hess_perm = _build_hess_perm(ns, nnzh_obj_per, nnzh_con_per)

    # Hessian buffer
    hess_buffer = similar(c.x0, total_nnzh)

    # Build BatchNLPModelMeta with matrices
    T = eltype(c.x0)
    x0_mat = _to_matrix(model.meta.x0, nv, ns)
    lvar_mat = _to_matrix(model.meta.lvar, nv, ns)
    uvar_mat = _to_matrix(model.meta.uvar, nv, ns)
    y0_mat = _to_matrix(model.meta.y0, nc, ns)
    lcon_mat = _to_matrix(model.meta.lcon, nc, ns)
    ucon_mat = _to_matrix(model.meta.ucon, nc, ns)

    MT = typeof(x0_mat)
    meta = NLPModels.BatchNLPModelMeta{T, MT}(
        ns,
        nv;
        x0 = x0_mat,
        lvar = lvar_mat,
        uvar = uvar_mat,
        ncon = nc,
        y0 = y0_mat,
        lcon = lcon_mat,
        ucon = ucon_mat,
        nnzj = nnzj_per,
        nnzh = nnzh_per,
        minimize = model.meta.minimize,
    )

    VT = typeof(c.x0)
    return BatchExaModel{T, VT, MT, typeof(model)}(
        meta,
        model,
        objbuffer,
        hess_perm,
        hess_buffer,
        ns,
        nv,
        nc,
        nθ,
        nobj_per,
        nnzj_per,
        nnzh_per,
        nnzh_obj_per,
        nnzh_con_per,
    )
end

# ============================================================================
# Accessors
# ============================================================================

num_scenarios(m::BatchExaModel) = m.ns

"""
    get_model(model::BatchExaModel)

Get the underlying fused ExaModel for direct NLPModels interface usage (e.g. Ipopt).
"""
get_model(model::BatchExaModel) = model.model

"""
    var_indices(model::BatchExaModel, i) -> UnitRange

Index range for variables of scenario `i` in the global (fused) variable vector.
"""
function var_indices(model::BatchExaModel, i::Int)
    nv = model.nv
    return ((i - 1) * nv + 1):(i * nv)
end

"""
    cons_block_indices(model::BatchExaModel, i) -> UnitRange

Index range for constraints of scenario `i` in the global (fused) constraint vector.
"""
function cons_block_indices(model::BatchExaModel, i::Int)
    nc = model.nc
    return ((i - 1) * nc + 1):(i * nc)
end

# ============================================================================
# Parameter Updates
# ============================================================================

function set_scenario_parameters!(model::BatchExaModel, i::Int, θ_new::AbstractVector)
    nθ = model.nθ
    length(θ_new) != nθ && throw(
        DimensionMismatch("Parameter size mismatch: expected $nθ, got $(length(θ_new))"),
    )
    θ_start = (i - 1) * nθ + 1
    θ_end = i * nθ
    copyto!(view(model.model.θ, θ_start:θ_end), θ_new)
    return nothing
end

function set_all_scenario_parameters!(
        model::BatchExaModel,
        θ_sets::Vector{<:AbstractVector},
    )
    length(θ_sets) == model.ns ||
        throw(ArgumentError("θ_sets must have length $(model.ns)"))
    for i in 1:(model.ns)
        set_scenario_parameters!(model, i, θ_sets[i])
    end
    return nothing
end

# ============================================================================
# Objective buffer evaluation (CPU)
# ============================================================================

function _eval_objbuffer!(objbuffer, objs, x, θ)
    _eval_objbuffer!(objbuffer, objs.inner, x, θ)
    for i in eachindex(objs.itr)
        objbuffer[offset0(objs, i)] = objs.f(objs.itr[i], x, θ)
    end
    return
end
_eval_objbuffer!(objbuffer, ::ObjectiveNull, x, θ) = nothing

function _eval_objbuffer!(objbuffer, m::ExaModel, x)
    return _eval_objbuffer!(objbuffer, m.objs, x, m.θ)
end

# ============================================================================
# Batch API: obj!
# ============================================================================

function obj!(m::BatchExaModel, bx::AbstractMatrix, bf::AbstractVector)
    _eval_objbuffer!(m.objbuffer, m.model, vec(bx))
    obj_mat = reshape(m.objbuffer, m.nobj_per, m.ns)
    bf .= vec(sum(obj_mat; dims = 1))
    return bf
end

# ============================================================================
# Batch API: grad!
# ============================================================================

function grad!(m::BatchExaModel, bx::AbstractMatrix, bg::AbstractMatrix)
    grad!(m.model, vec(bx), vec(bg))
    return bg
end

# ============================================================================
# Batch API: cons!
# ============================================================================

function cons!(m::BatchExaModel, bx::AbstractMatrix, bc::AbstractMatrix)
    cons_nln!(m.model, vec(bx), vec(bc))
    return bc
end

# ============================================================================
# Batch API: jac_structure!
# ============================================================================

function jac_structure!(
        m::BatchExaModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
    )
    total_nnzj = NLPModels.get_nnzj(m.model)
    full_rows = zeros(Int, total_nnzj)
    full_cols = zeros(Int, total_nnzj)
    jac_structure!(m.model, full_rows, full_cols)

    # Scenario 1 uses zero offset → its entries are already local (1:nc, 1:nv)
    for k in 1:(m.nnzj_per)
        rows[k] = full_rows[k]
        cols[k] = full_cols[k]
    end
    return rows, cols
end

# ============================================================================
# Batch API: jac_coord!
# ============================================================================

function jac_coord!(m::BatchExaModel, bx::AbstractMatrix, bjvals::AbstractMatrix)
    jac_coord!(m.model, vec(bx), vec(bjvals))
    return bjvals
end

# ============================================================================
# Batch API: hess_structure!
# ============================================================================

function hess_structure!(
        m::BatchExaModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
    )
    total_nnzh = NLPModels.get_nnzh(m.model)
    full_rows = zeros(Int, total_nnzh)
    full_cols = zeros(Int, total_nnzh)
    hess_structure!(m.model, full_rows, full_cols)

    # Extract scenario 1's entries using hess_perm (interleaves obj+con)
    for k in 1:(m.nnzh_per)
        idx = m.hess_perm[k]
        rows[k] = full_rows[idx]
        cols[k] = full_cols[idx]
    end
    return rows, cols
end

# ============================================================================
# Batch API: hess_coord!
# ============================================================================

function hess_coord!(
        m::BatchExaModel,
        bx::AbstractMatrix,
        by::AbstractMatrix,
        bobj_weight::AbstractVector,
        bhvals::AbstractMatrix,
    )
    x_flat = vec(bx)
    y_flat = vec(by)
    nh = m.nnzh_per
    perm = m.hess_perm

    # Move perm to device for GPU-compatible gather indexing
    perm_dev = similar(m.hess_buffer, Int, length(perm))
    copyto!(perm_dev, perm)

    bobj_weight_cpu = Array(bobj_weight)
    if allequal(bobj_weight_cpu)
        # Common case: uniform obj_weight → single fused call + permute
        w = bobj_weight_cpu[1]
        hess_coord!(m.model, x_flat, y_flat, m.hess_buffer; obj_weight = w)
        bhvals_flat = vec(bhvals)
        bhvals_flat .= m.hess_buffer[perm_dev]
    else
        # Varying weights: 2-pass approach
        # Pass 1: objective hessian only (y=0, obj_weight=1)
        y_zero = similar(y_flat)
        fill!(y_zero, zero(eltype(y_flat)))
        hess_obj = similar(m.hess_buffer)
        hess_coord!(m.model, x_flat, y_zero, hess_obj; obj_weight = one(eltype(x_flat)))

        # Pass 2: constraint hessian only (obj_weight=0)
        hess_coord!(m.model, x_flat, y_flat, m.hess_buffer; obj_weight = zero(eltype(x_flat)))

        # Build per-element weight vector: element i belongs to scenario (i-1)÷nh+1
        w_cpu = [bobj_weight_cpu[(i - 1) ÷ nh + 1] for i in 1:length(perm)]
        w_dev = similar(bobj_weight, length(perm))
        copyto!(w_dev, w_cpu)

        # Combine per scenario (vectorized for GPU)
        bhvals_flat = vec(bhvals)
        bhvals_flat .= w_dev .* hess_obj[perm_dev] .+ m.hess_buffer[perm_dev]
    end
    return bhvals
end
