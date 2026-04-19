"""
    BatchNLPModels

Template module for batched NLP models. Defines abstract types and
generic API functions following the NLPModels.jl pattern.

Key design: `AbstractBatchNLPModel <: NLPModels.AbstractNLPModel`, so batch
models participate in the standard NLPModels dispatch hierarchy.
"""
module BatchNLPModels

import NLPModels:
    NLPModels,
    AbstractNLPModel,
    AbstractNLPModelMeta,
    NLPModelMeta,
    obj!

# ============================================================================
# Abstract types
# ============================================================================

"""
    AbstractBatchNLPModel{T, S} <: AbstractNLPModel{T, S}

Abstract type for batched NLP models. Subtypes `AbstractNLPModel` so that
batch models participate in the standard NLPModels dispatch hierarchy.

Implementations must provide:
- `meta` field of type `NLPModelMeta`
- `counters` field of type `NLPModels.Counters`
- Batch API methods: `obj!`, `grad!`, `cons!`, `jac_structure!`, `jac_coord!`,
  `hess_structure!`, `hess_coord!`
"""
abstract type AbstractBatchNLPModel{T, S} <: AbstractNLPModel{T, S} end

# ============================================================================
# get_nbatch — derived from the VT type of the meta
# ============================================================================

get_nbatch(meta::NLPModelMeta{T, <:AbstractMatrix}) where {T} = Base.size(meta.x0, 2)
get_nbatch(meta::NLPModelMeta) = 1
get_nbatch(m::AbstractNLPModel) = get_nbatch(m.meta)


# ============================================================================
# Generic batch API — function stubs
# ============================================================================

"""
    obj(m::AbstractBatchNLPModel, bx::AbstractMatrix) -> Vector

Allocating version of `NLPModels.obj!`.
"""
function NLPModels.obj(m::AbstractBatchNLPModel{T}, bx::AbstractMatrix) where {T}
    bf = similar(bx, T, get_nbatch(m))
    obj!(m, bx, bf)
    return bf
end

"""
    grad!(m::AbstractBatchNLPModel, bx::AbstractMatrix, bg::AbstractMatrix)

Evaluate per-instance gradients. `bx` and `bg` are `(nvar, nbatch)`.
"""
function NLPModels.grad!(m::AbstractBatchNLPModel, bx::AbstractMatrix, bg::AbstractMatrix)
    error("grad! not implemented for $(typeof(m))")
end

"""
    grad(m::AbstractBatchNLPModel, bx::AbstractMatrix) -> Matrix

Allocating version of `grad!`.
"""
function NLPModels.grad(m::AbstractBatchNLPModel{T}, bx::AbstractMatrix) where {T}
    bg = similar(bx, T, NLPModels.get_nvar(m), get_nbatch(m))
    NLPModels.grad!(m, bx, bg)
    return bg
end

"""
    cons!(m::AbstractBatchNLPModel, bx::AbstractMatrix, bc::AbstractMatrix)

Evaluate per-instance constraints. `bx` is `(nvar, nbatch)`, `bc` is `(ncon, nbatch)`.
"""
function NLPModels.cons!(m::AbstractBatchNLPModel, bx::AbstractMatrix, bc::AbstractMatrix)
    error("cons! not implemented for $(typeof(m))")
end

"""
    cons(m::AbstractBatchNLPModel, bx::AbstractMatrix) -> Matrix

Allocating version of `cons!`.
"""
function NLPModels.cons(m::AbstractBatchNLPModel{T}, bx::AbstractMatrix) where {T}
    bc = similar(bx, T, NLPModels.get_ncon(m), get_nbatch(m))
    NLPModels.cons!(m, bx, bc)
    return bc
end

"""
    jac_structure!(m::AbstractBatchNLPModel, rows, cols)

Per-instance Jacobian sparsity pattern (local indices).
"""
function NLPModels.jac_structure!(
    m::AbstractBatchNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    error("jac_structure! not implemented for $(typeof(m))")
end

"""
    jac_coord!(m::AbstractBatchNLPModel, bx::AbstractMatrix, jvals::AbstractVector)

Evaluate batch Jacobian values.
"""
function NLPModels.jac_coord!(
    m::AbstractBatchNLPModel,
    bx::AbstractMatrix,
    jvals::AbstractVector,
)
    error("jac_coord! not implemented for $(typeof(m))")
end

"""
    hess_structure!(m::AbstractBatchNLPModel, rows, cols)

Per-instance Hessian sparsity pattern (local indices).
"""
function NLPModels.hess_structure!(
    m::AbstractBatchNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    error("hess_structure! not implemented for $(typeof(m))")
end

"""
    hess_coord!(m::AbstractBatchNLPModel, bx, by, hvals; obj_weight = 1)

Evaluate batch Hessian values.
"""
function NLPModels.hess_coord!(
    m::AbstractBatchNLPModel{T},
    bx::AbstractMatrix,
    by::AbstractMatrix,
    hvals::AbstractVector;
    obj_weight = one(T),
) where {T}
    error("hess_coord! not implemented for $(typeof(m))")
end

# ============================================================================
# FlatNLPModel
# ============================================================================

"""
    FlatNLPModel{T, VT, M} <: AbstractNLPModel{T, VT}

Wrapper that presents a batch NLP model as a flat (Vector-based) NLP model.
All NLPModels callbacks delegate to the underlying batch model's matrix API.

    FlatNLPModel(model::AbstractNLPModel)

Construct a flat model from a batch model whose `meta.x0` is a matrix.
"""
struct FlatNLPModel{T, VT <: AbstractVector{T}, M <: AbstractNLPModel{T}} <: AbstractNLPModel{T, VT}
    batch::M
    meta::NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

function FlatNLPModel(model::AbstractNLPModel{T}) where {T}
    nb = get_nbatch(model)
    nvar = NLPModels.get_nvar(model) * nb
    ncon = NLPModels.get_ncon(model) * nb
    nnzj = NLPModels.get_nnzj(model) * nb
    nnzh = NLPModels.get_nnzh(model) * nb
    x0 = vec(model.meta.x0)
    VT = typeof(x0)
    meta = NLPModelMeta{T, VT}(
        nvar,
        x0, vec(model.meta.lvar), vec(model.meta.uvar),
        Int[], Int[], Int[], Int[], collect(1:nvar), Int[],
        nvar, nvar, nvar,
        ncon,
        vec(model.meta.y0), vec(model.meta.lcon), vec(model.meta.ucon),
        Int[], Int[], Int[], Int[], Int[], Int[],
        nvar, nnzj, 0, nnzj, nnzh,
        0, ncon, Int[], collect(1:ncon),
        model.meta.minimize, false, String(model.meta.name),
        false, false, true, true, true, ncon > 0, true, ncon > 0, ncon > 0, true,
    )
    return FlatNLPModel(model, meta, NLPModels.Counters())
end

function NLPModels.obj(m::FlatNLPModel{T}, x::AbstractVector) where {T}
    nb = get_nbatch(m.batch)
    nvar = NLPModels.get_nvar(m.batch)
    bx = reshape(x, nvar, nb)
    bf = similar(x, T, nb)
    obj!(m.batch, bx, bf)
    return sum(bf)
end

function NLPModels.grad!(m::FlatNLPModel{T}, x::AbstractVector, g::AbstractVector) where {T}
    nb = get_nbatch(m.batch)
    nvar = NLPModels.get_nvar(m.batch)
    NLPModels.grad!(m.batch, reshape(x, nvar, nb), reshape(g, nvar, nb))
    return g
end

function NLPModels.cons_nln!(m::FlatNLPModel{T}, x::AbstractVector, c::AbstractVector) where {T}
    nb = get_nbatch(m.batch)
    nvar = NLPModels.get_nvar(m.batch)
    ncon = NLPModels.get_ncon(m.batch)
    NLPModels.cons!(m.batch, reshape(x, nvar, nb), reshape(c, ncon, nb))
    return c
end

function NLPModels.jac_nln_structure!(m::FlatNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    nb = get_nbatch(m.batch)
    nvar = NLPModels.get_nvar(m.batch)
    ncon = NLPModels.get_ncon(m.batch)
    nnzj = NLPModels.get_nnzj(m.batch)

    # Get per-instance structure
    r1 = @view rows[1:nnzj]
    c1 = @view cols[1:nnzj]
    NLPModels.jac_structure!(m.batch, r1, c1)

    # Replicate for each instance with shifted indices
    @inbounds for s in 2:nb
        offset = (s - 1) * nnzj
        row_shift = (s - 1) * ncon
        col_shift = (s - 1) * nvar
        for k in 1:nnzj
            rows[offset + k] = r1[k] + row_shift
            cols[offset + k] = c1[k] + col_shift
        end
    end
    return rows, cols
end

function NLPModels.jac_nln_coord!(m::FlatNLPModel{T}, x::AbstractVector, jvals::AbstractVector) where {T}
    nb = get_nbatch(m.batch)
    nvar = NLPModels.get_nvar(m.batch)
    nnzj = NLPModels.get_nnzj(m.batch)
    NLPModels.jac_coord!(m.batch, reshape(x, nvar, nb), reshape(jvals, nnzj, nb))
    return jvals
end

function NLPModels.hess_structure!(m::FlatNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    nb = get_nbatch(m.batch)
    nvar = NLPModels.get_nvar(m.batch)
    nnzh = NLPModels.get_nnzh(m.batch)

    # Get per-instance structure
    r1 = @view rows[1:nnzh]
    c1 = @view cols[1:nnzh]
    NLPModels.hess_structure!(m.batch, r1, c1)

    # Replicate for each instance with shifted indices
    @inbounds for s in 2:nb
        offset = (s - 1) * nnzh
        shift = (s - 1) * nvar
        for k in 1:nnzh
            rows[offset + k] = r1[k] + shift
            cols[offset + k] = c1[k] + shift
        end
    end
    return rows, cols
end

function NLPModels.hess_coord!(
    m::FlatNLPModel{T}, x::AbstractVector, y::AbstractVector,
    hvals::AbstractVector; obj_weight = one(T),
) where {T}
    nb = get_nbatch(m.batch)
    nvar = NLPModels.get_nvar(m.batch)
    ncon = NLPModels.get_ncon(m.batch)
    nnzh = NLPModels.get_nnzh(m.batch)
    NLPModels.hess_coord!(m.batch, reshape(x, nvar, nb), reshape(y, ncon, nb), reshape(hvals, nnzh, nb); obj_weight)
    return hvals
end

# ============================================================================
# Exports
# ============================================================================

export AbstractBatchNLPModel,
    FlatNLPModel

end # module BatchNLPModels
