"""
    BatchNLPModels

Template module for batched NLP models. Defines abstract types, metadata,
and generic API functions following the NLPModels.jl pattern.

Key design: `AbstractBatchNLPModel <: NLPModels.AbstractNLPModel`, so batch
models participate in the standard NLPModels dispatch hierarchy.
"""
module BatchNLPModels

import NLPModels:
    NLPModels,
    AbstractNLPModel,
    AbstractNLPModelMeta

# ============================================================================
# Abstract types
# ============================================================================

"""
    AbstractBatchNLPModel{T, S} <: AbstractNLPModel{T, S}

Abstract type for batched NLP models. Subtypes `AbstractNLPModel` so that
batch models participate in the standard NLPModels dispatch hierarchy.

Implementations must provide:
- `meta` field of type `<: AbstractBatchNLPModelMeta`
- `counters` field of type `NLPModels.Counters`
- Batch API methods: `obj!`, `grad!`, `cons!`, `jac_structure!`, `jac_coord!`,
  `hess_structure!`, `hess_coord!`
"""
abstract type AbstractBatchNLPModel{T, S} <: AbstractNLPModel{T, S} end

# ============================================================================
# BatchNLPModelMeta
# ============================================================================

"""
    BatchNLPModelMeta{T, VT, VI} <: AbstractNLPModelMeta{T, VT}

Metadata for a batched NLP where `nbatch` independent problems share
identical structure (dimensions, sparsity patterns). Extends the standard
`NLPModelMeta` interface with a batch dimension.

All `VT`-typed arrays are either vectors (nbatch=1) or matrices with
columns indexing instances.

# Type parameters
- `T`:  element type (e.g. `Float64`)
- `VT`: storage type — `Matrix{T}` when `nbatch > 1` (columns = instances),
        `Vector{T}` when `nbatch == 1`
- `VI`: integer index vector type (typically `Vector{Int}`)
"""
struct BatchNLPModelMeta{T, VT, VI} <: AbstractNLPModelMeta{T, VT}
    nvar::Int
    x0::VT
    lvar::VT
    uvar::VT
    ifix::VI
    ilow::VI
    iupp::VI
    irng::VI
    ifree::VI
    iinf::VI
    nlvb::Int
    nlvo::Int
    nlvc::Int
    ncon::Int
    y0::VT
    lcon::VT
    ucon::VT
    jfix::VI
    jlow::VI
    jupp::VI
    jrng::VI
    jfree::VI
    jinf::VI
    nnzo::Int
    nnzj::Int
    lin_nnzj::Int
    nln_nnzj::Int
    nnzh::Int
    nlin::Int
    nnln::Int
    lin::VI
    nln::VI
    minimize::Bool
    islp::Bool
    name::String
    variable_bounds_analysis::Bool
    constraint_bounds_analysis::Bool
    sparse_jacobian::Bool
    sparse_hessian::Bool
    grad_available::Bool
    jac_available::Bool
    hess_available::Bool
    jprod_available::Bool
    jtprod_available::Bool
    hprod_available::Bool
end

# ============================================================================
# Accessors — auto-generate get_* for all fields
# ============================================================================

for field in fieldnames(BatchNLPModelMeta)
    meth = Symbol("get_", field)
    # Extend NLPModels accessor if it exists, otherwise define locally
    if isdefined(NLPModels, meth)
        @eval begin
            NLPModels.$meth(meta::BatchNLPModelMeta) = getproperty(meta, $(QuoteNode(field)))
            NLPModels.$meth(m::AbstractBatchNLPModel) = NLPModels.$meth(m.meta)
        end
    else
        @eval begin
            $meth(meta::BatchNLPModelMeta) = getproperty(meta, $(QuoteNode(field)))
            $meth(m::AbstractBatchNLPModel) = $meth(m.meta)
            export $meth
        end
    end
end

# get_nbatch is derived from the VT type, not stored as a field
get_nbatch(meta::BatchNLPModelMeta{T, <:AbstractMatrix}) where {T} = Base.size(meta.x0, 2)
get_nbatch(meta::BatchNLPModelMeta) = 1
get_nbatch(m::AbstractBatchNLPModel) = get_nbatch(m.meta)
export get_nbatch

# ============================================================================
# Constructor helpers
# ============================================================================

"""
    _first_instance(v)

Extract first-instance data for bounds classification.
"""
_first_instance(v::AbstractVector) = v
_first_instance(m::AbstractMatrix) = @view m[:, 1]

function _classify_bounds(lb, ub, ::Type{T}) where {T}
    ifix  = findall(lb .== ub)
    ilow  = findall((lb .> T(-Inf)) .& (ub .== T(Inf)))
    iupp  = findall((lb .== T(-Inf)) .& (ub .< T(Inf)))
    irng  = findall((lb .> T(-Inf)) .& (ub .< T(Inf)) .& (lb .< ub))
    ifree = findall((lb .== T(-Inf)) .& (ub .== T(Inf)))
    iinf  = findall(lb .> ub)
    return ifix, ilow, iupp, irng, ifree, iinf
end

"""
    BatchNLPModelMeta(nbatch, nvar, x0, lvar, uvar, ncon, y0, lcon, ucon; kwargs...)

Construct batch NLP metadata. Bounds analysis (variable/constraint
classification) is performed on the first instance.
"""
function BatchNLPModelMeta(
    nvar::Int,
    x0::VT,
    lvar::VT,
    uvar::VT,
    ncon::Int,
    y0::VT,
    lcon::VT,
    ucon::VT;
    nnzj::Int = nvar * ncon,
    nnzh::Int = nvar * (nvar + 1) ÷ 2,
    minimize::Bool = true,
    islp::Bool = false,
    name::String = "Generic",
) where {VT}
    T = eltype(VT)

    # Variable bounds analysis (first instance)
    ifix, ilow, iupp, irng, ifree, iinf = _classify_bounds(
        _first_instance(lvar), _first_instance(uvar), T,
    )

    # Constraint bounds analysis (first instance)
    if ncon > 0
        jfix, jlow, jupp, jrng, jfree, jinf = _classify_bounds(
            _first_instance(lcon), _first_instance(ucon), T,
        )
    else
        jfix = jlow = jupp = jrng = jfree = jinf = Int[]
    end

    nln = collect(1:ncon)
    VI = Vector{Int}

    return BatchNLPModelMeta{T, VT, VI}(
        nvar,
        x0, lvar, uvar,
        ifix, ilow, iupp, irng, ifree, iinf,
        nvar, nvar, nvar,    # nlvb, nlvo, nlvc
        ncon,
        y0, lcon, ucon,
        jfix, jlow, jupp, jrng, jfree, jinf,
        nvar,                # nnzo
        nnzj, 0, nnzj,      # nnzj, lin_nnzj, nln_nnzj
        nnzh,
        0, ncon,             # nlin, nnln
        Int[], nln,          # lin, nln
        minimize, islp, name,
        true, true,          # variable/constraint_bounds_analysis
        true, true,          # sparse_jacobian/hessian
        true,                # grad_available
        ncon > 0,            # jac_available
        true,                # hess_available
        ncon > 0,            # jprod_available
        ncon > 0,            # jtprod_available
        true,                # hprod_available
    )
end

# ============================================================================
# Generic batch API — function stubs
# ============================================================================
#
# Concrete implementations must define the `!` (in-place) methods.
# Allocating wrappers are provided as defaults.

"""
    obj!(m::AbstractBatchNLPModel, bx::AbstractMatrix, bf::AbstractVector)

Evaluate per-instance objectives. `bx` is `(nvar, nbatch)`, `bf` is `(nbatch,)`.
"""
function obj! end

"""
    obj(m::AbstractBatchNLPModel, bx::AbstractMatrix) -> Vector

Allocating version of `obj!`.
"""
function NLPModels.obj(m::AbstractBatchNLPModel{T}, bx::AbstractMatrix) where {T}
    bf = Vector{T}(undef, get_nbatch(m))
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
    bg = Matrix{T}(undef, get_nvar(m), get_nbatch(m))
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
    bc = Matrix{T}(undef, get_ncon(m), get_nbatch(m))
    NLPModels.cons!(m, bx, bc)
    return bc
end

"""
    jac_structure!(m::AbstractBatchNLPModel, rows, cols)

Per-instance Jacobian sparsity pattern (local indices). `rows` and `cols`
have length `nnzj` (per instance).
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

Evaluate batch Jacobian values. `jvals` is a flat vector of length
`nnzj * nbatch`, laid out instance by instance.
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

Per-instance Hessian sparsity pattern (local indices). `rows` and `cols`
have length `nnzh` (per instance).
"""
function NLPModels.hess_structure!(
    m::AbstractBatchNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    error("hess_structure! not implemented for $(typeof(m))")
end

"""
    hess_coord!(m::AbstractBatchNLPModel, bx, by, bobj_weight, hvals)

Evaluate batch Hessian values. `hvals` is a flat vector of length
`nnzh * nbatch`, laid out instance by instance.

- `bx`: `(nvar, nbatch)` primal values
- `by`: `(ncon, nbatch)` dual values
- `bobj_weight`: `(nbatch,)` per-instance objective weights
- `hvals`: `(nnzh * nbatch,)` flat Hessian values
"""
function NLPModels.hess_coord!(
    m::AbstractBatchNLPModel,
    bx::AbstractMatrix,
    by::AbstractMatrix,
    bobj_weight::AbstractVector,
    hvals::AbstractVector,
)
    error("hess_coord! not implemented for $(typeof(m))")
end

# ============================================================================
# Exports
# ============================================================================

export AbstractBatchNLPModel,
    BatchNLPModelMeta,
    obj!

end # module BatchNLPModels
