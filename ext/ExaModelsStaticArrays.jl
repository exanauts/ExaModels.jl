module ExaModelsStaticArrays

using ExaModels, StaticArrays
using LinearAlgebra: LinearAlgebra

# ============================================================================
# Scalar ExaNode interface for StaticArrays
#
# StaticArrays computes det, dot, norm, cross, etc. via unrolled scalar code
# generated at compile time.  The extras needed are:
#   • zero / one      — accumulators and identity elements
#   • adjoint / transpose / conj — ExaNodes are real-valued, so all are identity
#   • LinearAlgebra.dot(a, b) — scalar dot = a * b for real nodes
# ============================================================================

# zero/one return symbolic zero/one nodes so that StaticArrays initializers work.
@inline Base.zero(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(nothing)
@inline Base.zero(::N) where {N <: ExaModels.AbstractNode} = ExaModels.Null(nothing)
@inline Base.one(::Type{<:ExaModels.AbstractNode}) = ExaModels.Null(1)
@inline Base.one(::N) where {N <: ExaModels.AbstractNode} = ExaModels.Null(1)

# ExaNodes are real-valued → conj, adjoint, and transpose are all identity.
@inline Base.conj(x::ExaModels.AbstractNode) = x
@inline Base.adjoint(x::ExaModels.AbstractNode) = x
@inline Base.adjoint(x::ExaModels.AbstractAdjointNode) = x
@inline Base.adjoint(x::ExaModels.AbstractSecondAdjointNode) = x
@inline Base.transpose(x::ExaModels.AbstractNode) = x
@inline Base.transpose(x::ExaModels.AbstractAdjointNode) = x
@inline Base.transpose(x::ExaModels.AbstractSecondAdjointNode) = x

# Scalar dot product: for real nodes conj(a)*b == a*b.
# StaticArrays' _vecdot initialises via dot(zero(elem), zero(elem)) so this is required.
@inline LinearAlgebra.dot(a::ExaModels.AbstractNode, b::ExaModels.AbstractNode) = a * b
@inline LinearAlgebra.dot(a::ExaModels.AbstractNode, b::Real) = a * b
@inline LinearAlgebra.dot(a::Real, b::ExaModels.AbstractNode) = a * b

# StaticArrays.det calls arithmetic_closure(eltype(A)) to determine a "numeric closure" type
# and then converts the matrix to that type.  For ExaNode types, arithmetic is already
# closed (operations return other AbstractNode subtypes), so we declare the element type
# itself as the closure type to skip the no-op conversion.
StaticArrays.arithmetic_closure(::Type{T}) where {T <: ExaModels.AbstractNode} = T

end # module ExaModelsStaticArrays
