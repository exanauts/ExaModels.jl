# Abstract node type for symbolic expression
abstract type AbstractNode end

# Abstract node type for first-order forward pass
abstract type AbstractAdjointNode end

# Abstract node type for second-order forward pass
abstract type AbstractSecondAdjointNode end

"""
    Var{I}

A variable node for symbolic expression tree

# Fields:
- `i::I`: DESCRIPTION
"""
struct Var{I} <: AbstractNode
    i::I
end
"""
    Par

DOCSTRING

"""
struct Par <: AbstractNode end
"""
    ParIndexed{I, J}

DOCSTRING

# Fields:
- `inner::I`: DESCRIPTION
"""
struct ParIndexed{I,J} <: AbstractNode
    inner::I
end

@inline ParIndexed(inner::I, n) where {I} = ParIndexed{I,n}(inner)
"""
    Node1{F, I}

DOCSTRING

# Fields:
- `inner::I`: DESCRIPTION
"""
struct Node1{F,I} <: AbstractNode
    inner::I
end
"""
    Node2{F, I1, I2}

DOCSTRING

# Fields:
- `inner1::I1`: DESCRIPTION
- `inner2::I2`: DESCRIPTION
"""
struct Node2{F,I1,I2} <: AbstractNode
    inner1::I1
    inner2::I2
end

struct FirstFixed{F}
    inner::F
end
struct SecondFixed{F}
    inner::F
end

@inline Base.getindex(n::Par, i) = ParIndexed(n, i)

Par(iter::DataType) = Par()
Par(iter, idx...) = ParIndexed(Par(iter, idx[2:end]...), idx[1])
Par(iter::Type{T}, idx...) where {T<:Tuple} =
    Tuple(Par(p, i, idx...) for (i, p) in enumerate(T.parameters))

Par(iter::Type{T}, idx...) where {T<:NamedTuple} = NamedTuple{T.parameters[1]}(
    Par(p, i, idx...) for (i, p) in enumerate(T.parameters[2].parameters)
)

@inline Node1(f::F, inner::I) where {F,I} = Node1{F,I}(inner)
@inline Node2(f::F, inner1::I1, inner2::I2) where {F,I1,I2} = Node2{F,I1,I2}(inner1, inner2)


struct Identity end

struct NaNSource{T} <: AbstractVector{T} end
@inline Base.getindex(::NaNSource{T}, i) where {T} = T(NaN)


@inline (v::Var{I})(i, x) where {I} = @inbounds x[v.i(i, x)]
@inline (v::Par)(i, x) = i
@inline (v::ParIndexed{I,n})(i, x) where {I,n} = @inbounds v.inner(i, x)[n]

(v::ParIndexed)(i::Identity, x) = NaN16 # despecialized
(v::Par)(i::Identity, x) = NaN16 # despecialized
(v::Var)(i::Identity, x) = @inbounds x[v.i] # despecialized

"""
    AdjointNode1{F, T, I}

DOCSTRING

# Fields:
- `x::T`: DESCRIPTION
- `y::T`: DESCRIPTION
- `inner::I`: DESCRIPTION
"""
struct AdjointNode1{F,T,I} <: AbstractAdjointNode
    x::T
    y::T
    inner::I
end
"""
    AdjointNode2{F, T, I1, I2}

DOCSTRING

# Fields:
- `x::T`: DESCRIPTION
- `y1::T`: DESCRIPTION
- `y2::T`: DESCRIPTION
- `inner1::I1`: DESCRIPTION
- `inner2::I2`: DESCRIPTION
"""
struct AdjointNode2{F,T,I1,I2} <: AbstractAdjointNode
    x::T
    y1::T
    y2::T
    inner1::I1
    inner2::I2
end
"""
    AdjointNodeVar{I, T}

DOCSTRING

# Fields:
- `i::I`: DESCRIPTION
- `x::T`: DESCRIPTION
"""
struct AdjointNodeVar{I,T} <: AbstractAdjointNode
    i::I
    x::T
end
"""
    AdjointNodeSource{T, VT <: AbstractVector{T}}

DOCSTRING

# Fields:
- `inner::VT`: DESCRIPTION
"""
struct AdjointNodeSource{T,VT<:AbstractVector{T}}
    inner::VT
end
"""
    AdjointNodeNullSource

DOCSTRING

"""
struct AdjointNodeNullSource end

@inline AdjointNode1(f::F, x::T, y, inner::I) where {F,T,I} =
    AdjointNode1{F,T,I}(x, y, inner)
@inline AdjointNode2(f::F, x::T, y1, y2, inner1::I1, inner2::I2) where {F,T,I1,I2} =
    AdjointNode2{F,T,I1,I2}(x, y1, y2, inner1, inner2)


AdjointNodeSource(::Nothing) = AdjointNodeNullSource()

@inline Base.getindex(x::I, i) where {I<:AdjointNodeNullSource} =
    AdjointNodeVar(i, NaN16)
@inline Base.getindex(x::I, i) where {I<:AdjointNodeSource} =
    @inbounds AdjointNodeVar(i, x.inner[i])


"""
    SecondAdjointNode1{F, T, I}

DOCSTRING

# Fields:
- `x::T`: DESCRIPTION
- `y::T`: DESCRIPTION
- `h::T`: DESCRIPTION
- `inner::I`: DESCRIPTION
"""
struct SecondAdjointNode1{F,T,I} <: AbstractSecondAdjointNode
    x::T
    y::T
    h::T
    inner::I
end
"""
    SecondAdjointNode2{F, T, I1, I2}

DOCSTRING

# Fields:
- `x::T`: DESCRIPTION
- `y1::T`: DESCRIPTION
- `y2::T`: DESCRIPTION
- `h11::T`: DESCRIPTION
- `h12::T`: DESCRIPTION
- `h22::T`: DESCRIPTION
- `inner1::I1`: DESCRIPTION
- `inner2::I2`: DESCRIPTION
"""
struct SecondAdjointNode2{F,T,I1,I2} <: AbstractSecondAdjointNode
    x::T
    y1::T
    y2::T
    h11::T
    h12::T
    h22::T
    inner1::I1
    inner2::I2
end

"""
    SecondAdjointNodeVar{I, T}

DOCSTRING

# Fields:
- `i::I`: DESCRIPTION
- `x::T`: DESCRIPTION
"""
struct SecondAdjointNodeVar{I,T} <: AbstractSecondAdjointNode
    i::I
    x::T
end
"""
    SecondAdjointNodeSource{T, VT <: AbstractVector{T}}

DOCSTRING

# Fields:
- `inner::VT`: DESCRIPTION
"""
struct SecondAdjointNodeSource{T,VT<:AbstractVector{T}}
    inner::VT
end

@inline SecondAdjointNode1(f::F, x::T, y, h, inner::I) where {F,T,I} =
    SecondAdjointNode1{F,T,I}(x, y, h, inner)
@inline SecondAdjointNode2(
    f::F,
    x::T,
    y1,
    y2,
    h11,
    h12,
    h22,
    inner1::I1,
    inner2::I2,
) where {F,T,I1,I2} =
    SecondAdjointNode2{F,T,I1,I2}(x, y1, y2, h11, h12, h22, inner1, inner2)

struct SecondAdjointNodeNullSource end
SecondAdjointNodeSource(::Nothing) = SecondAdjointNodeNullSource()

@inline Base.getindex(x::I, i) where {I<:SecondAdjointNodeNullSource} =
    SecondAdjointNodeVar(i, NaN)
@inline Base.getindex(x::I, i) where {I<:SecondAdjointNodeSource} =
    @inbounds SecondAdjointNodeVar(i, x.inner[i])
