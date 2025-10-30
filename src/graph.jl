# Abstract node type for the computation graph for symbolic expression
abstract type AbstractNode end

# Abstract node type for first-order forward pass tree
abstract type AbstractAdjointNode end

# Abstract node type for the computation graph for second-order forward pass
abstract type AbstractSecondAdjointNode end

"""
    Null
A null node

"""
struct Null{T} <: AbstractNode
    value::T
end
Null() = Null(nothing)

"""
    Null
A null node

"""
struct AdjointNull <: AbstractAdjointNode end

"""
    Null
A null node

"""
struct SecondAdjointNull <: AbstractSecondAdjointNode end


"""
    VarSource

A source of variable nodes

"""
struct VarSource <: AbstractNode end


"""
    Var{I}

A variable node for symbolic expression tree

# Fields:
- `i::I`: (parameterized) index 
"""
struct Var{I} <: AbstractNode
    i::I
end

struct ParameterSource <: AbstractNode end
struct ParameterNode{I} <: AbstractNode
    i::I
end

"""
    ParSource

A source of parameterized data

"""
struct ParSource <: AbstractNode end

"""
    ParIndexed{I, J}

A parameterized data node

# Fields:
- `inner::I`: parameter for the data
"""
struct ParIndexed{I,J} <: AbstractNode
    inner::I
end

@inline ParIndexed(inner::I, n) where {I} = ParIndexed{I,n}(inner)
"""
    Node1{F, I}

A node with one child for symbolic expression tree

# Fields:
- `inner::I`: children
"""
struct Node1{F,I} <: AbstractNode
    inner::I
end

"""
    Node2{F, I1, I2}

A node with two children for symbolic expression tree

# Fields:
- `inner1::I1`: children #1
- `inner2::I2`: children #2
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

@inline Base.getproperty(n::ParSource, s::Symbol) = ParIndexed(n, s)
@inline Base.getindex(n::ParSource, i) = ParIndexed(n, i)
@inline Base.indexed_iterate(n::P, idx, start = 1) where P <: Union{ParSource, ParIndexed} = (ParIndexed(n, idx), idx + 1)

@inline Base.getproperty(v::ParIndexed{I, n}, s::Symbol) where {I, n} = ParIndexed(v, s)
@inline Base.getindex(v::ParIndexed{I, n}, i) where {I, n} = ParIndexed(v, i)
@inline Base.indexed_iterate(v::ParIndexed{I, n}, idx, start = 1) where {I, n} = (ParIndexed(v, idx), idx + 1)


@inline Base.getindex(n::VarSource, i) = Var(i)
@inline Base.getindex(::ParameterSource, i) = ParameterNode(i)

@inline Node1(f::F, inner::I) where {F,I} = Node1{F,I}(inner)
@inline Node2(f::F, inner1::I1, inner2::I2) where {F,I1,I2} = Node2{F,I1,I2}(inner1, inner2)


struct Identity end

@inline (v::Var{I})(i, x, θ) where {I<:AbstractNode} = @inbounds x[v.i(i, x, θ)]
@inline (v::Var{I})(i, x, θ) where {I} = @inbounds x[v.i]
@inline (v::Var{I})(i::Identity, x, θ) where {I<:AbstractNode} = @inbounds x[v.i]

@inline (v::ParameterNode{I})(i, x, θ) where {I<:AbstractNode} = @inbounds θ[v.i(i, x, θ)]
@inline (v::ParameterNode{I})(::Any, x, θ) where {I} = @inbounds θ[v.i]
@inline (v::ParameterNode{I})(::Identity, x, θ) where {I<:AbstractNode} = @inbounds θ[v.i]

@inline (v::ParameterNode{I})(i, x, ::Nothing) where {I<:AbstractNode} = NaN
@inline (v::ParameterNode{I})(::Any, x, ::Nothing) where {I} = NaN
@inline (v::ParameterNode{I})(::Identity, x, ::Nothing) where {I<:AbstractNode} = NaN

@inline (v::ParSource)(i, x, θ) = i
@inline (v::ParIndexed{I,n})(i, x, θ) where {I,n} = @inbounds getfield(getfield(v, :inner)(i, x, θ), n)

(v::ParIndexed)(i::Identity, x, θ) = NaN # despecialized
(v::ParSource)(i::Identity, x, θ) = NaN # despecialized

"""
    AdjointNode1{F, T, I}

A node with one child for first-order forward pass tree

# Fields:
- `x::T`: function value
- `y::T`: first-order sensitivity
- `inner::I`: children
"""
struct AdjointNode1{F,T,I} <: AbstractAdjointNode
    x::T
    y::T
    inner::I
end
"""
    AdjointNode2{F, T, I1, I2}

A node with two children for first-order forward pass tree

# Fields:
- `x::T`: function value
- `y1::T`: first-order sensitivity w.r.t. first argument
- `y2::T`: first-order sensitivity w.r.t. second argument
- `inner1::I1`: children #1
- `inner2::I2`: children #2
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

A variable node for first-order forward pass tree

# Fields:
- `i::I`: index
- `x::T`: value
"""
struct AdjointNodeVar{I,T} <: AbstractAdjointNode
    i::I
    x::T
end

"""
    AdjointNodeSource{VT}

A source of `AdjointNode`. `adjoint_node_source[i]` returns an `AdjointNodeVar` at index `i`.

# Fields:
- `inner::VT`: variable vector
"""
struct AdjointNodeSource{VT}
    inner::VT
end

@inline AdjointNode1(f::F, x::T, y, inner::I) where {F,T,I} =
    AdjointNode1{F,T,I}(x, y, inner)
@inline AdjointNode2(f::F, x::T, y1, y2, inner1::I1, inner2::I2) where {F,T,I1,I2} =
    AdjointNode2{F,T,I1,I2}(x, y1, y2, inner1, inner2)

@inline Base.getindex(x::I, i) where {I<:AdjointNodeSource{Nothing}} =
    AdjointNodeVar(i, NaN)
@inline Base.getindex(x::I, i) where {I<:AdjointNodeSource} =
    @inbounds AdjointNodeVar(i, x.inner[i])


"""
    SecondAdjointNode1{F, T, I}

A node with one child for second-order forward pass tree

# Fields:
- `x::T`: function value
- `y::T`: first-order sensitivity
- `h::T`: second-order sensitivity
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

A node with one child for second-order forward pass tree

# Fields:
- `x::T`: function value
- `y1::T`: first-order sensitivity w.r.t. first argument
- `y2::T`: first-order sensitivity w.r.t. first argument
- `h11::T`: second-order sensitivity w.r.t. first argument
- `h12::T`: second-order sensitivity w.r.t. first and second argument
- `h22::T`: second-order sensitivity w.r.t. second argument
- `inner1::I1`: children #1
- `inner2::I2`: children #2
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

A variable node for first-order forward pass tree

# Fields:
- `i::I`: index
- `x::T`: value
"""
struct SecondAdjointNodeVar{I,T} <: AbstractSecondAdjointNode
    i::I
    x::T
end

"""
    SecondAdjointNodeSource{VT}

A source of `AdjointNode`. `adjoint_node_source[i]` returns an `AdjointNodeVar` at index `i`.

# Fields:
- `inner::VT`: variable vector
"""
struct SecondAdjointNodeSource{VT}
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

@inline Base.getindex(x::I, i) where {I<:SecondAdjointNodeSource{Nothing}} =
    SecondAdjointNodeVar(i, NaN)
@inline Base.getindex(x::I, i) where {I<:SecondAdjointNodeSource} =
    @inbounds SecondAdjointNodeVar(i, x.inner[i])


@inline (v::Null{Nothing})(i, x::V, θ) where {T,V<:AbstractVector{T}} = zero(T)
@inline (v::Null{N})(i, x::V, θ) where {N,T,V<:AbstractVector{T}} = T(v.value)
@inline (v::Null{N})(i, x::AdjointNodeSource{T}, θ) where {N,T} = AdjointNull()
@inline (v::Null{N})(i, x::SecondAdjointNodeSource{T}, θ) where {N,T} = SecondAdjointNull()
