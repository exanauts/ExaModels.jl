abstract type AbstractIndex end
abstract type AbstractNode end
abstract type AbstractPar <: AbstractNode end
abstract type AbstractAdjointNode end
abstract type AbstractSecondAdjointNode  end

struct Var{I} <: AbstractNode
    i::I
end
struct Par <: AbstractPar end
struct ParIndexed{I,J} <: AbstractPar
    inner::I
end
@inline ParIndexed(inner::I,n) where I = ParIndexed{I,n}(inner)
struct Node1{F,I} <: AbstractNode
    inner::I
end
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

@inline Base.getindex(n::Par,i) = ParIndexed(n,i)

Par(iter::DataType) = Par()
Par(iter,idx...) = ParIndexed(Par(iter,idx[2:end]...),idx[1])
Par(iter::Type{T},idx...) where {T <: Tuple} =
    Tuple(
        Par(p,i,idx...)
        for (i,p) in enumerate(T.parameters))

Par(iter::Type{T},idx...) where {T <: NamedTuple} =
    NamedTuple{T.parameters[1]}(
        Par(p,i,idx...)
        for (i,p) in enumerate(T.parameters[2].parameters))

@inline Node1(f::F,inner::I) where {F,I} = Node1{F,I}(inner)
@inline Node2(f::F,inner1::I1,inner2::I2) where {F,I1,I2} = Node2{F,I1,I2}(inner1,inner2)


struct Identity end

struct NaNSource{T} <: AbstractVector{T} end
@inline Base.getindex(::NaNSource{T},i) where T = T(NaN)


@inbounds @inline (v::Var{I})(i,x) where I = x[v.i(i,x)]
@inbounds @inline (v::Par)(i,x) = i
@inbounds @inline (v::ParIndexed{I,n})(i,x) where {I,n} = v.inner(i,x)[n]

@inbounds (v::ParIndexed)(i::Identity,x) = NaN16 # despecialized
@inbounds (v::Par)(i::Identity,x) = NaN16 # despecialized
@inbounds (v::Var)(i::Identity,x) = x[v.i] # despecialized


struct AdjointNode1{F, T, I} <: AbstractAdjointNode
    x::T
    y::T
    inner::I
end
struct AdjointNode2{F, T, I1,I2} <: AbstractAdjointNode
    x::T
    y1::T
    y2::T
    inner1::I1
    inner2::I2
end
struct AdjointNodeVar{I,T} <: AbstractAdjointNode
    i::I
    x::T
end
struct AdjointNodeSource{T, VT <: AbstractVector{T}}
    inner::VT
end
struct AdjointNodeNullSource end

@inline AdjointNode1(f::F,x::T,y,inner::I) where {F,T,I} = AdjointNode1{F,T,I}(x,y,inner)
@inline AdjointNode2(f::F,x::T,y1,y2,inner1::I1,inner2::I2) where {F,T,I1,I2} = AdjointNode2{F,T,I1,I2}(x,y1,y2,inner1,inner2)


AdjointNodeSource(::Nothing) = AdjointNodeNullSource()

@inbounds @inline Base.getindex(x::I,i) where I <: AdjointNodeNullSource = AdjointNodeVar(i,NaN16)
@inbounds @inline Base.getindex(x::I,i) where I <: AdjointNodeSource = AdjointNodeVar(i, x.inner[i])


struct SecondAdjointNode1{F, T, I} <: AbstractSecondAdjointNode
    x::T
    y::T
    h::T
    inner::I
end
struct SecondAdjointNode2{F, T, I1,I2} <: AbstractSecondAdjointNode
    x::T
    y1::T
    y2::T
    h11::T
    h12::T
    h22::T
    inner1::I1
    inner2::I2
end

struct SecondAdjointNodeVar{I,T} <: AbstractSecondAdjointNode
    i::I
    x::T
end
struct SecondAdjointNodeSource{T, VT <: AbstractVector{T}}
    inner::VT
end

@inline SecondAdjointNode1(f::F,x::T,y,h,inner::I) where {F,T,I} = SecondAdjointNode1{F,T,I}(x,y,h,inner)
@inline SecondAdjointNode2(f::F,x::T,y1,y2,h11,h12,h22,inner1::I1,inner2::I2) where {F,T,I1,I2} = SecondAdjointNode2{F,T,I1,I2}(x,y1,y2,h11,h12,h22,inner1,inner2)

struct SecondAdjointNodeNullSource end
SecondAdjointNodeSource(::Nothing) = SecondAdjointNodeNullSource()

@inbounds @inline Base.getindex(x::I,i) where I <: SecondAdjointNodeNullSource = SecondAdjointNodeVar(i,NaN)
@inbounds @inline Base.getindex(x::I,i) where I <: SecondAdjointNodeSource = SecondAdjointNodeVar(i, x.inner[i])
