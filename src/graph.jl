abstract type AbstractIndex end
abstract type AbstractNode end
abstract type AbstractPar <: AbstractNode end
abstract type AbstractDual end
abstract type AbstractTriple  end

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
