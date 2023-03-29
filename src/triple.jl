struct Triple1{F, T, I} <: AbstractTriple
    x::T
    y::T
    h::T
    inner::I
end
struct Triple2{F, T, I1,I2} <: AbstractTriple
    x::T
    y1::T
    y2::T
    h11::T
    h12::T
    h22::T
    inner1::I1
    inner2::I2
end

struct TripleVar{I,T} <: AbstractTriple
    i::I
    x::T
end
struct TripleSource{T, VT <: AbstractVector{T}}
    inner::VT
end

@inline Triple1(f::F,x::T,y,h,inner::I) where {F,T,I} = Triple1{F,T,I}(x,y,h,inner)
@inline Triple2(f::F,x::T,y1,y2,h11,h12,h22,inner1::I1,inner2::I2) where {F,T,I1,I2} = Triple2{F,T,I1,I2}(x,y1,y2,h11,h12,h22,inner1,inner2)

struct TripleNullSource end
TripleSource(::Nothing) = TripleNullSource()

@inbounds @inline Base.getindex(x::I,i) where I <: TripleNullSource = TripleVar(i,NaN)
@inbounds @inline Base.getindex(x::I,i) where I <: TripleSource = TripleVar(i, x.inner[i])
