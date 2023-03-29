struct Dual1{F, T, I} <: AbstractDual
    x::T
    y::T
    inner::I
end
struct Dual2{F, T, I1,I2} <: AbstractDual
    x::T
    y1::T
    y2::T
    inner1::I1
    inner2::I2
end
struct DualVar{I,T} <: AbstractDual
    i::I
    x::T
end
struct DualSource{T, VT <: AbstractVector{T}}
    inner::VT
end
struct DualNullSource end

@inline Dual1(f::F,x::T,y,inner::I) where {F,T,I} = Dual1{F,T,I}(x,y,inner)
@inline Dual2(f::F,x::T,y1,y2,inner1::I1,inner2::I2) where {F,T,I1,I2} = Dual2{F,T,I1,I2}(x,y1,y2,inner1,inner2)


DualSource(::Nothing) = DualNullSource()

@inbounds @inline Base.getindex(x::I,i) where I <: DualNullSource = DualVar(i,NaN16)
@inbounds @inline Base.getindex(x::I,i) where I <: DualSource = DualVar(i, x.inner[i])
