@inline (a::Pair{P,S} where {P,S<:AbstractNode})(i, x, θ) = a.second(i, x, θ)

"""
    Compressor{I}

Data structure for the sparse index

# Fields:
- `inner::I`: stores the sparse index as a tuple form
"""
struct Compressor{I}
    inner::I
end
@inline (i::Compressor{I})(n) where {I} = @inbounds i.inner[n]

struct SIMDFunction{F,C1,C2}
    f::F
    comp1::C1
    comp2::C2
    o0::Int
    o1::Int
    o2::Int
    o1step::Int
    o2step::Int
end

@inline (sf::SIMDFunction{F,C1,C2})(i, x, θ) where {F,C1,C2} = sf.f(i, x, θ)
@inline (sf::SIMDFunction{F,C1,C2})(i, x, θ) where {F <: Real,C1,C2} = sf.f

"""
    SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)

Returns a `SIMDFunction` using the `gen`.

# Arguments:
- `gen`: an iterable function specified in `Base.Generator` format
- `o0`: offset for the function evaluation
- `o1`: offset for the derivative evalution
- `o2`: offset for the second-order derivative evalution
"""
function SIMDFunction(T, gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)
    _simdfunction(T, gen.f(ParSource()), o0, o1, o2)
end

function _simdfunction(T, f::F, o0, o1, o2) where {F<:Real}
    f = replace_T(T, f)
    SIMDFunction(
        f,
        ExaModels.Compressor{Tuple{}}(()),
        ExaModels.Compressor{Tuple{}}(()),
        o0,
        o1,
        o2,
        0,
        0,
    )
end

function _simdfunction(T, f, o0, o1, o2)
    f = replace_T(T, f)
    
    d = f(Identity(), AdjointNodeSource(NaNSource{T}()), NaNSource{T}())
    y1 = []
    ExaModels.grpass(d, nothing, y1, NaNSource{T}(), 0, T(NaN))

    t = f(Identity(), SecondAdjointNodeSource(NaNSource{T}()), NaNSource{T}())
    y2 = []
    ExaModels.hrpass0(t, nothing, y2, NaNSource{T}(), NaNSource{T}(), 0, T(NaN), T(NaN))

    a1 = unique(y1)
    o1step = length(a1)
    c1 = Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    a2 = unique(y2)
    o2step = length(a2)
    c2 = Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

    SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step)
end


@inline replace_T(t, n) = n
@inline function replace_T(t, n::Node1{F,I}) where {F, I}
    i = replace_T(t, n.inner)
    return Node1{F,typeof(i)}(i)
end
@inline function replace_T(t, n::Node2{F,I1,I2}) where {F, I1, I2}
    i1 = replace_T(t, n.inner1)
    i2 = replace_T(t, n.inner2)
    return Node2{F,typeof(i1),typeof(i2)}(i1, i2)
end
@inline replace_T(t, n::Null{T}) where T <: AbstractFloat = Null{t}(t(n.value))
@inline replace_T(::Type{T1}, n::T2) where {T1, T2 <: AbstractFloat} = T1(n)
