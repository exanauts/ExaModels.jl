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
@inline function SIMDFunction(T, gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)
    _simdfunction(T, gen.f(ParSource()), o0, o1, o2)
end

@inline function _simdfunction(T, f::F, o0, o1, o2) where {F<:Real}
    f = replace_T(T, f)
    SIMDFunction(
        f,
        Compressor{Tuple{}}(()),
        Compressor{Tuple{}}(()),
        o0,
        o1,
        o2,
        0,
        0,
    )
end

@inline function _simdfunction(T, f, o0, o1, o2)
    f = replace_T(T, f)
    
    d = f(Identity(), AdjointNodeSource(NaNSource{T}()), NaNSource{T}())
    a1, _ = grpass(d, nothing, nothing, NaNSource{T}(), ((),()), T(NaN))
    
    t = f(Identity(), SecondAdjointNodeSource(NaNSource{T}()), NaNSource{T}())
    a2, _ = hrpass0(t, nothing, nothing, NaNSource{T}(), NaNSource{T}(), ((), ()), T(NaN), T(NaN))

    o1step = maximum(a1; init = 0)
    o2step = maximum(a2; init = 0)
    c1 = Compressor(a1)
    c2 = Compressor(a2)

    SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step)
end

struct NodeWrap{I}
    inner::I
    unique::Bool
end
@inline function update_sparsity(cnt::Int, uni::Bool, new, y0, ys...)
    cnt += y0.unique
    uni = uni && new != y0.inner
    cntnew, ys = update_sparsity(cnt, uni, new, ys...)
    return (uni ? cntnew : cnt)::Int, (y0, ys...)
end
@inline update_sparsity(cnt::Int, uni::Bool, new) = cnt+1, (NodeWrap(new, uni),)


@inline replace_T(t, n::Union{AbstractNode,Real}) = n
@inline replace_T(t, (a,b)::Pair) = a => replace_T(t, b)
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
