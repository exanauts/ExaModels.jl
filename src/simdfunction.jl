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
function SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)

    f = gen.f(ParSource())

    _simdfunction(f, o0, o1, o2)
end

function _simdfunction(f::F, o0, o1, o2) where {F<:Real}
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

function _simdfunction(f, o0, o1, o2)
    d = f(Identity(), AdjointNodeSource(nothing), nothing)
    (ddup1, c1) = ExaModels.grpass(d, nothing, nothing, nothing, ((), ()), NaN)
    o1step = snoc_len(ddup1)

    t = f(Identity(), SecondAdjointNodeSource(nothing), nothing)
    (ddup2, c2) = ExaModels.hrpass0(t, nothing, nothing, nothing, nothing, ((), ()), NaN, NaN)
    o2step = snoc_len(ddup2)

    SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step)
end

function snoc_len(snoc::Tuple{})
    0
end
function snoc_len(snoc::Tuple{T1,T2}) where {T1<:Tuple,T2}
    1+snoc_len(snoc[1])
end

function update_snoc(ddup, inds, x)
    return update_snoc(ddup, inds, x, 1)
end

function update_snoc(ddup::Tuple{S, T1}, inds, x::T2, ind) where {S,T1,T2}
    (new_ddup, new_inds) = update_snoc(ddup[1], inds, x, ind+1)
    return ((new_ddup, ddup[2]), new_inds)
end
function update_snoc(ddup::Tuple{S, T}, inds, x::T, ind) where {S,T}
    return (ddup, (inds..., ind))
end
function update_snoc(ddup::Tuple{}, inds, x::T, ind) where {T}
    return (((), x), (inds..., ind))
end
