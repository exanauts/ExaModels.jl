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
    # ((((), 1), 2), 3)
    snoc_dup1 = ExaModels.grpass(d, nothing, nothing, nothing, (), NaN)
    snoc_ddup1 = ddup_snoc(snoc_dup1)
    (o1step, c1) = unsnoc(snoc_ddup1, snoc_ddup1)
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, (), NaN)
    a1 = unique(y1)
    old_o1step = length(a1)
    old_c1 = Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))
    @info a1
    @info old_o1step
    @info old_c1
    @info (old_o1step == o1step)
    @info (old_c1 == c1)
    @info snoc_dup1
    @info snoc_ddup1
    @info o1step
    @info c1

    t = f(Identity(), SecondAdjointNodeSource(nothing), nothing)
    snoc_dup2 = ExaModels.hrpass0(t, nothing, nothing, nothing, nothing, (), NaN, NaN)
    snoc_ddup2 = ddup_snoc(snoc_dup2)
    (o2step, c2) = unsnoc(snoc_ddup2, snoc_ddup2)
    @info snoc_dup2
    @info snoc_ddup2
    @info o2step
    @info c2

    SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step)
end

function unsnoc(ddup, snoc::Tuple{})
    (0, ())
end
function unsnoc(ddup, snoc::Tuple{T1,T2}) where {T1<:Tuple,T2}
    (step, comp) = unsnoc(ddup, snoc[1])
    (step+1, (comp..., find_snoc(ddup, snoc[2])))
end

function find_snoc(ddup::Tuple{}, x)
    @error "failed to find x in dduped compressor"
end
function find_snoc(ddup, x)
    ddup[2] == x ? 1 : 1 + find_snoc(ddup[1], x)
end

function ddup_snoc(snoc::Tuple{})
    return ()
end
function ddup_snoc(snoc::Tuple{T1, T2}) where {T1<:Tuple,T2}
    return snoc_insert(ddup_snoc(snoc[1]), snoc[2])
end

function snoc_insert(snoc::Tuple{S, T}, x::T) where {S,T}
    return snoc
end
function snoc_insert(snoc::Tuple{S, T1}, x::T2) where {S,T1,T2}
    return (snoc_insert(snoc[1], x), snoc[2])
end
function snoc_insert(snoc::Tuple{}, x::T) where {T}
    return ((), x)
end
