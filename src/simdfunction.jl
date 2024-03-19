@inline (a::Pair{P,S} where {P,S<:AbstractNode})(i, x) = a.second(i, x)

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

struct SIMDFunction{F,C1,C2,T}
    f::F
    comp1::C1
    comp2::C2
    o0::Int
    o1::Int
    o2::Int
    o1step::Int
    o2step::Int
    tsize::T
end

"""
    SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)

Returns a `SIMDFunction` using the `gen`.

# Arguments:
- `gen`: an iterable function specified in `Base.Generator` format
- `o0`: offset for the function evaluation
- `o1`: offset for the derivative evalution
- `o2`: offset for the second-order derivative evalution
"""
function SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0; tsize = ())

    f = gen.f(Par(eltype(gen.iter)))

    _simdfunction(f, o0, o1, o2; tsize = tsize)
end

function _simdfunction(f, o0, o1, o2; tsize = ())
    d = f(Identity(), AdjointNodeSource(nothing))
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    t = f(Identity(), SecondAdjointNodeSource(nothing))
    y2 = []
    ExaModels.hrpass0(t, nothing, y2, nothing, nothing, 0, NaN, NaN)

    a1 = unique(y1)
    o1step = length(a1)
    c1 = Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    a2 = unique(y2)
    o2step = length(a2)
    c2 = Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

    SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step, tsize)
end
