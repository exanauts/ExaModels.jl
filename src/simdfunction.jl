@inline (a::Pair{P,S} where {P<:AbstractNode,S<:AbstractNode})(i, x) =
    a.second(i, x)

"""
    Compressor{I}

DOCSTRING

# Fields:
- `inner::I`: DESCRIPTION
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

"""
    SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)

DOCSTRING

# Arguments:
- `gen`: DESCRIPTION
- `o0`: DESCRIPTION
- `o1`: DESCRIPTION
- `o2`: DESCRIPTION
"""
function SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)

    p = Par(eltype(gen.iter))
    f = gen.f(p)


    d = f(Identity(), AdjointNodeSource())
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, nothing)

    t = f(Identity(), SecondAdjointNodeSource())
    y2 = []
    ExaModels.hrpass0(t, nothing, y2, nothing, nothing, 0, nothing, nothing)

    a1 = unique(y1)
    o1step = length(a1)
    c1 = Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    a2 = unique(y2)
    o2step = length(a2)
    c2 = Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

    SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step)
end
