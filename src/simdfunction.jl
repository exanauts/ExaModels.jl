@inbounds @inline (a::Pair{P, S} where {P <: AbstractNode, S <: AbstractNode})(i,x) = a.second(i,x)  

struct Compressor{I}
    inner::I
end
@inbounds @inline (i::Compressor{I})(n) where I = i.inner[n]

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
function SIMDFunction(gen::Base.Generator, o0=0, o1=0, o2=0)

    p = Par(eltype(gen.iter))
    f = gen.f(p)

    
    d = f(Identity(),AdjointNodeSource(NaNSource{Float16}()))
    y1 = []
    SIMDiff.grpass(d,nothing,y1,nothing,0,NaN16)

    t = f(Identity(),SecondAdjointNodeSource(NaNSource{Float16}()))
    y2 = []
    SIMDiff.hrpass0(t,nothing,y2,nothing,nothing,0,NaN16,NaN16)

    a1 = unique(y1)
    o1step = length(a1)
    c1 = Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    a2 = unique(y2)
    o2step = length(a2)
    c2 = Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))
    
    SIMDFunction(f,c1, c2, o0, o1, o2, o1step, o2step)
end



