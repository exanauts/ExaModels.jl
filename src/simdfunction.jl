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
@inline (sf::SIMDFunction{F,C1,C2})(i, x, θ) where {F<:Real,C1,C2} = sf.f

"""
    SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0)

Returns a `SIMDFunction` using the `gen`.

# Arguments:
- `gen`: an iterable function specified in `Base.Generator` format
- `o0`: offset for the function evaluation
- `o1`: offset for the derivative evalution
- `o2`: offset for the second-order derivative evalution
"""
function SIMDFunction(gen::Base.Generator, offset_exps, isexp, o0=0, o1=0, o2=0)
    f = gen.f(ParSource())
    _simdfunction(f, offset_exps, isexp, o0, o1, o2)
end

function _simdfunction(f::F, exps, isexp, o0, o1, o2) where {F<:Real}
    SIMDFunction(
        f,
        ExaModels.Compressor{Tuple{}}(()),
        ExaModels.Compressor{Tuple{}}(()),
        ExaModels.Compressor{Tuple{}}(()),
        ExaModels.Compressor{Tuple{}}(()),
        o0,
        o1,
        o2,
        0,
        0,
    )
end

function compress_ref_cnts(y, a)
    i = 1
    cnt = 0
    ret = []
    for yi in y
        cnt += 1
        if i <= length(a) && a[i] == yi
            if i != 1
                push!(ret, cnt)
            end
            i += 1
            cnt = 0
        end
    end
    push!(ret, cnt)
    ret
end

function _simdfunction(f, offset_exps, isexp, o0, o1, o2)
    y1 = []
    d = f(Identity(), AdjointNodeSource(nothing, offset_exps), nothing)
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)
    dump(d)
    dump(y1)

    y2 = []
    t = f(Identity(), SecondAdjointNodeSource(nothing, offset_exps), nothing)
    ExaModels.hrpass0(nothing, nothing, nothing, nothing, nothing, nothing, t, nothing, y2, nothing, nothing, 0, NaN, NaN)
    dump(t)
    dump(y2)

    a1 = unique(y1)
    o1step = length(a1)
    c1 = Compressor(Tuple(findfirst(isequal(di), a1) for di in y1))

    a2 = unique(y2)
    o2step = length(a2)
    c2 = Compressor(Tuple(findfirst(isequal(di), a2) for di in y2))

    f = SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step)
    dump(y2)
    dump(a2)
    dump(c2)
    if length(f.comp2.inner) == 0
        dump(t)
        dump(f)
        dump(stacktrace())
        exit()
    end
    return f
end
