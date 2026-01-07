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
function SIMDFunction(gen::Base.Generator, full_exp_refs1, full_exp_refs2, exps, isexp, o0 = 0, o1 = 0, o2 = 0)
    f = gen.f(ParSource())
    _simdfunction(f, full_exp_refs1, full_exp_refs2, exps, isexp, o0, o1, o2)
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

function exp_index(exps, exp)
    exp_i = 1
    while exps != ExpressionNull && exps.offset > exp
        exps = exps.inner
        exp_i += 1
    end
    if exps == ExpressionNull
        @error "ExaModels extracted the incorrect expression offset"
    end
    return exp_i
end

function get_full_exp_refs(full_exp_refs, exps, isexp, y_raw)
    y = []
    for y_rawi in y_raw
        offset = typeof(y_rawi) <: Node2{typeof(+), T, Int} where T ? y_rawi.inner2+1 : 1
        if isexp[offset] != 0
            exp_i = exp_index(exps, offset)
            Base.append!(y, full_exp_refs[exp_i])
        else
            push!(y, y_rawi)
        end
    end
    return y
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

function _simdfunction(f, full_exp_refs1, full_exp_refs2, exps, isexp, o0, o1, o2)
    y1_raw = []
    d = f(Identity(), AdjointNodeSource(nothing, nothing), nothing)
    ExaModels.grpass(d, nothing, y1_raw, nothing, 0, NaN)
    y1 = get_full_exp_refs(full_exp_refs1, exps, isexp, y1_raw)

    y2_raw = []
    t = f(Identity(), SecondAdjointNodeSource(nothing, nothing), nothing)
    ExaModels.hrpass(t, nothing, y2_raw, nothing, nothing, 0, NaN, NaN)
    y2 = get_full_exp_refs(full_exp_refs2, exps, isexp, y2_raw)

    a1 = unique(y1)
    o1step = length(a1)
    c1 = Compressor(Tuple(findfirst(isequal(di), a1) for di in y1))

    a2 = unique(y2)
    o2step = length(a2)
    c2 = Compressor(Tuple(findfirst(isequal(di), a2) for di in y2))

    SIMDFunction(f, c1, c2, o0, o1, o2, o1step, o2step)
end
