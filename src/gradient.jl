"""
    drpass(d::D, y, adj)

DOCSTRING

# Arguments:
- `d`: DESCRIPTION
- `y`: DESCRIPTION
- `adj`: DESCRIPTION
"""
@inline function drpass(d::D, y, adj) where {D<:AdjointNode1}
    offset = drpass(d.inner, y, adj * d.y)
    nothing
end
@inline function drpass(d::D, y, adj) where {D<:AdjointNode2}
    offset = drpass(d.inner1, y, adj * d.y1)
    offset = drpass(d.inner2, y, adj * d.y2)
    nothing
end
@inline function drpass(d::D, y, adj) where {D<:AdjointNodeVar}
    @inbounds y[d.i] += adj
    nothing
end
@inline function drpass(f::F, x, y, adj) where {F<:SIMDFunction} end

"""
    gradient!(y, f, x, adj)

DOCSTRING

# Arguments:
- `y`: DESCRIPTION
- `f`: DESCRIPTION
- `x`: DESCRIPTION
- `adj`: DESCRIPTION
"""
function gradient!(y, f, x, adj)
    @simd for k in eachindex(f.itr)
        @inbounds drpass(f.f.f(f.itr[k], AdjointNodeSource(x)), y, adj)
    end
    return y
end

"""
    grpass(d::D, comp, y, o1, cnt, adj)

DOCSTRING

# Arguments:
- `d`: DESCRIPTION
- `comp`: DESCRIPTION
- `y`: DESCRIPTION
- `o1`: DESCRIPTION
- `cnt`: DESCRIPTION
- `adj`: DESCRIPTION
"""
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNode1}
    cnt = grpass(d.inner, comp, y, o1, cnt, adj * d.y)
    return cnt
end
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNode2}
    cnt = grpass(d.inner1, comp, y, o1, cnt, adj * d.y1)
    cnt = grpass(d.inner2, comp, y, o1, cnt, adj * d.y2)
    return cnt
end
@inline function grpass(d::D, comp, y, o1, cnt, adj) where {D<:AdjointNodeVar}
    @inbounds y[o1+comp(cnt += 1)] += adj
    return cnt
end

@inline function grpass(d::AdjointNodeVar, comp::Nothing, y, o1, cnt, adj) # despecialization
    push!(y, d.i)
    return (cnt += 1)
end
@inline function grpass(
    d::D,
    comp,
    y::V,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,V<:AbstractVector{Tuple{Int,Int}}}
    ind = o1 + comp(cnt += 1)
    @inbounds y[ind] = (d.i, ind)
    return cnt
end

"""
    sgradient!(y, f, x, adj)

DOCSTRING

# Arguments:
- `y`: DESCRIPTION
- `f`: DESCRIPTION
- `x`: DESCRIPTION
- `adj`: DESCRIPTION
"""
function sgradient!(y, f, x, adj)
    @simd for k in eachindex(f.itr)
        @inbounds grpass(f.f.f(f.itr[k], AdjointNodeSource(x)), f.itr.comp1, y, offset1(f, k), 0, adj)
    end
    return y
end
