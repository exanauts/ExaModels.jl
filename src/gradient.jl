"""
    drpass(d::D, y, adj)

Performs dense gradient evaluation via the reverse pass on the computation (sub)graph formed by forward pass

# Arguments:
- `d`: first-order computation (sub)graph
- `y`: result vector
- `adj`: adjoint propagated up to the current node
"""
@inline function drpass(e, e_starts, e_cnts, d::D, y, adj) where {D<:AdjointNull}
    nothing
end
@inline function drpass(e, e_starts, e_cnts, d::D, y, adj) where {D<:AdjointNode1}
    drpass(e, e_starts, e_cnts, d.inner, y, adj * d.y)
    nothing
end
@inline function drpass(e, e_starts, e_cnts, d::D, y, adj) where {D<:AdjointNode2}
    drpass(e, e_starts, e_cnts, d.inner1, y, adj * d.y1)
    drpass(e, e_starts, e_cnts, d.inner2, y, adj * d.y2)
    nothing
end
@inline function drpass(e, e_starts, e_cnts, d::D, y, adj) where {D<:AdjointNodeVar}
    @inbounds y[d.i] += adj
    nothing
end
@inline function drpass(e, e_starts, e_cnts, d::D, y, adj) where {D<:AdjointNodeExpr}
    y[d.i] += e[e_starts[d.i][2]]
    nothing
end

"""
    gradient!(y, f, x, adj)

Performs dense gradient evalution

# Arguments:
- `y`: result vector
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `adj`: initial adjoint
"""
function gradient!(isexp, e, e_starts, e_cnts, y, f, x, θ, adj)
    @simd for k in eachindex(f.itr)
        @inbounds gradient!(isexp, e, e_starts, e_cnts, y, f.f, x, θ, f.itr[k], adj)
    end
    return y
end
function gradient!(isexp, e, e_starts, e_cnts, y, f, x, θ, p, adj)
    graph = f(p, AdjointNodeSource(x, nothing), θ)
    drpass(e, e_starts, e_cnts, graph, y, adj)
    return y
end

"""
    grpass(d::D, comp, y, o1, cnt, adj)

Performs dsparse gradient evaluation via the reverse pass on the computation (sub)graph formed by forward pass

# Arguments:
- `d`: first-order computation (sub)graph
- `comp`: a `Compressor`, which helps map counter to sparse vector index
- `y`: result vector
- `o1`: index offset
- `cnt`: counter
- `adj`: adjoint propagated up to the current node
    """
@inline function grpass(
    d::D,
    comp,
    y,
    o1,
    cnt,
    adj,
) where {D<:Union{AdjointNull,ParIndexed,Real}}
    return cnt
end
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
    @inbounds y[o1+comp(cnt+=1)] += adj
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

""" sgradient!(y, f, x, adj)

Performs sparse gradient evalution

# Arguments:
- `y`: result vector
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `adj`: initial adjoint
"""
function sgradient!(y, f, x, θ, adj, isexp)
    @simd for k in eachindex(f.itr)
        @inbounds sgradient!(y, f.f, f.itr[k], x, θ, f.itr.comp1, offset1(f, k), adj)
    end
    return y
end

function sgradient!(y, f, p, x, θ, comp, o1, adj, isexp)
    graph = f(p, AdjointNodeSource(x, nothing), θ)
    grpass(graph, comp, y, o1, 0, adj)
    return y
end
