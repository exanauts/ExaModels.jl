function egrad!(egrad, m, x)
    _egrad!(m.isexp, egrad, m.exps, x, m.θ)
end
function _egrad!(isexp, egrad, exp, x, θ)
    if typeof(exp) <: ExpressionNull
        return 0
    end
    o = _egrad!(isexp, egrad, exp.inner, x, θ)
    @simd for i in eachindex(exp.itr)
        graph = exp.f.f(exp.itr[i], AdjointNodeSource(x), θ)
        @info typeof(egrad)
        @info Base.size(egrad)
        @info Base.size(egrad[i+o])
        drpass(isexp, egrad, graph, egrad[i+o], one(eltype(x)))
        exit()
    end
    return o + total(exp.ns)
end

"""
    drpass(d::D, y, adj)

Performs dense gradient evaluation via the reverse pass on the computation (sub)graph formed by forward pass

# Arguments:
- `d`: first-order computation (sub)graph
- `y`: result vector
- `adj`: adjoint propagated up to the current node
"""
@inline function drpass(isexp, egrad, d::D, y, adj) where {D<:AdjointNull}
    nothing
end
@inline function drpass(isexp, egrad, d::D, y, adj) where {D<:AdjointNode1}
    offset = drpass(isexp, egrad, d.inner, y, adj * d.y)
    nothing
end
@inline function drpass(isexp, egrad, d::D, y, adj) where {D<:AdjointNode2}
    offset = drpass(isexp, egrad, d.inner1, y, adj * d.y1)
    offset = drpass(isexp, egrad, d.inner2, y, adj * d.y2)
    nothing
end
@inline function drpass(isexp, egrad, d::D, y, adj) where {D<:AdjointNodeVar}
    if isexp[d.i] == typemax(UInt)
        @inbounds y[d.i] += adj
    else
        @inbounds y += egrad[isexp[d.i]]
    end
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
function gradient!(y, f, x, θ, adj)
    @simd for k in eachindex(f.itr)
        @inbounds gradient!(y, f.f, x, θ, f.itr[k], adj)
    end
    return y
end
function gradient!(isexp, egrad, y, f, x, θ, p, adj)
    graph = f(p, AdjointNodeSource(x), θ)
    drpass(isexp, egrad, graph, y, adj)
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
function sgradient!(y, f, x, θ, adj)
    @simd for k in eachindex(f.itr)
        @inbounds sgradient!(y, f.f, f.itr[k], x, θ, f.itr.comp1, offset1(f, k), adj)
    end
    return y
end

function sgradient!(y, f, p, x, θ, comp, o1, adj)
    graph = f(p, AdjointNodeSource(x), θ)
    grpass(graph, comp, y, o1, 0, adj)
    return y
end
