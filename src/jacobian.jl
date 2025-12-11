struct ConstVector{T} <: AbstractVector{T}
    v::T
    l::Int
end

function Base.checkbounds(v::ConstVector, i::Int)
    if !checkindex(Bool, 1:v.l, i)
        throw(BoundsError(v, r))
    end
end
function Base.getindex(v::ConstVector, i::Int)
    @boundscheck checkbounds(v, i)
    return v.v
end

@inline function ejac!(m, ey1, ey2, x, v)
    _ejac!(m.isexp, ey1, ey2, m.exps, x, m.θ, v)
end
@inline @views function _ejac!(isexp, ey1, ey2, exp, x, θ, v)
    if typeof(exp) <: ExpressionNull
        return 0
    end
    o = _ejac!(isexp, ey1, ey2, exp.inner, x, θ, v)
    @simd for i in eachindex(exp.itr)
        y1_v = isnothing(ey1) ? nothing : (ey1[i+o:i+o], v)
        y2_v = isnothing(ey2) ? nothing : (ey2[:,i+o], v)
        @inbounds sjacobian!(
            isexp,
            ey1,
            ey2,
            y1_v,
            y2_v,
            exp.f.f,
            exp.itr[i],
            x,
            θ,
            exp.f.comp1,
            1,
            offset1(exp.f, i),
            one(eltype(x)),
        )
    end
    return o + total(exp.size)
end

"""
    jrpass(d::D, isexp, ey1, ey2, comp, i, y1, y2, o1, cnt, adj)

Performs sparse jacobian evaluation via the reverse pass on the computation (sub)graph formed by forward pass

# Arguments:
- `d`: first-order computation (sub)graph
- `comp`: a `Compressor`, which helps map counter to sparse vector index
- `i`: constraint index (this is `i`-th constraint)
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `o1`: index offset
- `cnt`: counter
- `adj`: adjoint propagated up to the current node
"""
@inline function jrpass(
    d::D,
    isexp,
    ey1,
    ey2,
    comp,
    i,
    y1,
    y2,
    o1,
    cnt,
    adj,
) where {D<:Union{AdjointNull,Real}}
    return cnt
end
@inline function jrpass(d::D, isexp, ey1, ey2, comp, i, y1, y2, o1, cnt, adj) where {D<:AdjointNode1}
    cnt = jrpass(d.inner, isexp, ey1, ey2, comp, i, y1, y2, o1, cnt, adj * d.y)
    return cnt
end
@inline function jrpass(d::D, isexp, ey1, ey2, comp, i, y1, y2, o1, cnt, adj) where {D<:AdjointNode2}
    cnt = jrpass(d.inner1, isexp, ey1, ey2, comp, i, y1, y2, o1, cnt, adj * d.y1)
    cnt = jrpass(d.inner2, isexp, ey1, ey2, comp, i, y1, y2, o1, cnt, adj * d.y2)
    return cnt
end
# jac_coord
@inline function jrpass(d::D, isexp, ey1, ey2, comp, i, y1, y2, o1, cnt, adj) where {D<:AdjointNodeVar}
    @inbounds y1[o1+comp(cnt+=1)] += adj
    return cnt
end
# jprod_nln
@inline function jrpass(
    d::D,
    isexp,
    ey1,
    ey2,
    comp,
    i,
    y1::Tuple{V1,V2},
    y2::Nothing,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,V1<:AbstractVector,V2<:AbstractVector}
    (y, v) = y1
    @inbounds y[i] += adj * v[d.i]
    return 0
end
# jtprod_nln
@inline function jrpass(
    d::D,
    isexp,
    ey1,
    ey2,
    comp,
    i,
    y1::Nothing,
    y2::Tuple{V1,V2},
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,V1<:AbstractVector,V2<:AbstractVector}
    y, v = y2
    @inbounds y[d.i] += adj * v[i]
    return 0
end
# jac_structure
@inline function jrpass(
    d::D,
    isexp,
    ey1,
    ey2,
    comp,
    i,
    y1::V,
    y2::V,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,I<:Integer,V<:AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    @inbounds y1[ind] = i
    @inbounds y2[ind] = d.i
    return cnt
end
@inline function jrpass(
    d::D,
    isexp,
    ey1,
    ey2,
    comp,
    i,
    y1::V,
    y2,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeVar,I<:Tuple{Tuple{Int,Int},Int},V<:AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    @inbounds y1[ind] = ((i, d.i), ind)
    return cnt
end
# jprod_nln expr
@inline function jrpass(
    d::D,
    isexp,
    ey1,
    ey2,
    comp,
    i,
    y1::Tuple{V1,V2},
    y2,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeExpr,V1<:AbstractVector,V2<:AbstractVector}
    (y, v) = y1
    @inbounds y[i] .+= ey1[isexp[d.i]]
    return 0
end
# jtprod_nln expr
@inline function jrpass(
    d::D,
    isexp,
    ey1,
    ey2,
    comp,
    i,
    y1,
    y2::Tuple{V1,V2},
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeExpr,V1<:AbstractVector,V2<:AbstractVector}
    y, v = y2
    @inbounds y .+= (adj * v[i]) .* ey2[:, isexp[d.i]]
    return 0
end
# jac_structure expr
@inline function jrpass(
    d::D,
    isexp,
    ey1,
    ey2,
    comp,
    i,
    y1::V,
    y2::V,
    o1,
    cnt,
    adj,
) where {D<:AdjointNodeExpr,I<:Integer,V<:AbstractVector{I}}
    expr_size = ey1[isexp[d.i]]
    @inbounds y2[] = ey2
    for i in 1:expr_size
        ind = o1 + comp(cnt += 1)
        @inbounds y1[ind] = i
    end
    return cnt
end

"""
    sjacobian!(y1, y2, f, x, adj)

Performs sparse jacobian evalution

# Arguments:
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `adj`: initial adjoint
"""
function sjacobian!(isexp, ey1, ey2, y1, y2, f, x, θ, adj)
    @info "sjacobian!"
    @simd for i in eachindex(f.itr)
        @inbounds sjacobian!(
            isexp,
            ey1,
            ey2,
            y1,
            y2,
            f.f,
            f.itr[i],
            x,
            θ,
            f.f.comp1,
            offset0(f, i),
            offset1(f, i),
            adj,
        )
    end
end

function sjacobian!(isexp, ey1, ey2, y1, y2, f, p, x, θ, comp, o0, o1, adj)
    graph = f(p, AdjointNodeSource(x, isexp), θ)
    jrpass(graph, isexp, ey1, ey2, comp, o0, y1, y2, o1, 0, adj)
end
