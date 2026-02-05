"""
    jrpass(d::D, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, cnt, adj) where {D<:AdjointNode1}

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
        e,
        e_starts,
        e_cnts,
        isexp,
        comp,
        i,
        y1,
        y2,
        o1,
        cnt,
        adj,
    ) where {D <: Union{AdjointNull, Real}}
    return cnt
end
@inline function jrpass(d::D, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, cnt, adj) where {D <: AdjointNode1}
    cnt = jrpass(d.inner, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, cnt, adj * d.y)
    return cnt
end
@inline function jrpass(d::D, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, cnt, adj) where {D <: AdjointNode2}
    cnt = jrpass(d.inner1, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, cnt, adj * d.y1)
    cnt = jrpass(d.inner2, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, cnt, adj * d.y2)
    return cnt
end
# jac_coord
@inline function jrpass(d::D, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, cnt, adj) where {D <: AdjointNodeVar}
    @inbounds y1[o1 + comp(cnt += 1)] += adj
    return cnt
end
@inline function jrpass(d::D, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, cnt, adj) where {D <: AdjointNodeExpr}
    (cnt_start, e_start) = e_starts[d.i]
    len = e_cnts[cnt_start]
    cnt += 1
    for i in 1:len
        @inbounds y1[o1 + comp(cnt)] += adj * e[e_start + i - 1]
        cnt += e_cnts[cnt_start + i]
    end
    return cnt
end
# jprod_nln
@inline function jrpass(
        d::D,
        e,
        e_starts,
        e_cnts,
        isexp,
        comp,
        o0,
        y1::Tuple{V1, V2},
        y2::Nothing,
        o1,
        cnt,
        adj,
    ) where {D <: AdjointNodeVar, V1 <: AbstractVector, V2 <: AbstractVector}
    (y, v) = y1
    @inbounds y[o0] += adj * v[d.i]
    return 0
end
# jtprod_nln
@inline function jrpass(
        d::D,
        e,
        e_starts,
        e_cnts,
        isexp,
        comp,
        o0,
        y1::Nothing,
        y2::Tuple{V1, V2},
        o1,
        cnt,
        adj,
    ) where {D <: AdjointNodeVar, V1 <: AbstractVector, V2 <: AbstractVector}
    y, v = y2
    @inbounds y[d.i] += adj * v[o0]
    return 0
end
# jac_structure
@inline function jrpass(
        d::D,
        e,
        e_starts,
        e_cnts,
        isexp,
        comp,
        o0,
        y1::V,
        y2::V,
        o1,
        cnt,
        adj,
    ) where {D <: AdjointNodeVar, I <: Integer, V <: AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    @inbounds y1[ind] = o0
    @inbounds y2[ind] = d.i
    return cnt
end
@inline function jrpass(
        d::D,
        e,
        e_starts,
        e_cnts,
        isexp,
        comp,
        o0,
        y1::V,
        y2::V,
        o1,
        cnt,
        adj,
    ) where {D <: AdjointNodeExpr, I <: Integer, V <: AbstractVector{I}}
    (cnt_start, e_start) = e_starts[d.i]
    len = e_cnts[cnt_start]
    cnt += 1
    for i in 1:len
        ind = o1 + comp(cnt)
        @inbounds y1[ind] = o0
        @inbounds y2[ind] = e[e_start + i - 1]
        cnt += e_cnts[cnt_start + i]
    end
    return cnt
end
# no rows when precomputing expressions
@inline function jrpass(
        d::D,
        e,
        e_starts,
        e_cnts,
        isexp,
        comp,
        o0,
        y1::Nothing,
        y2::V,
        o1,
        cnt,
        adj,
    ) where {D <: AdjointNodeVar, I <: Integer, V <: AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    @inbounds y2[ind] = d.i
    return cnt
end
@inline function jrpass(
        d::D,
        e,
        e_starts,
        e_cnts,
        isexp,
        comp,
        o0,
        y1::Nothing,
        y2::V,
        o1,
        cnt,
        adj,
    ) where {D <: AdjointNodeExpr, I <: Integer, V <: AbstractVector{I}}
    (cnt_start, e_start) = e_starts[d.i]
    len = e_cnts[cnt_start]
    cnt += 1
    for i in 1:len
        ind = o1 + comp(cnt)
        @inbounds y2[ind] = e[e_start + i - 1]
        cnt += e_cnts[cnt_start + i]
    end
    return cnt
end
@inline function jrpass(
        d::D,
        e,
        e_starts,
        e_cnts,
        isexp,
        comp,
        o0,
        y1::V,
        y2,
        o1,
        cnt,
        adj,
    ) where {D <: AdjointNodeVar, I <: Tuple{Tuple{Int, Int}, Int}, V <: AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    @inbounds y1[ind] = ((o0, d.i), ind)
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
function sjacobian!(e, e_starts, e_cnts, isexp, y1, y2, f, x, θ, adj)
    return @simd for i in eachindex(f.itr)
        @inbounds sjacobian!(
            isexp,
            y1,
            y2,
            f.f,
            e,
            e_starts,
            e_cnts,
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

function sjacobian!(isexp, y1, y2, f, e, e_starts, e_cnts, p, x, θ, comp, o0, o1, adj)
    s = AdjointNodeSource(x, isexp)
    graph = f(p, s, θ)
    return jrpass(graph, e, e_starts, e_cnts, isexp, comp, o0, y1, y2, o1, 0, adj)
end
