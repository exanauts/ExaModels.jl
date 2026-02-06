"""
    hdrpass(e, e_starts, e_cnts, isexp, t1::T1, t2::T2, comp, y1, y2, o2, cnt, adj)

Performs sparse hessian evaluation (`(df1/dx)(df2/dx)'` portion) via the reverse pass on the computation (sub)graph formed by second-order forward pass

# Arguments:
- `e`: expression Jacobian values
- `e_starts`: expression start indices
- `e_cnts`: expression counts
- `isexp`: expression indicator vector
- `t1`: second-order computation (sub)graph regarding f1
- `t2`: second-order computation (sub)graph regarding f2
- `comp`: a `Compressor`, which helps map counter to sparse vector index
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `o2`: index offset
- `cnt`: counter
- `adj`: second adjoint propagated up to the current node
"""
@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNode1, T2 <: SecondAdjointNode1}
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y * t2.y)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNode1,
        t2::SecondAdjointNode1,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    )  # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y * t2.y)
    return cnt
end


@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: Union{SecondAdjointNodeVar, SecondAdjointNodeExpr}, T2 <: SecondAdjointNode1}
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner, comp, y1, y2, o2, cnt, adj * t2.y)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNodeVar,
        t2::SecondAdjointNode1,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    )  # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner, comp, y1, y2, o2, cnt, adj * t2.y)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNodeExpr,
        t2::SecondAdjointNode1,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    )  # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner, comp, y1, y2, o2, cnt, adj * t2.y)
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNode1, T2 <: Union{SecondAdjointNodeVar, SecondAdjointNodeExpr}}
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2, comp, y1, y2, o2, cnt, adj * t1.y)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNode1,
        t2::SecondAdjointNodeVar,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    )  # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2, comp, y1, y2, o2, cnt, adj * t1.y)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNode1,
        t2::SecondAdjointNodeExpr,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    )  # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2, comp, y1, y2, o2, cnt, adj * t1.y)
    return cnt
end


@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNode2, T2 <: SecondAdjointNode2}
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y2)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y2)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNode2,
        t2::SecondAdjointNode2,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y2)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y2)
    return cnt
end


@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNode1, T2 <: SecondAdjointNode2}
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y * t2.y2)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNode1,
        t2::SecondAdjointNode2,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y * t2.y2)
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNode2, T2 <: SecondAdjointNode1}
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNode2,
        t2::SecondAdjointNode1,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y)
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: Union{SecondAdjointNodeVar, SecondAdjointNodeExpr}, T2 <: SecondAdjointNode2}
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner1, comp, y1, y2, o2, cnt, adj * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner2, comp, y1, y2, o2, cnt, adj * t2.y2)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNodeVar,
        t2::SecondAdjointNode2,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner1, comp, y1, y2, o2, cnt, adj * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner2, comp, y1, y2, o2, cnt, adj * t2.y2)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNodeExpr,
        t2::SecondAdjointNode2,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner1, comp, y1, y2, o2, cnt, adj * t2.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1, t2.inner2, comp, y1, y2, o2, cnt, adj * t2.y2)
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNode2, T2 <: Union{SecondAdjointNodeVar, SecondAdjointNodeExpr}}
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2, comp, y1, y2, o2, cnt, adj * t1.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2, comp, y1, y2, o2, cnt, adj * t1.y2)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNode2,
        t2::SecondAdjointNodeVar,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2, comp, y1, y2, o2, cnt, adj * t1.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2, comp, y1, y2, o2, cnt, adj * t1.y2)
    return cnt
end
function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNode2,
        t2::SecondAdjointNodeExpr,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) # despecialized
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner1, t2, comp, y1, y2, o2, cnt, adj * t1.y1)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t1.inner2, t2, comp, y1, y2, o2, cnt, adj * t1.y2)
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNodeVar, T2 <: SecondAdjointNodeVar}
    i, j = t1.i, t2.i
    @inbounds if i == j
        y1[o2 + comp(cnt += 1)] += 2 * adj
    else
        y1[o2 + comp(cnt += 1)] += adj
    end
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNodeExpr, T2 <: SecondAdjointNodeVar}
    (cnt_start, e_start) = e_starts[t1.i]
    len = e_cnts[cnt_start]
    cnt += 1
    for i in 1:len
        @inbounds y1[o2 + comp(cnt)] += e[e_start + i - 1] * adj
        cnt += e_cnts[cnt_start + i]
    end
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNodeVar, T2 <: SecondAdjointNodeExpr}
    (cnt_start, e_start) = e_starts[t2.i]
    len = e_cnts[cnt_start]
    cnt += 1
    for i in 1:len
        @inbounds y1[o2 + comp(cnt)] += e[e_start + i - 1] * adj
        cnt += e_cnts[cnt_start + i]
    end
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNodeExpr, T2 <: SecondAdjointNodeExpr}
    (cnt_start1, e_start1) = e_starts[t1.i]
    len1 = e_cnts[cnt_start1]
    (cnt_start2, e_start2) = e_starts[t2.i]
    len2 = e_cnts[cnt_start2]

    cnt += 1
    for i in 1:len1
        val1 = e[e_start1 + i - 1]
        for j in 1:len2
            val2 = e[e_start2 + j - 1]
            ind = o2 + comp(cnt)
            @inbounds if t1.i == t2.i && i == j
                y1[ind] += 2 * val1 * val2 * adj
            else
                y1[ind] += val1 * val2 * adj
            end
            cnt += e_cnts[cnt_start2 + j]
        end
        cnt += e_cnts[cnt_start1 + i]
    end
    return cnt
end


@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1::Tuple{V1, V2},
        y2,
        o2,
        cnt,
        adj,
    ) where {
        T1 <: SecondAdjointNodeVar,
        T2 <: SecondAdjointNodeVar,
        V1 <: AbstractVector,
        V2 <: AbstractVector,
    }
    i, j = t1.i, t2.i
    y, v = y1
    @inbounds if i == j
        y[i] += 2 * adj * v[i]
    else
        y[i] += adj * v[j]
        y[j] += adj * v[i]
    end
    return (cnt += 1)
end

"""
    hrpass(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t::D, comp, y1, y2, o2, cnt, adj, adj2)

Performs sparse hessian evaluation (`d²f/dx²` portion) via the reverse pass on the computation (sub)graph formed by second-order forward pass

# Arguments:
- `e`: expression Jacobian values (e1)
- `e_starts`: expression Jacobian start indices
- `e_cnts`: expression Jacobian counts  
- `e2`: expression Hessian values
- `e2_starts`: expression Hessian start indices
- `e2_cnts`: expression Hessian counts
- `isexp`: expression indicator vector
- `comp`: a `Compressor`, which helps map counter to sparse vector index
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `o2`: index offset
- `cnt`: counter
- `adj`: first adjoint propagated up to the current node
- `adj2`: second adjoint propagated up to the current node
"""

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: Union{SecondAdjointNull, Real}}
    return cnt
end

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNodeExpr}
    (cnt_start2, e_start2) = e2_starts[t.i]
    len2 = e2_cnts[cnt_start2]
    cnt += 1
    for i in 1:len2
        @inbounds y1[o2 + comp(cnt)] += adj * e2[e_start2 + i - 1]
        cnt += e2_cnts[cnt_start2 + i]
    end
    return cnt
end

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1::V,
        y2::V,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNodeExpr, I <: Integer, V <: AbstractVector{I}}
    (cnt_start2, e_start2) = e2_starts[t.i]
    len2 = e2_cnts[cnt_start2]
    cnt += 1
    for i in 1:len2
        ind = o2 + comp(cnt)
        val = e2[e_start2 + i - 1]
        r = unpack_row(val)
        c = unpack_col(val)
        if y1 === y2
            if r != 0 || c != 0
                @inbounds y1[ind] = pack_indices(r, c)
            end
        else
            if r != 0 || c != 0
                @inbounds y1[ind] = r
                @inbounds y2[ind] = c
            end
        end
        cnt += e2_cnts[cnt_start2 + i]
    end
    return cnt
end

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNode1}
    cnt = hrpass(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner, comp, y1, y2, o2, cnt, adj * t.y, adj2 * (t.y)^2 + adj * t.h)
    return cnt
end

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNode2}
    adj2y1y2 = adj2 * t.y1 * t.y2
    adjh12 = adj * t.h12
    cnt = hrpass(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner1, comp, y1, y2, o2, cnt, adj * t.y1, adj2 * (t.y1)^2 + adj * t.h11)
    cnt = hrpass(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner2, comp, y1, y2, o2, cnt, adj * t.y2, adj2 * (t.y2)^2 + adj * t.h22)
    cnt = hdrpass(e, e_starts, e_cnts, isexp, t.inner1, t.inner2, comp, y1, y2, o2, cnt, adj2y1y2 + adjh12)
    return cnt
end

@inline hrpass0(args...) = hrpass(args...)

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {N <: Union{FirstFixed{typeof(*)}, SecondFixed{typeof(*)}}, D <: SecondAdjointNode1{N}}
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner, comp, y1, y2, o2, cnt, adj * t.y, adj2 * (t.y)^2)
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {N <: Union{FirstFixed{typeof(+)}, SecondFixed{typeof(+)}}, D <: SecondAdjointNode1{N}}
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNode1{FirstFixed{typeof(-)}}}
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner, comp, y1, y2, o2, cnt, -adj, adj2)
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNode1{SecondFixed{typeof(-)}}}
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNode1{typeof(+)}}
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNode1{typeof(-)}}
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner, comp, y1, y2, o2, cnt, -adj, adj2)
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNode2{typeof(+)}}
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner1, comp, y1, y2, o2, cnt, adj, adj2)
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner2, comp, y1, y2, o2, cnt, adj, adj2)
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::D,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {D <: SecondAdjointNode2{typeof(-)}}
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner1, comp, y1, y2, o2, cnt, adj, adj2)
    cnt = hrpass0(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t.inner2, comp, y1, y2, o2, cnt, -adj, adj2)
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::T,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {T <: SecondAdjointNodeVar}
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::T,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {T <: SecondAdjointNodeVar}
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::T,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {T <: SecondAdjointNodeExpr}
    (cnt_start2, e_start2) = e2_starts[t.i]
    len2 = e2_cnts[cnt_start2]
    cnt += 1
    for i in 1:len2
        @inbounds y1[o2 + comp(cnt)] += adj * e2[e_start2 + i - 1]
        cnt += e2_cnts[cnt_start2 + i]
    end


    return cnt
end

function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::SecondAdjointNodeVar,
        t2::SecondAdjointNodeVar,
        comp::Nothing,
        y1,
        y2,
        o2,
        cnt,
        adj,
    )
    cnt += 1
    push!(y1, (t1.i, t2.i))
    return cnt
end

function hrpass(e, e_starts, e_cnts, e2, e2_starts, e2_cnts, isexp, t::SecondAdjointNodeVar, comp::Nothing, y1, y2, o2, cnt, adj, adj2)
    cnt += 1
    push!(y1, (t.i, t.i))
    return cnt
end

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::T,
        comp,
        y1::Tuple{V1, V2},
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {T <: SecondAdjointNodeVar, V1 <: AbstractVector, V2 <: AbstractVector}
    y, v = y1
    @inbounds y[t.i] += adj2 * v[t.i]
    return (cnt += 1)
end

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::T,
        comp,
        y1,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {T <: SecondAdjointNodeVar}
    @inbounds y1[o2 + comp(cnt += 1)] += adj2
    return cnt
end

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::T,
        comp,
        y1::V,
        y2::V,
        o2,
        cnt,
        adj,
        adj2,
    ) where {T <: SecondAdjointNodeVar, I <: Integer, V <: AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    if y1 === y2
        if t.i != 0
            @inbounds y1[ind] = pack_indices(t.i, t.i)
        end
    else
        if t.i != 0
            @inbounds y1[ind] = t.i
            @inbounds y2[ind] = t.i
        end
    end
    return cnt
end

@inline function hrpass(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::T,
        comp,
        y1::V,
        y2,
        o2,
        cnt,
        adj,
        adj2,
    ) where {T <: SecondAdjointNodeVar, I <: Tuple{Tuple{Int, Int}, Int}, V <: AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    @inbounds y1[ind] = ((t.i, t.i), ind)
    return cnt
end

@inline pack_indices(i, j) = (UInt64(i) << 32) | UInt64(j)
@inline unpack_row(v) = Int(v >> 32)
@inline unpack_col(v) = Int(v & 0xFFFFFFFF)

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1::V,
        y2::V,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNodeVar, T2 <: SecondAdjointNodeVar, I <: Integer, V <: AbstractVector{I}}
    i, j = t1.i, t2.i
    ind = o2 + comp(cnt += 1)

    if y1 === y2
        if i != 0 || j != 0
            @inbounds if i >= j
                y1[ind] = pack_indices(i, j)
            else
                y1[ind] = pack_indices(j, i)
            end
        end
    else
        if i != 0 || j != 0
            @inbounds if i >= j
                y1[ind] = i
                y2[ind] = j
            else
                y1[ind] = j
                y2[ind] = i
            end
        end
    end
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1::V,
        y2,
        o2,
        cnt,
        adj,
    ) where {
        T1 <: SecondAdjointNodeVar,
        T2 <: SecondAdjointNodeVar,
        I <: Tuple{Tuple{Int, Int}, Int},
        V <: AbstractVector{I},
    }
    i, j = t1.i, t2.i
    ind = o2 + comp(cnt += 1)
    @inbounds if i >= j
        y1[ind] = ((i, j), ind)
    else
        y1[ind] = ((j, i), ind)
    end
    return cnt
end

@inline function hrpass0(
        e,
        e_starts,
        e_cnts,
        e2,
        e2_starts,
        e2_cnts,
        isexp,
        t::T,
        comp,
        y1::V,
        y2::V,
        o2,
        cnt,
        adj,
        adj2,
    ) where {T <: SecondAdjointNodeExpr, I <: Integer, V <: AbstractVector{I}}
    (cnt_start2, e_start2) = e2_starts[t.i]
    len2 = e2_cnts[cnt_start2]
    cnt += 1
    for i in 1:len2
        ind = o2 + comp(cnt)
        val = e2[e_start2 + i - 1]
        r = unpack_row(val)
        c = unpack_col(val)
        if y1 === y2
            if r != 0 || c != 0
                @inbounds y1[ind] = pack_indices(r, c)
            end
        else
            if r != 0 || c != 0
                @inbounds y1[ind] = r
                @inbounds y2[ind] = c
            end
        end
        cnt += e2_cnts[cnt_start2 + i]
    end
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1::V,
        y2::V,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNodeExpr, T2 <: SecondAdjointNodeVar, I <: Integer, V <: AbstractVector{I}}
    (cnt_start, e_start) = e_starts[t1.i]
    len = e_cnts[cnt_start]
    j = t2.i
    cnt += 1
    for i in 1:len
        ind = o2 + comp(cnt)
        idx = e[e_start + i - 1]
        if y1 === y2
            if idx != 0 || j != 0
                @inbounds if idx >= j
                    y1[ind] = pack_indices(idx, j)
                else
                    y1[ind] = pack_indices(j, idx)
                end
            end
        else
            if idx != 0 || j != 0
                @inbounds if idx >= j
                    y1[ind] = idx
                    y2[ind] = j
                else
                    y1[ind] = j
                    y2[ind] = idx
                end
            end
        end
        cnt += e_cnts[cnt_start + i]
    end
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1::V,
        y2::V,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNodeVar, T2 <: SecondAdjointNodeExpr, I <: Integer, V <: AbstractVector{I}}
    i = t1.i
    (cnt_start, e_start) = e_starts[t2.i]
    len = e_cnts[cnt_start]
    cnt += 1
    for k in 1:len
        ind = o2 + comp(cnt)
        idx = e[e_start + k - 1]
        if y1 === y2
            if i != 0 || idx != 0
                @inbounds if i >= idx
                    y1[ind] = pack_indices(i, idx)
                else
                    y1[ind] = pack_indices(idx, i)
                end
            end
        else
            if i != 0 || idx != 0
                @inbounds if i >= idx
                    y1[ind] = i
                    y2[ind] = idx
                else
                    y1[ind] = idx
                    y2[ind] = i
                end
            end
        end
        cnt += e_cnts[cnt_start + k]
    end
    return cnt
end

@inline function hdrpass(
        e,
        e_starts,
        e_cnts,
        isexp,
        t1::T1,
        t2::T2,
        comp,
        y1::V,
        y2::V,
        o2,
        cnt,
        adj,
    ) where {T1 <: SecondAdjointNodeExpr, T2 <: SecondAdjointNodeExpr, I <: Integer, V <: AbstractVector{I}}
    (cnt_start1, e_start1) = e_starts[t1.i]
    len1 = e_cnts[cnt_start1]
    (cnt_start2, e_start2) = e_starts[t2.i]
    len2 = e_cnts[cnt_start2]

    cnt += 1
    for i in 1:len1
        idx1 = e[e_start1 + i - 1]
        for j in 1:len2
            idx2 = e[e_start2 + j - 1]
            ind = o2 + comp(cnt)
            if y1 === y2
                if idx1 != 0 || idx2 != 0
                    @inbounds if idx1 >= idx2
                        y1[ind] = pack_indices(idx1, idx2)
                    else
                        y1[ind] = pack_indices(idx2, idx1)
                    end
                end
            else
                if idx1 != 0 || idx2 != 0
                    @inbounds if idx1 >= idx2
                        y1[ind] = idx1
                        y2[ind] = idx2
                    else
                        y1[ind] = idx2
                        y2[ind] = idx1
                    end
                end
            end
            cnt += e_cnts[cnt_start2 + j]
        end
        cnt += e_cnts[cnt_start1 + i]
    end
    return cnt
end

"""
    shessian!(y1, y2, f, x, θ, e1, e1_starts, e1_cnts, e2, e2_starts, e2_cnts, adj1, adj2, isexp)

Performs sparse hessian evaluation

# Arguments:
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity)
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `θ`: parameter vector
- `e1`: expression Jacobian values
- `e1_starts`: expression Jacobian start indices
- `e1_cnts`: expression Jacobian counts
- `e2`: expression Hessian values
- `e2_starts`: expression Hessian start indices
- `e2_cnts`: expression Hessian counts
- `adj1`: initial first adjoint
- `adj2`: initial second adjoint
- `isexp`: expression indicator vector
"""
function shessian!(y1, y2, f, x, θ, e1, e1_starts, e1_cnts, e2, e2_starts, e2_cnts, adj1, adj2, isexp)
    return @simd for k in eachindex(f.itr)
        @inbounds shessian!(
            y1,
            y2,
            f.f,
            f.itr[k],
            x,
            θ,
            e1, e1_starts, e1_cnts,
            e2, e2_starts, e2_cnts,
            f.f.comp2,
            offset2(f, k),
            adj1,
            adj2,
            isexp,
        )
    end
end

function shessian!(y1, y2, f, x, θ, e1, e1_starts, e1_cnts, e2, e2_starts, e2_cnts, adj1s::V, adj2, isexp) where {V <: AbstractVector}
    return @simd for k in eachindex(f.itr)
        @inbounds shessian!(
            y1,
            y2,
            f.f,
            f.itr[k],
            x,
            θ,
            e1, e1_starts, e1_cnts,
            e2, e2_starts, e2_cnts,
            f.f.comp2,
            offset2(f, k),
            adj1s[offset0(f, k)],
            adj2,
            isexp,
        )
    end
end

function shessian!(y1, y2, f, p, x, θ, e1, e1_starts, e1_cnts, e2, e2_starts, e2_cnts, comp, o2, adj1, adj2, isexp)
    graph = f(p, SecondAdjointNodeSource(x, isexp), θ)
    return hrpass0(e1, e1_starts, e1_cnts, e2, e2_starts, e2_cnts, isexp, graph, comp, y1, y2, o2, 0, adj1, adj2)
end
