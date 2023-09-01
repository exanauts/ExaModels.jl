@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode1,T2<:SecondAdjointNode1}
    cnt = hdrpass(t1.inner, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y * t2.y)
    cnt
end
@inbounds function hdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode1,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
)  # despecialized
    cnt = hdrpass(t1.inner, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y * t2.y)
    cnt
end


@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNode1}
    cnt = hdrpass(t1, t2.inner, comp, y1, y2, o2, cnt, adj * t2.y)
    cnt
end
@inbounds function hdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointNode1,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
)  # despecialized
    cnt = hdrpass(t1, t2.inner, comp, y1, y2, o2, cnt, adj * t2.y)
    cnt
end


@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode1,T2<:SecondAdjointNodeVar}
    cnt = hdrpass(t1.inner, t2, comp, y1, y2, o2, cnt, adj * t1.y)
    cnt
end
@inbounds function hdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNodeVar,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
)  # despecialized
    cnt = hdrpass(t1.inner, t2, comp, y1, y2, o2, cnt, adj * t1.y)
    cnt
end


@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode2,T2<:SecondAdjointNode2}
    cnt = hdrpass(t1.inner1, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y1)
    cnt = hdrpass(t1.inner1, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y2)
    cnt = hdrpass(t1.inner2, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y1)
    cnt = hdrpass(t1.inner2, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y2)
    cnt
end
@inbounds function hdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode2,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1.inner1, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y1)
    cnt = hdrpass(t1.inner1, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y2)
    cnt = hdrpass(t1.inner2, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y1)
    cnt = hdrpass(t1.inner2, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y2)
    cnt
end


@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode1,T2<:SecondAdjointNode2}
    cnt = hdrpass(t1.inner, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y * t2.y1)
    cnt = hdrpass(t1.inner, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y * t2.y2)
    cnt
end
@inbounds function hdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode2,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1.inner, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y * t2.y1)
    cnt = hdrpass(t1.inner, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y * t2.y2)
    cnt
end

@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode2,T2<:SecondAdjointNode1}
    cnt = hdrpass(t1.inner1, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y)
    cnt = hdrpass(t1.inner2, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y)
    cnt
end
@inbounds function hdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode1,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1.inner1, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y)
    cnt = hdrpass(t1.inner2, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y)
    cnt
end

@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNode2}
    cnt = hdrpass(t1, t2.inner1, comp, y1, y2, o2, cnt, adj * t2.y1)
    cnt = hdrpass(t1, t2.inner2, comp, y1, y2, o2, cnt, adj * t2.y2)
    cnt
end
@inbounds function hdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointNode2,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1, t2.inner1, comp, y1, y2, o2, cnt, adj * t2.y1)
    cnt = hdrpass(t1, t2.inner2, comp, y1, y2, o2, cnt, adj * t2.y2)
    cnt
end

@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNode2,T2<:SecondAdjointNodeVar}
    cnt = hdrpass(t1.inner1, t2, comp, y1, y2, o2, cnt, adj * t1.y1)
    cnt = hdrpass(t1.inner2, t2, comp, y1, y2, o2, cnt, adj * t1.y2)
    cnt
end
@inbounds function hdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNodeVar,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
) # despecialized
    cnt = hdrpass(t1.inner1, t2, comp, y1, y2, o2, cnt, adj * t1.y1)
    cnt = hdrpass(t1.inner2, t2, comp, y1, y2, o2, cnt, adj * t1.y2)
    cnt
end


@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNodeVar}
    i, j = t1.i, t2.i
    if i == j
        y1[o2+comp(cnt += 1)] += 2 * adj
    else
        y1[o2+comp(cnt += 1)] += adj
    end
    cnt
end


@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1::Tuple{V1,V2},
    y2,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNodeVar,V1 <: AbstractVector,V2 <: AbstractVector}
    i, j = t1.i, t2.i
    y, v = y1
    if i == j
        y[i] += 2 * adj * v[i]
    else
        y[i] += adj * v[j]
        y[j] += adj * v[i]
    end
    return (cnt += 1)
end

@inbounds @inline function hrpass(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1}
    cnt = hrpass(t.inner, comp, y1, y2, o2, cnt, adj * t.y, adj2 * (t.y)^2 + adj * t.h)
    cnt
end
@inbounds @inline function hrpass(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode2}
    adj2y1y2 = adj2 * t.y1 * t.y2
    adjh12 = adj * t.h12
    cnt = hrpass(t.inner1, comp, y1, y2, o2, cnt, adj * t.y1, adj2 * (t.y1)^2 + adj * t.h11)
    cnt = hrpass(t.inner2, comp, y1, y2, o2, cnt, adj * t.y2, adj2 * (t.y2)^2 + adj * t.h22)
    cnt = hdrpass(t.inner1, t.inner2, comp, y1, y2, o2, cnt, adj2y1y2 + adjh12)
    cnt
end

@inbounds @inline hrpass0(args...) = hrpass(args...)


@inbounds @inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {N<:Union{FirstFixed{typeof(*)},SecondFixed{typeof(*)}},D<:SecondAdjointNode1{N}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, adj * t.y, adj2 * (t.y)^2)
    cnt
end
@inbounds @inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {N<:Union{FirstFixed{typeof(+)},SecondFixed{typeof(+)}},D<:SecondAdjointNode1{N}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end
@inbounds @inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1{FirstFixed{typeof(-)}}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, -adj, adj2)
    cnt
end
@inbounds @inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1{SecondFixed{typeof(-)}}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end

@inbounds @inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1{typeof(+)}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end
@inbounds @inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode1{typeof(-)}}
    cnt = hrpass0(t.inner, comp, y1, y2, o2, cnt, -adj, adj2)
    cnt
end

@inbounds @inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode2{typeof(+)}}
    cnt = hrpass0(t.inner1, comp, y1, y2, o2, cnt, adj, adj2)
    cnt = hrpass0(t.inner2, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end

@inbounds @inline function hrpass0(
    t::D,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {D<:SecondAdjointNode2{typeof(-)}}
    cnt = hrpass0(t.inner1, comp, y1, y2, o2, cnt, adj, adj2)
    cnt = hrpass0(t.inner2, comp, y1, y2, o2, cnt, -adj, adj2)
    cnt
end
@inbounds @inline function hrpass0(
    t::T,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar}
    cnt
end
@inbounds @inline function hrpass0(
    t::T,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar}
    cnt
end


@inbounds function hdrpass(
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
    cnt
end
@inbounds function hrpass(
    t::SecondAdjointNodeVar,
    comp::Nothing,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
)
    cnt += 1
    push!(y1, (t.i, t.i))
    cnt
end

@inbounds @inline function hrpass(
    t::T,
    comp,
    y1::Tuple{V1,V2},
    y2,
    o2,
    cnt,
    adj,
    adj2,
    ) where {T<:SecondAdjointNodeVar,V1 <: AbstractVector,V2 <: AbstractVector}
    y, v = y1
    y[t.i] += adj2 * v[t.i]
    return (cnt += 1)
end
@inbounds @inline function hrpass(
    t::T,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar}
    y1[o2+comp(cnt += 1)] += adj2
    cnt
end
@inbounds @inline function hrpass(
    t::T,
    comp,
    y1::V,
    y2::V,
    o2,
    cnt,
    adj,
    adj2,
) where {T<:SecondAdjointNodeVar,I<:Integer,V<:AbstractVector{I}}
    ind = o2 + comp(cnt += 1)
    y1[ind] = t.i
    y2[ind] = t.i
    cnt
end
@inbounds @inline function hdrpass(
    t1::T1,
    t2::T2,
    comp,
    y1::V,
    y2::V,
    o2,
    cnt,
    adj,
) where {T1<:SecondAdjointNodeVar,T2<:SecondAdjointNodeVar,I<:Integer,V<:AbstractVector{I}}
    i, j = t1.i, t2.i
    ind = o2 + comp(cnt += 1)
    if i >= j
        y1[ind] = i
        y2[ind] = j
    else
        y1[ind] = j
        y2[ind] = i
    end
    cnt
end

function shessian!(y1, y2, f, x, adj1, adj2)
    @simd for k in eachindex(f.itr)
        hrpass0(
            f.f.f(f.itr[k], SecondAdjointNodeSource(x)),
            f.f.comp2,
            y1,
            y2,
            offset2(f, k),
            0,
            adj1,
            adj2,
        )
    end
end

function shessian!(y1, y2, f, x, adj1s::V, adj2) where {V<:AbstractVector}
    @simd for k in eachindex(f.itr)
        hrpass0(
            f.f.f(f.itr[k], SecondAdjointNodeSource(x)),
            f.f.comp2,
            y1,
            y2,
            offset2(f, k),
            0,
            adj1s[offset0(f, k)],
            adj2,
        )
    end
end
