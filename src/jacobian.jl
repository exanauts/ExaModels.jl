@inbounds @inline function jrpass(d::D, comp, i, y1,y2, o1, cnt, adj) where D <: Dual1
    cnt = jrpass(d.inner, comp, i, y1,y2, o1, cnt, adj * d.y)
    return cnt
end
@inbounds @inline function jrpass(d::D, comp, i, y1,y2, o1, cnt, adj) where D <: Dual2
    cnt = jrpass(d.inner1, comp, i, y1,y2, o1, cnt, adj * d.y1)
    cnt = jrpass(d.inner2, comp, i, y1,y2, o1, cnt, adj * d.y2)
    return cnt
end
@inbounds @inline function jrpass(d::D, comp, i, y1,y2, o1, cnt, adj) where D <: DualVar
    y1[o1 + comp(cnt += 1)] += adj
    return cnt
end
@inbounds @inline function jrpass(d::D, comp, i, y1::V, y2::V, o1, cnt, adj) where {D <: DualVar, I <: Integer, V <: AbstractVector{I}}
    ind = o1 + comp(cnt += 1)
    y1[ind] = i
    y2[ind] = d.i
    return cnt
end


function sjacobian!(y1,y2,f,x,adj)
    @simd for i in eachindex(f.itr)
        jrpass(f.f.f(f.itr[i],DualSource(x)), f.f.comp1, offset0(f,i), y1, y2, offset1(f,i), 0, adj)
    end
end
