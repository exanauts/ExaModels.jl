@inbounds @inline function drpass(d::D, y, adj) where D <: Dual1
    offset = drpass(d.inner, y, adj * d.y)
    nothing
end
@inbounds @inline function drpass(d::D, y, adj) where D <: Dual2
    offset = drpass(d.inner1, y, adj * d.y1)
    offset = drpass(d.inner2, y, adj * d.y2)
    nothing
end
@inbounds @inline function drpass(d::D, y, adj) where D <: DualVar
    y[d.i] += adj
    nothing
end
@inbounds @inline function drpass(f::F, x, y, adj) where F <: Func
end
function gradient!(y,f,x, adj)
    @simd for k in eachindex(f.itr)
        drpass(f.f.f(f.itr[k],DualSource(x)), y, adj)
    end
    return y
end


@inbounds @inline function grpass(d::D, comp, y, o1, cnt, adj) where D <: Dual1
    cnt = grpass(d.inner, comp, y, o1, cnt, adj * d.y)
    return cnt
end
@inbounds @inline function grpass(d::D, comp, y, o1, cnt, adj) where D <: Dual2
    cnt = grpass(d.inner1, comp, y, o1, cnt, adj * d.y1)
    cnt = grpass(d.inner2, comp, y, o1, cnt, adj * d.y2)
    return cnt
end
@inbounds @inline function grpass(d::D, comp, y, o1, cnt, adj) where D <: DualVar
    y[o1 + comp(cnt += 1)] += adj
    return cnt
end

@inbounds @inline function grpass(d::DualVar, comp::Nothing, y, o1, cnt, adj) # despecialization
    push!(y,d.i)
    return (cnt += 1)
end
@inbounds @inline function grpass(d::D, comp, y::V, o1, cnt, adj) where {D <: DualVar, V <: AbstractVector{Tuple{Int,Int}}}
    ind = o1 + comp(cnt += 1)
    y[ind] = (d.i,ind)
    return cnt
end

function sgradient!(y,f,x, adj)
    @simd for k in eachindex(f.itr)
        grpass(f.f.f(f.itr[k], DualSource(x)), f.itr.comp1, y, offset1(f,k), 0, adj)
    end
    return y
end
