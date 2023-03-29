macro register_univariate(f,df,ddf)
    return esc(
        quote
            @inline $f(n::N) where {N <: SIMDiff.AbstractNode} =
                SIMDiff.Node1($f,n)
            @inline $f(d::D) where {D <: SIMDiff.AbstractDual} =
                SIMDiff.Dual1($f,$f(d.x), $df(d.x), d)
            @inline $f(t::T) where {T <: SIMDiff.AbstractTriple} =
                SIMDiff.Triple1($f,$f(t.x), $df(t.x), $ddf(t.x), t)
            
            @inbounds @inline (n::SIMDiff.Node1{typeof($f),I})(i,x) where I = $f(n.inner(i,x))
        end
    )
end

macro register_bivariate(f,df1,df2,ddf11,ddf12,ddf22)
    return esc(
        quote
            @inline function $f(d1::D1, d2::D2) where {
                D1 <: SIMDiff.AbstractNode,
                D2 <: SIMDiff.AbstractNode
                }
                SIMDiff.Node2(
                    $f,d1,d2
                )
            end
            @inline function $f(d1::D1, d2::D2) where {
                D1 <: SIMDiff.AbstractNode,
                D2 <: Real
                }
                SIMDiff.Node2(
                    $f,d1,d2
                )
            end
            @inline function $f(d1::D1, d2::D2) where {
                D1 <: Real,
                D2 <: SIMDiff.AbstractNode
                }
                SIMDiff.Node2(
                    $f,d1,d2
                )
            end

            @inline function $f(d1::D1, d2::D2) where {
                D1 <: SIMDiff.AbstractDual,
                D2 <: SIMDiff.AbstractDual
                }
                
                x1 = d1.x
                x2 = d2.x
                SIMDiff.Dual2(
                    $f,
                    $f(x1,x2),
                    $df1(x1,x2),
                    $df2(x1,x2),
                    d1,
                    d2
                )
            end
            @inline function $f(d1::D1, d2::D2) where {
                D1 <: SIMDiff.AbstractDual,
                D2 <: Real
                }
                
                x1 = d1.x
                x2 = d2

                SIMDiff.Dual1(
                    $f,
                    $f(x1,x2),
                    $df1(x1,x2),
                    d1,
                )
            end
            @inline function $f(d1::D1, d2::D2) where {
                D1 <: Real,
                D2 <: SIMDiff.AbstractDual
                }
                
                x1 = d1
                x2 = d2.x
                SIMDiff.Dual1(
                    $f,
                    $f(x1,x2),
                    $df2(x1,x2),
                    d2
                )
            end

            @inline function $f(t1::T1, t2::T2) where {
                T1 <: SIMDiff.AbstractTriple,
                T2 <: SIMDiff.AbstractTriple
                }
                
                x1 = t1.x
                x2 = t2.x
                SIMDiff.Triple2(
                    $f,
                    $f(x1,x2),
                    $df1(x1,x2),
                    $df2(x1,x2),
                    $ddf11(x1,x2),
                    $ddf12(x1,x2),
                    $ddf22(x1,x2),
                    t1,
                    t2
                )
            end
            @inline function $f(t1::T1, t2::T2) where {
                T1 <: SIMDiff.AbstractTriple,
                T2 <: Real
                }
                
                x1 = t1.x
                x2 = t2
                SIMDiff.Triple1(
                    SIMDiff.SecondFixed($f),
                    $f(x1,x2),
                    $df1(x1,x2),
                    $ddf11(x1,x2),
                    t1,
                )
            end
            @inline function $f(t1::T1, t2::T2) where {
                T1 <: Real,
                T2 <: SIMDiff.AbstractTriple
                }
                
                x1 = t1
                x2 = t2.x
                SIMDiff.Triple1(
                    SIMDiff.FirstFixed($f),
                    $f(x1,x2),
                    $df2(x1,x2),
                    $ddf22(x2,x1),
                    t2
                )
            end

            @inbounds @inline (n::SIMDiff.Node2{typeof($f),I1,I2})(i,x) where {I1, I2} = $f(n.inner1(i,x), n.inner2(i,x))
            @inbounds @inline (n::SIMDiff.Node2{typeof($f),I1,I2})(i,x) where {I1 <: Real, I2} = $f(n.inner1, n.inner2(i,x))
            @inbounds @inline (n::SIMDiff.Node2{typeof($f),I1,I2})(i,x) where {I1, I2 <: Real} = $f(n.inner1(i,x), n.inner2)
        end
    )
end
