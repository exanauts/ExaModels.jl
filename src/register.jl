macro register_univariate(f, df, ddf)
    return esc(
        quote
            @inline $f(n::N) where {N<:ExaModels.AbstractNode} = ExaModels.Node1($f, n)
            @inline $f(d::D) where {D<:ExaModels.AbstractAdjointNode} =
                ExaModels.AdjointNode1($f, $f(d.x), $df(d.x), d)
            @inline $f(t::T) where {T<:ExaModels.AbstractSecondAdjointNode} =
                ExaModels.SecondAdjointNode1($f, $f(t.x), $df(t.x), $ddf(t.x), t)

            @inbounds @inline (n::ExaModels.Node1{typeof($f),I})(i, x) where {I} =
                $f(n.inner(i, x))
        end,
    )
end

macro register_bivariate(f, df1, df2, ddf11, ddf12, ddf22)
    return esc(
        quote
            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {D1<:ExaModels.AbstractNode,D2<:ExaModels.AbstractNode}
                ExaModels.Node2($f, d1, d2)
            end
            @inline function $f(d1::D1, d2::D2) where {D1<:ExaModels.AbstractNode,D2<:Real}
                ExaModels.Node2($f, d1, d2)
            end
            @inline function $f(d1::D1, d2::D2) where {D1<:Real,D2<:ExaModels.AbstractNode}
                ExaModels.Node2($f, d1, d2)
            end

            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {D1<:ExaModels.AbstractAdjointNode,D2<:ExaModels.AbstractAdjointNode}

                x1 = d1.x
                x2 = d2.x
                ExaModels.AdjointNode2($f, $f(x1, x2), $df1(x1, x2), $df2(x1, x2), d1, d2)
            end
            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {D1<:ExaModels.AbstractAdjointNode,D2<:Real}

                x1 = d1.x
                x2 = d2

                ExaModels.AdjointNode1($f, $f(x1, x2), $df1(x1, x2), d1)
            end
            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {D1<:Real,D2<:ExaModels.AbstractAdjointNode}

                x1 = d1
                x2 = d2.x
                ExaModels.AdjointNode1($f, $f(x1, x2), $df2(x1, x2), d2)
            end

            @inline function $f(
                t1::T1,
                t2::T2,
            ) where {
                T1<:ExaModels.AbstractSecondAdjointNode,
                T2<:ExaModels.AbstractSecondAdjointNode,
            }

                x1 = t1.x
                x2 = t2.x
                ExaModels.SecondAdjointNode2(
                    $f,
                    $f(x1, x2),
                    $df1(x1, x2),
                    $df2(x1, x2),
                    $ddf11(x1, x2),
                    $ddf12(x1, x2),
                    $ddf22(x1, x2),
                    t1,
                    t2,
                )
            end
            @inline function $f(
                t1::T1,
                t2::T2,
            ) where {T1<:ExaModels.AbstractSecondAdjointNode,T2<:Real}

                x1 = t1.x
                x2 = t2
                ExaModels.SecondAdjointNode1(
                    ExaModels.SecondFixed($f),
                    $f(x1, x2),
                    $df1(x1, x2),
                    $ddf11(x1, x2),
                    t1,
                )
            end
            @inline function $f(
                t1::T1,
                t2::T2,
            ) where {T1<:Real,T2<:ExaModels.AbstractSecondAdjointNode}

                x1 = t1
                x2 = t2.x
                ExaModels.SecondAdjointNode1(
                    ExaModels.FirstFixed($f),
                    $f(x1, x2),
                    $df2(x1, x2),
                    $ddf22(x2, x1),
                    t2,
                )
            end

            @inbounds @inline (n::ExaModels.Node2{typeof($f),I1,I2})(i, x) where {I1,I2} =
                $f(n.inner1(i, x), n.inner2(i, x))
            @inbounds @inline (n::ExaModels.Node2{typeof($f),I1,I2})(
                i,
                x,
            ) where {I1<:Real,I2} = $f(n.inner1, n.inner2(i, x))
            @inbounds @inline (n::ExaModels.Node2{typeof($f),I1,I2})(
                i,
                x,
            ) where {I1,I2<:Real} = $f(n.inner1(i, x), n.inner2)
        end,
    )
end
