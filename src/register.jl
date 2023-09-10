"""
    @register_univariate(f, df, ddf)

Register a univariate function `f` to `ExaModels`, so that it can be used within objective and constraint expressions

# Arguments:
- `f`: function
- `df`: derivative function
- `ddf`: second-order derivative funciton

## Example
```jldoctest
julia> using ExaModels

julia> relu3(x) = x > 0 ? x^3 : zero(x)
relu3 (generic function with 1 method)

julia> drelu3(x) = x > 0 ? 3*x^2 : zero(x)
drelu3 (generic function with 1 method)

julia> ddrelu3(x) = x > 0 ? 6*x : zero(x)
ddrelu3 (generic function with 1 method)

julia> @register_univariate(relu3, drelu3, ddrelu3)
```
"""
macro register_univariate(f, df, ddf)
    return esc(
        quote
            if !hasmethod($f, Tuple{ExaModels.AbstractNode})
                @inline $f(n::N) where {N<:ExaModels.AbstractNode} = ExaModels.Node1($f, n)
            end

            @inline $f(d::D) where {D<:ExaModels.AbstractAdjointNode} =
                ExaModels.AdjointNode1($f, $f(d.x), $df(d.x), d)
            @inline $f(t::T) where {T<:ExaModels.AbstractSecondAdjointNode} =
                ExaModels.SecondAdjointNode1($f, $f(t.x), $df(t.x), $ddf(t.x), t)

            @inline (n::ExaModels.Node1{typeof($f),I})(i, x) where {I} =
                $f(n.inner(i, x))
        end,
    )
end

"""
    register_bivariate(f, df1, df2, ddf11, ddf12, ddf22)

Register a bivariate function `f` to `ExaModels`, so that it can be used within objective and constraint expressions

# Arguments:
- `f`: function
- `df1`: derivative function (w.r.t. first argument)
- `df2`: derivative function (w.r.t. second argument)
- `ddf11`: second-order derivative funciton (w.r.t. first argument)
- `ddf12`: second-order derivative funciton (w.r.t. first and second argument)
- `ddf22`: second-order derivative funciton (w.r.t. second argument)

## Example
```jldoctest
julia> using ExaModels

julia> relu23(x) = (x > 0 || y > 0) ? (x + y)^3 : zero(x)
relu23 (generic function with 1 method)

julia> drelu231(x) = (x > 0 || y > 0) ? 3 * (x + y)^2 : zero(x)
drelu231 (generic function with 1 method)

julia> drelu232(x) = (x > 0 || y > 0) ? 3 * (x + y)^2  : zero(x)
drelu232 (generic function with 1 method)

julia> ddrelu2311(x) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
ddrelu2311 (generic function with 1 method)

julia> ddrelu2312(x) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
ddrelu2312 (generic function with 1 method)

julia> ddrelu2322(x) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
ddrelu2322 (generic function with 1 method)

julia> @register_bivariate(relu23, drelu231, drelu232, ddrelu2311, ddrelu2312, ddrelu2322)
```
"""
macro register_bivariate(f, df1, df2, ddf11, ddf12, ddf22)
    return esc(
        quote
            if !hasmethod($f, Tuple{ExaModels.AbstractNode,ExaModels.AbstractNode})
                @inline function $f(
                    d1::D1,
                    d2::D2,
                ) where {D1<:ExaModels.AbstractNode,D2<:ExaModels.AbstractNode}
                    ExaModels.Node2($f, d1, d2)
                end
            end

            if !hasmethod($f, Tuple{ExaModels.AbstractNode,Real})
                @inline function $f(
                    d1::D1,
                    d2::D2,
                ) where {D1<:ExaModels.AbstractNode,D2<:Real}
                    ExaModels.Node2($f, d1, d2)
                end
            end

            if !hasmethod($f, Tuple{Real,ExaModels.AbstractNode})
                @inline function $f(
                    d1::D1,
                    d2::D2,
                ) where {D1<:Real,D2<:ExaModels.AbstractNode}
                    ExaModels.Node2($f, d1, d2)
                end
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
                    $ddf22(x1, x2),
                    t2,
                )
            end

            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(i, x) where {I1,I2} =
                $f(n.inner1(i, x), n.inner2(i, x))
            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(
                i,
                x,
            ) where {I1<:Real,I2} = $f(n.inner1, n.inner2(i, x))
            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(
                i,
                x,
            ) where {I1,I2<:Real} = $f(n.inner1(i, x), n.inner2)
        end,
    )
end
