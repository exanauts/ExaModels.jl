"""
    @register_multivariate(f, N, grad!, hess!)

Register an N-argument scalar function `f` to `ExaModels`, so that it can be used within
objective and constraint expressions. The derivatives are supplied by the user via
`grad!` and `hess!` callbacks.

# Arguments:
- `f`: function `f(x1, ..., xN) -> scalar`
- `N`: number of arguments (positive integer literal)
- `grad!`: gradient callback `grad!(g::AbstractVector, x1, ..., xN)` — fills `g[1:N]`
  with partial derivatives `∂f/∂xᵢ`
- `hess!`: Hessian callback `hess!(H::AbstractVector, x1, ..., xN)` — fills `H[1:N*(N+1)÷2]`
  with upper-triangular entries packed row-major: `H[k]` corresponds to `∂²f/∂xᵢ∂xⱼ`
  for `(i,j)` with `i ≤ j`, ordered as `(1,1),(1,2),...,(1,N),(2,2),...,(N,N)`.

## Example
```julia
using ExaModels

# f(x,y,z) = x^2 + 2*y^2 + 3*z^2
f3(x,y,z) = x^2 + 2*y^2 + 3*z^2

function grad_f3!(g, x, y, z)
    g[1] = 2*x
    g[2] = 4*y
    g[3] = 6*z
end

function hess_f3!(H, x, y, z)
    # upper-triangular row-major: (1,1),(1,2),(1,3),(2,2),(2,3),(3,3)
    H[1] = 2.0; H[2] = 0.0; H[3] = 0.0
    H[4] = 4.0; H[5] = 0.0
    H[6] = 6.0
end

@register_multivariate(f3, 3, grad_f3!, hess_f3!)
```
"""
macro register_multivariate(f, N_expr, grad!, hess!)
    N = N_expr isa Integer ? N_expr : N_expr  # handled at parse time
    # Build (d1, d2, ..., dN) argument symbols
    arg_syms = [Symbol("_d", k) for k in 1:N]
    arg_sym_types_node    = [:($(Symbol("D",k))<:ExaModels.AbstractNode) for k in 1:N]
    arg_sym_types_adjoint = [:($(Symbol("D",k))<:ExaModels.AbstractAdjointNode) for k in 1:N]
    arg_sym_types_sadjoint = [:($(Symbol("D",k))<:ExaModels.AbstractSecondAdjointNode) for k in 1:N]
    arg_decls_node    = [:($(arg_syms[k])::$(Symbol("D",k))) for k in 1:N]
    arg_decls_adjoint = [:($(arg_syms[k])::$(Symbol("D",k))) for k in 1:N]
    arg_decls_sadjoint = [:($(arg_syms[k])::$(Symbol("D",k))) for k in 1:N]
    where_node    = Expr(:where, :(_dummy), arg_sym_types_node...)
    where_adjoint = Expr(:where, :(_dummy), arg_sym_types_adjoint...)
    where_sadjoint = Expr(:where, :(_dummy), arg_sym_types_sadjoint...)

    # x_vals = (d1.x, d2.x, ..., dN.x)
    xvals = [:($(arg_syms[k]).x) for k in 1:N]
    # inner tuple of children for NodeN
    args_tuple = Expr(:tuple, arg_syms...)
    # xvals tuple for grad! / hess! calls
    xvals_tuple = Expr(:tuple, xvals...)

    # Build g-tuple from buffer after grad! call
    g_accesses = [:(g_buf[$k]) for k in 1:N]
    g_tuple = Expr(:tuple, g_accesses...)

    # Build h-tuple from buffer after hess! call
    Khess = N * (N + 1) ÷ 2
    h_accesses = [:(h_buf[$k]) for k in 1:Khess]
    h_tuple = Expr(:tuple, h_accesses...)

    # eval overload: (n::NodeN{typeof(f),...})(i, x, θ)
    eval_inner_calls = [:(n.args[$k](i, x, θ)) for k in 1:N]
    eval_call = :($f($(eval_inner_calls...)))

    return esc(
        quote
            # 1. AbstractNode overload — builds symbolic graph node
            if !hasmethod($f, Tuple{$(fill(:(ExaModels.AbstractNode), N)...)})
                @inline function $f($(arg_decls_node...)) where {$(arg_sym_types_node...)}
                    ExaModels.NodeN($f, ($(arg_syms...),))
                end
            end

            # 2. AbstractAdjointNode overload — first-order AD (calls grad!)
            @inline function $f($(arg_decls_adjoint...)) where {$(arg_sym_types_adjoint...)}
                _xs = ($([:($(arg_syms[k]).x) for k in 1:N]...),)
                g_buf = Vector{typeof(_xs[1])}($N)
                $grad!( g_buf, _xs...)
                ExaModels.AdjointNodeN(
                    $f,
                    $f(_xs...),
                    ($([:( g_buf[$k]) for k in 1:N]...),),
                    ($(arg_syms...),),
                )
            end

            # 3. AbstractSecondAdjointNode overload — second-order AD (calls grad! and hess!)
            @inline function $f($(arg_decls_sadjoint...)) where {$(arg_sym_types_sadjoint...)}
                _xs = ($([:($(arg_syms[k]).x) for k in 1:N]...),)
                g_buf = Vector{typeof(_xs[1])}($N)
                h_buf = Vector{typeof(_xs[1])}($(N * (N + 1) ÷ 2))
                $grad!( g_buf, _xs...)
                $hess!( h_buf, _xs...)
                ExaModels.SecondAdjointNodeN(
                    $f,
                    $f(_xs...),
                    ($([:( g_buf[$k]) for k in 1:N]...),),
                    ($([:( h_buf[$k]) for k in 1:N*(N+1)÷2]...),),
                    ($(arg_syms...),),
                )
            end

            # 4. Evaluation overload for NodeN
            @inline (n::ExaModels.NodeN{typeof($f),$N})(i, x, θ) =
                $f($([:(n.args[$k](i, x, θ)) for k in 1:N]...))
        end,
    )
end

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

            @inline (n::ExaModels.Node1{typeof($f),I})(i, x, θ) where {I} =
                $f(n.inner(i, x, θ))
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

julia> relu23(x,y) = (x > 0 || y > 0) ? (x + y)^3 : zero(x)
relu23 (generic function with 1 method)

julia> drelu231(x,y) = (x > 0 || y > 0) ? 3 * (x + y)^2 : zero(x)
drelu231 (generic function with 1 method)

julia> drelu232(x,y) = (x > 0 || y > 0) ? 3 * (x + y)^2  : zero(x)
drelu232 (generic function with 1 method)

julia> ddrelu2311(x,y) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
ddrelu2311 (generic function with 1 method)

julia> ddrelu2312(x,y) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
ddrelu2312 (generic function with 1 method)

julia> ddrelu2322(x,y) = (x > 0 || y > 0) ? 6 * (x + y) : zero(x)
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
            ) where {
                D1<:ExaModels.AbstractAdjointNode,
                D2<:Union{Real,ExaModels.ParameterNode},
            }

                x1 = d1.x
                x2 = d2

                ExaModels.AdjointNode1($f, $f(x1, x2), $df1(x1, x2), d1)
            end
            @inline function $f(
                d1::D1,
                d2::D2,
            ) where {
                D1<:Union{Real,ExaModels.ParameterNode},
                D2<:ExaModels.AbstractAdjointNode,
            }

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
            ) where {
                T1<:ExaModels.AbstractSecondAdjointNode,
                T2<:Union{Real,ExaModels.ParameterNode},
            }

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
            ) where {
                T1<:Union{Real,ExaModels.ParameterNode},
                T2<:ExaModels.AbstractSecondAdjointNode,
            }

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

            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ) where {I1,I2} =
                $f(n.inner1(i, x, θ), n.inner2(i, x, θ))
            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ) where {I1<:Real,I2} =
                $f(n.inner1, n.inner2(i, x, θ))
            @inline (n::ExaModels.Node2{typeof($f),I1,I2})(i, x, θ) where {I1,I2<:Real} =
                $f(n.inner1(i, x, θ), n.inner2)
        end,
    )
end
