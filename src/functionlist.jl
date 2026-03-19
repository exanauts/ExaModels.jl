@inline _mone(x) = -one(x)
@inline _one(x1, x2) = one(x1)
@inline _zero(x1, x2) = zero(x1)
@inline _mone(x1, x2) = -one(x1)
@inline _x1(x1, x2) = x1
@inline _x2(x1, x2) = x2
@inline _and(x::Bool, y::Bool) = x && y
@inline _or(x::Bool, y::Bool) = x || y
@inline _and(x, y::Bool) = x == 1 && y
@inline _or(x, y::Bool) = x == 1 || y
@inline _and(x::Bool, y) = x && y == 1
@inline _or(x::Bool, y) = x || y == 1
@inline _and(x, y) = x == 1 && y == 1
@inline _or(x, y) = x == 1 || y == 1

# Type-generic constant helpers (avoid Float64 literals for Float32 compatibility)
@inline _clog2(x) = oftype(x, log(2))
@inline _clog10(x) = oftype(x, log(10))
@inline _cpi(x) = oftype(x, π)
@inline _cd2r(x) = oftype(x, π / 180)
@inline _cr2d(x) = oftype(x, 180 / π)

# Runtime registration helpers — use Base.$fname pattern (symbols) to avoid
# syntax errors with operator function names like Base.:+.

# _needs_overload: true when no ExaModels-specific method exists yet.
# Plain `hasmethod` is too conservative — it returns true for
# untyped Base generics like max(x,y) = ifelse(isless(x,y),y,x),
# which prevents the Node2 overload from being added.  Check the
# module of the matched method: only skip if ExaModels already owns it
# (e.g. the identity-element specializations in specialization.jl).
function _needs_overload(f, types)
    hasmethod(f, types) || return true
    return which(f, types).module !== @__MODULE__
end

function _register_univ(fname::Symbol, df, ddf)
    if _needs_overload(getfield(Base, fname), Tuple{AbstractNode})
        @eval @inline Base.$fname(n::N) where {N <: AbstractNode} = Node1(Base.$fname, n)
    end
    @eval @inline Base.$fname(d::D) where {D <: AbstractAdjointNode} =
        AdjointNode1(Base.$fname, Base.$fname(d.x), ($df)(d.x), d)
    @eval @inline Base.$fname(t::T) where {T <: AbstractSecondAdjointNode} =
        SecondAdjointNode1(Base.$fname, Base.$fname(t.x), ($df)(t.x), ($ddf)(t.x), t)
    return @eval @inline (n::Node1{typeof(Base.$fname), I})(i, x, θ) where {I} =
        Base.$fname(n.inner(i, x, θ))
end

function _register_biv(fname::Symbol, df1, df2, ddf11, ddf12, ddf22)
    f = getfield(Base, fname)
    if _needs_overload(f, Tuple{AbstractNode, AbstractNode})
        @eval @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {D1 <: AbstractNode, D2 <: AbstractNode}
            return Node2(Base.$fname, d1, d2)
        end
    end
    if _needs_overload(f, Tuple{AbstractNode, Real})
        @eval @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {D1 <: AbstractNode, D2 <: Real}
            return Node2(Base.$fname, d1, d2)
        end
    end
    if _needs_overload(f, Tuple{Real, AbstractNode})
        @eval @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {D1 <: Real, D2 <: AbstractNode}
            return Node2(Base.$fname, d1, d2)
        end
    end
    if _needs_overload(f, Tuple{AbstractNode, Val{Real}})
        @eval @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {D1 <: AbstractNode, V, D2 <: Val{V}}
            return Node2(Base.$fname, d1, d2)
        end
    end
    if _needs_overload(f, Tuple{Val{Real}, AbstractNode})
        @eval @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {V, D1 <: Val{V}, D2 <: AbstractNode}
            return Node2(Base.$fname, d1, d2)
        end
    end
    return @eval begin
        @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {D1 <: AbstractAdjointNode, D2 <: AbstractAdjointNode}
            x1 = d1.x
            x2 = d2.x
            return AdjointNode2(Base.$fname, Base.$fname(x1, x2), ($df1)(x1, x2), ($df2)(x1, x2), d1, d2)
        end
        @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {D1 <: AbstractAdjointNode, D2 <: Union{Real, ParameterNode}}
            x1 = d1.x
            x2 = d2
            return AdjointNode1(Base.$fname, Base.$fname(x1, x2), ($df1)(x1, x2), d1)
        end
        @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {D1 <: Union{Real, ParameterNode}, D2 <: AbstractAdjointNode}
            x1 = d1
            x2 = d2.x
            return AdjointNode1(Base.$fname, Base.$fname(x1, x2), ($df2)(x1, x2), d2)
        end
        @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {V, D1 <: AbstractAdjointNode, D2 <: Val{V}}
            x1 = d1.x
            x2 = V
            return AdjointNode1(Base.$fname, Base.$fname(x1, x2), ($df1)(x1, x2), d1)
        end
        @inline function Base.$fname(
                d1::D1,
                d2::D2,
            ) where {V, D1 <: Val{V}, D2 <: AbstractAdjointNode}
            x1 = V
            x2 = d2.x
            return AdjointNode1(Base.$fname, Base.$fname(x1, x2), ($df2)(x1, x2), d2)
        end
        @inline function Base.$fname(
                t1::T1,
                t2::T2,
            ) where {T1 <: AbstractSecondAdjointNode, T2 <: AbstractSecondAdjointNode}
            x1 = t1.x
            x2 = t2.x
            return SecondAdjointNode2(
                Base.$fname,
                Base.$fname(x1, x2),
                ($df1)(x1, x2),
                ($df2)(x1, x2),
                ($ddf11)(x1, x2),
                ($ddf12)(x1, x2),
                ($ddf22)(x1, x2),
                t1,
                t2,
            )
        end
        @inline function Base.$fname(
                t1::T1,
                t2::T2,
            ) where {T1 <: AbstractSecondAdjointNode, T2 <: Union{Real, ParameterNode}}
            x1 = t1.x
            x2 = t2
            return SecondAdjointNode1(
                SecondFixed(Base.$fname),
                Base.$fname(x1, x2),
                ($df1)(x1, x2),
                ($ddf11)(x1, x2),
                t1,
            )
        end
        @inline function Base.$fname(
                t1::T1,
                t2::T2,
            ) where {T1 <: Union{Real, ParameterNode}, T2 <: AbstractSecondAdjointNode}
            x1 = t1
            x2 = t2.x
            return SecondAdjointNode1(
                FirstFixed(Base.$fname),
                Base.$fname(x1, x2),
                ($df2)(x1, x2),
                ($ddf22)(x1, x2),
                t2,
            )
        end
        @inline function Base.$fname(
                t1::T1,
                t2::T2,
            ) where {T1 <: AbstractSecondAdjointNode, V, T2 <: Val{V}}
            x1 = t1.x
            x2 = t2
            return SecondAdjointNode1(
                SecondFixed(Base.$fname),
                Base.$fname(x1, x2),
                ($df1)(x1, x2),
                ($ddf11)(x1, x2),
                t1,
            )
        end
        @inline function Base.$fname(
                t1::T1,
                t2::T2,
            ) where {V, T1 <: Val{V}, T2 <: AbstractSecondAdjointNode}
            x1 = t1
            x2 = t2.x
            return SecondAdjointNode1(
                FirstFixed(Base.$fname),
                Base.$fname(x1, x2),
                ($df2)(x1, x2),
                ($ddf22)(x1, x2),
                t2,
            )
        end
        @inline (n::Node2{typeof(Base.$fname), I1, I2})(i, x, θ) where {I1, I2} =
            Base.$fname(n.inner1(i, x, θ), n.inner2(i, x, θ))
        @inline (n::Node2{typeof(Base.$fname), I1, I2})(i, x, θ) where {I1 <: Real, I2} =
            Base.$fname(n.inner1, n.inner2(i, x, θ))
        @inline (n::Node2{typeof(Base.$fname), I1, I2})(i, x, θ) where {I1, I2 <: Real} =
            Base.$fname(n.inner1(i, x, θ), n.inner2)
        @inline (n::Node2{typeof(Base.$fname), I1, I2})(i, x, θ) where {V, I1 <: Val{V}, I2} =
            Base.$fname(V, n.inner2(i, x, θ))
        @inline (n::Node2{typeof(Base.$fname), I1, I2})(i, x, θ) where {I1, V, I2 <: Val{V}} =
            Base.$fname(n.inner1(i, x, θ), V)
    end
end

# ============================================================================
# Univariate functions: (symbol, df, ddf)
# Auto-generated by deps/generate_functionlist.jl using Symbolics.jl
# ============================================================================

const _UNIVARIATES = [
    (:+, x -> one(x), x -> zero(x)),
    (:-, x -> -one(x), x -> zero(x)),
    (:inv, x -> -1 / (x^2), x -> 2 / (x^3)),
    (:sqrt, x -> 1 / (2sqrt(x)), x -> -1 / (4(sqrt(x)^3))),
    (:cbrt, x -> 1 / (3(cbrt(x)^2)), x -> -2 / (9(cbrt(x)^5))),
    (:abs, x -> ifelse(signbit(x), -one(x), one(x)), x -> zero(x)),
    (:abs2, x -> 2x, x -> oftype(x, 2)),
    (:sign, x -> zero(x), x -> zero(x)),
    (:exp, x -> exp(x), x -> exp(x)),
    (:exp2, x -> _clog2(x) * exp2(x), x -> _clog2(x)^2 * exp2(x)),
    (:exp10, x -> _clog10(x) * exp10(x), x -> _clog10(x)^2 * exp10(x)),
    (:expm1, x -> exp(x), x -> exp(x)),
    (:log, x -> 1 / x, x -> -1 / (x^2)),
    (:log2, x -> 1 / (_clog2(x) * x), x -> -_clog2(x) / (_clog2(x)^2 * (x^2))),
    (:log1p, x -> 1 / (1 + x), x -> -1 / ((1 + x)^2)),
    (:log10, x -> 1 / (_clog10(x) * x), x -> -_clog10(x) / (_clog10(x)^2 * (x^2))),
    (:sin, x -> cos(x), x -> -sin(x)),
    (:cos, x -> -sin(x), x -> -cos(x)),
    (:tan, x -> sec(x)^2, x -> 2(sec(x)^2) * tan(x)),
    (:asin, x -> 1 / sqrt(1 - (x^2)), x -> x / ((1 - (x^2)) * sqrt(1 - (x^2)))),
    (:acos, x -> -1 / sqrt(1 - (x^2)), x -> (-x) / ((1 - (x^2)) * sqrt(1 - (x^2)))),
    (:atan, x -> 1 / (1 + x^2), x -> (-2x) / ((1 + x^2)^2)),
    (:acot, x -> -1 / (1 + x^2), x -> (2x) / ((1 + x^2)^2)),
    (:csc, x -> -cot(x) * csc(x), x -> -(-1 - (cot(x)^2)) * csc(x) + (cot(x)^2) * csc(x)),
    (:sec, x -> sec(x) * tan(x), x -> sec(x)^3 + sec(x) * (tan(x)^2)),
    (:cot, x -> -1 - (cot(x)^2), x -> -2cot(x) * (-1 - (cot(x)^2))),
    (:sinh, x -> cosh(x), x -> sinh(x)),
    (:cosh, x -> sinh(x), x -> cosh(x)),
    (:tanh, x -> 1 - (tanh(x)^2), x -> -2tanh(x) * (1 - (tanh(x)^2))),
    (:asinh, x -> 1 / sqrt(1 + x^2), x -> (-x) / ((1 + x^2) * sqrt(1 + x^2))),
    (:acosh, x -> 1 / sqrt(-1 + x^2), x -> (-x) / ((-1 + x^2) * sqrt(-1 + x^2))),
    (:csch, x -> -csch(x) * coth(x), x -> csch(x)^3 + csch(x) * (coth(x)^2)),
    (:sech, x -> -tanh(x) * sech(x), x -> -(1 - (tanh(x)^2)) * sech(x) + (tanh(x)^2) * sech(x)),
    (:coth, x -> -(csch(x)^2), x -> 2(csch(x)^2) * coth(x)),
    (:sind, x -> deg2rad(cosd(x)), x -> -_cd2r(x) * deg2rad(sind(x))),
    (:cosd, x -> -deg2rad(sind(x)), x -> -_cd2r(x) * deg2rad(cosd(x))),
    (:tand, x -> deg2rad(1 + tand(x)^2), x -> (2 * _cd2r(x)) * tand(x) * deg2rad(1 + tand(x)^2)),
    (:cscd, x -> -deg2rad(cscd(x) * cotd(x)), x -> -_cd2r(x) * (-deg2rad(cscd(x) * cotd(x)) * cotd(x) - cscd(x) * deg2rad(1 + cotd(x)^2))),
    (:secd, x -> deg2rad(tand(x) * secd(x)), x -> _cd2r(x) * (deg2rad(tand(x) * secd(x)) * tand(x) + deg2rad(1 + tand(x)^2) * secd(x))),
    (:cotd, x -> -deg2rad(1 + cotd(x)^2), x -> (2 * _cd2r(x)) * cotd(x) * deg2rad(1 + cotd(x)^2)),
    (:atand, x -> 1 / deg2rad(1 + x^2), x -> (-(2 * _cd2r(x)) * x) / (deg2rad(1 + x^2)^2)),
    (:acotd, x -> -1 / deg2rad(1 + x^2), x -> ((2 * _cd2r(x)) * x) / (deg2rad(1 + x^2)^2)),
    (:sinpi, x -> _cpi(x) * cospi(x), x -> -_cpi(x)^2 * sinpi(x)),
    (:cospi, x -> -_cpi(x) * sinpi(x), x -> -_cpi(x)^2 * cospi(x)),
    (:sinc, x -> (-sinpi(x) + _cpi(x) * x * cospi(x)) / (_cpi(x) * (x^2)), x -> ((2 * _cpi(x)^2) * sinpi(x) - (2 * _cpi(x)^3) * x * cospi(x) - _cpi(x)^4 * (x^2) * sinpi(x)) / (_cpi(x)^3 * (x^3))),
    (:deg2rad, x -> _cd2r(x), x -> zero(x)),
    (:rad2deg, x -> _cr2d(x), x -> zero(x)),
    (:signbit, x -> zero(x), x -> zero(x)),
    (:floor, x -> zero(x), x -> zero(x)),
    (:ceil, x -> zero(x), x -> zero(x)),
    # Manual entries (domain guards)
    (:atanh, x -> abs(x) > one(x) ? oftype(x, NaN) : inv(1 - x^2), x -> abs(x) > one(x) ? oftype(x, NaN) : (-abs2(inv(1 - x^2))) * (-2x)),
    (:acoth, x -> abs(x) < one(x) ? oftype(x, NaN) : inv(1 - x^2), x -> abs(x) < one(x) ? oftype(x, NaN) : (-abs2(inv(1 - x^2))) * (-2x)),
]

for (fname, df, ddf) in _UNIVARIATES
    _register_univ(fname, df, ddf)
end

# ============================================================================
# Bivariate functions: (symbol, df1, df2, ddf11, ddf12, ddf22)
# Auto-generated by deps/generate_functionlist.jl using Symbolics.jl
# ============================================================================

const _BIVARIATES = [
    (:+, (x1, x2) -> one(x1), (x1, x2) -> one(x1), (x1, x2) -> zero(x1), (x1, x2) -> zero(x1), (x1, x2) -> zero(x1)),
    (:-, (x1, x2) -> one(x1), (x1, x2) -> -one(x1), (x1, x2) -> zero(x1), (x1, x2) -> zero(x1), (x1, x2) -> zero(x1)),
    (:*, (x1, x2) -> x2, (x1, x2) -> x1, (x1, x2) -> zero(x1), (x1, x2) -> one(x1), (x1, x2) -> zero(x1)),
    (:/, (x1, x2) -> 1 / x2, (x1, x2) -> (-x1) / (x2^2), (x1, x2) -> zero(x1), (x1, x2) -> -1 / (x2^2), (x1, x2) -> (2x1) / (x2^3)),
    (:^, (x1, x2) -> x2 * (x1^(-1 + x2)), (x1, x2) -> log(x1) * (x1^x2), (x1, x2) -> (-1 + x2) * x2 * (x1^(-2 + x2)), (x1, x2) -> x1^(-1 + x2) + x2 * (x1^(-1 + x2)) * log(x1), (x1, x2) -> (log(x1)^2) * (x1^x2)),
    (:atan, (x1, x2) -> x2 / (x1^2 + x2^2), (x1, x2) -> (-x1) / (x1^2 + x2^2), (x1, x2) -> (-2x1 * x2) / ((x1^2 + x2^2)^2), (x1, x2) -> (x1^2 - (x2^2)) / (x1^4 + 2(x1^2) * (x2^2) + x2^4), (x1, x2) -> (2x1 * x2) / ((x1^2 + x2^2)^2)),
    (:hypot, (x1, x2) -> x1 / hypot(x1, x2), (x1, x2) -> x2 / hypot(x1, x2), (x1, x2) -> (-(x1^2) + hypot(x1, x2)^2) / (hypot(x1, x2)^3), (x1, x2) -> (-x1 * x2) / (hypot(x1, x2)^3), (x1, x2) -> (-(x2^2) + hypot(x1, x2)^2) / (hypot(x1, x2)^3)),
    (:max, (x1, x2) -> ifelse(x1 > x2, one(x1), zero(x1)), (x1, x2) -> ifelse(x1 > x2, zero(x1), one(x1)), (x1, x2) -> zero(x1), (x1, x2) -> zero(x1), (x1, x2) -> zero(x1)),
    (:min, (x1, x2) -> ifelse(x1 < x2, one(x1), zero(x1)), (x1, x2) -> ifelse(x1 < x2, zero(x1), one(x1)), (x1, x2) -> zero(x1), (x1, x2) -> zero(x1), (x1, x2) -> zero(x1)),
]

for (fname, df1, df2, ddf11, ddf12, ddf22) in _BIVARIATES
    _register_biv(fname, df1, df2, ddf11, ddf12, ddf22)
end
