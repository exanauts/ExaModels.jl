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

@register_univariate(Base.:+, one, zero)
@register_univariate(Base.:-, _mone, zero)
@register_univariate(Base.inv, x -> -abs2(inv(x)), x -> -(2 * inv(x)) * (-abs2(inv(x))))
@register_univariate(Base.abs, x -> (ifelse(x >= 0, one(x), -one(x))), zero)
@register_univariate(
    Base.sqrt,
    x -> (0.5 / sqrt(x)),
    x -> ((0.5 * -(0.5 / sqrt(x))) / sqrt(x)^2)
)
@register_univariate(
    Base.cbrt,
    x -> (0.3333333333333333 / cbrt(x)^2),
    x -> (
        (0.3333333333333333 * -(2 * (0.3333333333333333 / cbrt(x)^2) * cbrt(x))) /
            (cbrt(x)^2)^2
    )
)
@register_univariate(Base.abs2, x -> 2x, x -> 2)
@register_univariate(Base.exp, exp, exp)
@register_univariate(
    Base.exp2,
    x -> exp2(x) * 0.69314718055994528622676398299518041312694549560546875,
    x ->
    exp2(x) *
        0.69314718055994528622676398299518041312694549560546875 *
        0.69314718055994528622676398299518041312694549560546875
)
@register_univariate(
    Base.exp10,
    x -> exp10(x) * 2.30258509299404590109361379290930926799774169921875,
    x ->
    exp10(x) *
        2.30258509299404590109361379290930926799774169921875 *
        2.30258509299404590109361379290930926799774169921875
)
@register_univariate(Base.log, inv, x -> -abs2(inv(x)))
@register_univariate(
    Base.log2,
    x -> inv(x) / 0.69314718055994528622676398299518041312694549560546875,
    x -> (-abs2(inv(x))) / 0.69314718055994528622676398299518041312694549560546875
)
@register_univariate(Base.log1p, x -> (1 / (1 + x)), x -> (-1 / (1 + x)^2))
@register_univariate(
    Base.log10,
    x -> inv(x) / 2.30258509299404590109361379290930926799774169921875,
    x -> (-abs2(inv(x))) / 2.30258509299404590109361379290930926799774169921875
)
@register_univariate(Base.sin, cos, x -> -sin(x))
@register_univariate(Base.cos, x -> -sin(x), x -> -cos(x))
@register_univariate(Base.tan, x -> 1 + tan(x)^2, x -> 2 * tan(x) * (1 + tan(x)^2))
@register_univariate(
    Base.asin,
    x -> (1 / sqrt(1 - x^2)),
    x -> (-(-(2x) * (0.5 / sqrt(1 - x^2))) / sqrt(1 - x^2)^2)
)
@register_univariate(
    Base.acos,
    x -> (-1 / sqrt(1 - x^2)),
    x -> (-(-(-(2x) * (0.5 / sqrt(1 - x^2)))) / sqrt(1 - x^2)^2)
)
@register_univariate(
    Base.csc,
    x -> (-csc(x)) * cot(x),
    x -> (-(-csc(x)) * cot(x)) * cot(x) + (-(1 + cot(x)^2)) * (-csc(x))
)
@register_univariate(
    Base.sec,
    x -> sec(x) * tan(x),
    x -> sec(x) * tan(x) * tan(x) + (1 + tan(x)^2) * sec(x)
)
@register_univariate(Base.cot, x -> -(1 + cot(x)^2), x -> -2 * cot(x) * (-(1 + cot(x)^2)))
@register_univariate(Base.atan, x -> inv(1 + x^2), x -> (-abs2(inv(1 + x^2))) * 2x)
@register_univariate(Base.acot, x -> -inv(1 + x^2), x -> -(-abs2(inv(1 + x^2))) * 2x)
@register_univariate(
    Base.sind,
    x -> 0.0174532925199432954743716805978692718781530857086181640625 * cosd(x),
    x ->
    0.0174532925199432954743716805978692718781530857086181640625 *
        -0.0174532925199432954743716805978692718781530857086181640625 *
        sind(x)
)
@register_univariate(
    Base.cosd,
    x -> -0.0174532925199432954743716805978692718781530857086181640625 * sind(x),
    x ->
    -0.0174532925199432954743716805978692718781530857086181640625 *
        0.0174532925199432954743716805978692718781530857086181640625 *
        cosd(x)
)
@register_univariate(
    Base.tand,
    x -> 0.0174532925199432954743716805978692718781530857086181640625 * (1 + tand(x)^2),
    x ->
    0.0174532925199432954743716805978692718781530857086181640625 *
        2 *
        tand(x) *
        0.0174532925199432954743716805978692718781530857086181640625 *
        (1 + tand(x)^2)
)
@register_univariate(
    Base.cscd,
    x -> -0.0174532925199432954743716805978692718781530857086181640625 * cscd(x) * cotd(x),
    x ->
    -0.0174532925199432954743716805978692718781530857086181640625 *
        -0.0174532925199432954743716805978692718781530857086181640625 *
        cscd(x) *
        cotd(x) *
        cotd(x) +
        -0.0174532925199432954743716805978692718781530857086181640625 *
        (1 + cotd(x)^2) *
        -0.0174532925199432954743716805978692718781530857086181640625 *
        cscd(x)
)
@register_univariate(
    Base.secd,
    x -> 0.0174532925199432954743716805978692718781530857086181640625 * secd(x) * tand(x),
    x ->
    0.0174532925199432954743716805978692718781530857086181640625 *
        0.0174532925199432954743716805978692718781530857086181640625 *
        secd(x) *
        tand(x) *
        tand(x) +
        0.0174532925199432954743716805978692718781530857086181640625 *
        (1 + tand(x)^2) *
        0.0174532925199432954743716805978692718781530857086181640625 *
        secd(x)
)
@register_univariate(
    Base.cotd,
    x -> -0.0174532925199432954743716805978692718781530857086181640625 * (1 + cotd(x)^2),
    x ->
    -0.0174532925199432954743716805978692718781530857086181640625 *
        2 *
        cotd(x) *
        -0.0174532925199432954743716805978692718781530857086181640625 *
        (1 + cotd(x)^2)
)
@register_univariate(
    Base.atand,
    x -> 57.29577951308232286464772187173366546630859375 / (1 + x^2),
    x -> -57.29577951308232286464772187173366546630859375 * 2 * x / (1 + x^2)^2
)
@register_univariate(
    Base.acotd,
    x -> -57.29577951308232286464772187173366546630859375 / (1 + x^2),
    x -> 57.29577951308232286464772187173366546630859375 * 2 * x / (1 + x^2)^2
)
@register_univariate(Base.sinh, cosh, sinh)
@register_univariate(Base.asinh, x -> 1 / sqrt(x^2 + 1), x -> -x / sqrt(x^2 + 1)^3)
@register_univariate(Base.cosh, sinh, cosh)
@register_univariate(
    Base.acosh,
    x -> 1 / sqrt((x - 1) * (x + 1)),
    x -> -x / sqrt((x - 1) * (x + 1))^3
)
@register_univariate(Base.tanh, x -> 1 - tanh(x)^2, x -> -2 * tanh(x) * (1 - tanh(x)^2))
@register_univariate(
    Base.csch,
    x -> (-coth(x)) * csch(x),
    x -> csch(x)^2 * csch(x) + (-coth(x)) * csch(x) * (-coth(x))
)
@register_univariate(
    Base.sech,
    x -> (-tanh(x)) * sech(x),
    x -> (-(1 - tanh(x)^2)) * sech(x) + (-tanh(x)) * sech(x) * (-tanh(x))
)
@register_univariate(Base.coth, x -> -csch(x)^2, x -> -2 * csch(x) * (-coth(x)) * csch(x))
@register_univariate(
    Base.atanh,
    x -> abs(x) > 1.0 ? NaN : inv(1 - x^2),
    x -> abs(x) > 1.0 ? NaN : (-abs2(inv(1 - x^2))) * (-2x)
)
@register_univariate(
    Base.acoth,
    x -> abs(x) < 1.0 ? NaN : inv(1 - x^2),
    x -> abs(x) < 1.0 ? NaN : (-abs2(inv(1 - x^2))) * (-2x)
)

@register_bivariate(Base.:+, _one, _one, _zero, _zero, _zero)
@register_bivariate(Base.:-, _one, _mone, _zero, _zero, _zero)
@register_bivariate(Base.:*, _x2, _x1, _zero, _one, _zero)
@register_bivariate(
    Base.:^,
    ((x1, x2) -> x2 * x1^(x2 - 1)),
    ((x1, x2) -> x1^x2 * log(x1)),
    ((x1, x2) -> x2 * (x2 - 1) * x1^(x2 - 2)),
    ((x1, x2) -> x2 * x1^(x2 - 1) * log(x1) + x1^(x2 - 1)),
    ((x1, x2) -> x1^x2 * log(x1) * log(x1))
)
@register_bivariate(
    Base.:/,
    ((x1, x2) -> 1 / x2),
    ((x1, x2) -> -x1 / x2^2),
    _zero,
    ((x1, x2) -> -1 / x2^2),
    ((x1, x2) -> 2x1 / x2^3),
)
# @register_bivariate(Base.:<=, _zero, _zero, _zero, _zero, _zero)
# @register_bivariate(Base.:>=, _zero, _zero, _zero, _zero, _zero)
# @register_bivariate(Base.:(==), _zero, _zero, _zero, _zero, _zero)
# @register_bivariate(Base.:<, _zero, _zero, _zero, _zero, _zero)
# @register_bivariate(Base.:>, _zero, _zero, _zero, _zero, _zero)
# @register_bivariate(_and, _zero, _zero, _zero, _zero, _zero)
# @register_bivariate(_or, _zero, _zero, _zero, _zero, _zero)
