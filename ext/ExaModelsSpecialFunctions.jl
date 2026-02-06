module ExaModelsSpecialFunctions

using ExaModels, SpecialFunctions

ExaModels.@register_univariate(
    SpecialFunctions.erfi,
    x -> 1.1283791670955125585606992899556644260883331298828125 * exp(x^2),
    x -> 1.1283791670955125585606992899556644260883331298828125 * exp(x^2) * 2x
)
ExaModels.@register_univariate(
    SpecialFunctions.erfcinv,
    x -> -0.8862269254527579409597137782839126884937286376953125 * exp(erfcinv(x)^2),
    x ->
    -0.8862269254527579409597137782839126884937286376953125 *
        exp(erfcinv(x)^2) *
        2 *
        erfcinv(x) *
        -0.8862269254527579409597137782839126884937286376953125 *
        exp(erfcinv(x)^2)
)
ExaModels.@register_univariate(
    SpecialFunctions.erfcx,
    x -> 2 * x * erfcx(x) - 1.1283791670955125585606992899556644260883331298828125,
    x ->
    2 * erfcx(x) +
        (2 * x * erfcx(x) - 1.1283791670955125585606992899556644260883331298828125) * 2 * x
)
ExaModels.@register_univariate(
    SpecialFunctions.invdigamma,
    x -> inv(trigamma(invdigamma(x))),
    x ->
    (-abs2(inv(trigamma(invdigamma(x))))) *
        polygamma(2, invdigamma(x)) *
        inv(trigamma(invdigamma(x)))
)
ExaModels.@register_univariate(
    SpecialFunctions.bessely1,
    x -> (bessely0(x) - bessely(2, x)) / 2,
    x -> (-bessely1(x) + -(bessely(1, x) - bessely(3, x)) / 2) / 2
)
ExaModels.@register_univariate(
    SpecialFunctions.besselj1,
    x -> (besselj0(x) - besselj(2, x)) / 2,
    x -> (-besselj1(x) + -(besselj(1, x) - besselj(3, x)) / 2) / 2
)
ExaModels.@register_univariate(
    SpecialFunctions.dawson,
    x -> 1 - 2 * x * dawson(x),
    x -> -(2 * dawson(x) + (1 - 2 * x * dawson(x)) * 2 * x)
)
ExaModels.@register_univariate(
    SpecialFunctions.airyaiprime,
    x -> x * airyai(x),
    x -> airyai(x) + airyaiprime(x) * x
)
ExaModels.@register_univariate(
    SpecialFunctions.erf,
    x -> 1.1283791670955125585606992899556644260883331298828125 * exp(-x * x),
    x -> 1.1283791670955125585606992899556644260883331298828125 * exp(-x * x) * (-2x)
)
ExaModels.@register_univariate(SpecialFunctions.digamma, trigamma, x -> polygamma(2, x))
ExaModels.@register_univariate(
    SpecialFunctions.gamma,
    x -> digamma(x) * gamma(x),
    x -> trigamma(x) * gamma(x) + digamma(x) * gamma(x) * digamma(x)
)
ExaModels.@register_univariate(SpecialFunctions.airyai, airyaiprime, x -> x * airyai(x))
ExaModels.@register_univariate(SpecialFunctions.airybi, airybiprime, x -> x * airybi(x))
ExaModels.@register_univariate(
    SpecialFunctions.erfinv,
    x -> 0.8862269254527579409597137782839126884937286376953125 * exp(erfinv(x)^2),
    x ->
    0.8862269254527579409597137782839126884937286376953125 *
        exp(erfinv(x)^2) *
        2 *
        erfinv(x) *
        0.8862269254527579409597137782839126884937286376953125 *
        exp(erfinv(x)^2)
)
ExaModels.@register_univariate(
    SpecialFunctions.bessely0,
    x -> -bessely1(x),
    x -> -(bessely0(x) - bessely(2, x)) / 2
)
ExaModels.@register_univariate(
    SpecialFunctions.erfc,
    x -> -1.1283791670955125585606992899556644260883331298828125 * exp(-x * x),
    x -> -1.1283791670955125585606992899556644260883331298828125 * exp(-x * x) * (-2x)
)
ExaModels.@register_univariate(
    SpecialFunctions.trigamma,
    x -> polygamma(2, x),
    x -> polygamma(3, x)
)
ExaModels.@register_univariate(
    SpecialFunctions.airybiprime,
    x -> x * airybi(x),
    x -> airybi(x) + airybiprime(x) * x
)
ExaModels.@register_univariate(
    SpecialFunctions.besselj0,
    x -> -besselj1(x),
    x -> -(besselj0(x) - besselj(2, x)) / 2
)

ExaModels.@register_bivariate(
    SpecialFunctions.beta,
    (x1, x2) -> beta(x1, x2) * (digamma(x1) - digamma(x1 + x2)),
    (x1, x2) -> beta(x1, x2) * (digamma(x2) - digamma(x1 + x2)),
    (x1, x2) ->
    beta(x1, x2) * (digamma(x1) - digamma(x1 + x2)) * (digamma(x1) - digamma(x1 + x2)) +
        (trigamma(x1) + -trigamma(x1 + x2)) * beta(x1, x2),
    (x1, x2) ->
    beta(x1, x2) * (digamma(x2) - digamma(x1 + x2)) * (digamma(x1) - digamma(x1 + x2)) +
        (-trigamma(x1 + x2)) * beta(x1, x2),
    (x1, x2) ->
    beta(x1, x2) * (digamma(x2) - digamma(x1 + x2)) * (digamma(x2) - digamma(x1 + x2)) +
        (trigamma(x2) + -trigamma(x1 + x2)) * beta(x1, x2)
)
ExaModels.@register_bivariate(
    SpecialFunctions.logbeta,
    (x1, x2) -> digamma(x1) - digamma(x1 + x2),
    (x1, x2) -> digamma(x2) - digamma(x1 + x2),
    (x1, x2) -> trigamma(x1) + -trigamma(x1 + x2),
    (x1, x2) -> -trigamma(x1 + x2),
    (x1, x2) -> trigamma(x2) + -trigamma(x1 + x2)
)

end # module ExaModelsSpecialFunctions
