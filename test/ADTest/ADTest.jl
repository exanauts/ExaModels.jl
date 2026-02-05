module ADTest

using ExaModels
using Test, ForwardDiff, SpecialFunctions, NLPModels

include("expression.jl")

const FUNCTIONS = [
    ("basic-functions-:+", x -> +(x[1])),
    ("basic-functions-:-", x -> -(x[1])),
    ("basic-functions-inv", x -> inv(x[1])),
    ("basic-functions-abs", x -> abs(x[1])),
    ("basic-functions-sqrt", x -> sqrt(x[1])),
    ("basic-functions-cbrt", x -> cbrt(x[1])),
    ("basic-functions-abs2", x -> abs2(x[1])),
    ("basic-functions-exp", x -> exp(x[1])),
    ("basic-functions-exp2", x -> exp2(x[1])),
    ("basic-functions-exp10", x -> exp10(x[1])),
    ("basic-functions-log", x -> log(x[1])),
    ("basic-functions-log2", x -> log2(x[1])),
    ("basic-functions-log1p", x -> log1p(x[1])),
    ("basic-functions-log10", x -> log10(x[1])),
    ("basic-functions-sin", x -> sin(x[1])),
    ("basic-functions-cos", x -> cos(x[1])),
    ("basic-functions-tan", x -> tan(x[1])),
    ("basic-functions-asin", x -> asin(x[1])),
    ("basic-functions-acos", x -> acos(x[1])),
    ("basic-functions-csc", x -> csc(x[1])),
    ("basic-functions-sec", x -> sec(x[1])),
    ("basic-functions-cot", x -> cot(x[1])),
    ("basic-functions-atan", x -> atan(x[1])),
    ("basic-functions-acot", x -> acot(x[1])),
    # ("basic-functions-sind", x-> sind(x[1])), # cannot extend function 
    # ("basic-functions-cosd", x-> cosd(x[1])), # cannot extend function 
    # ("basic-functions-tand", x-> tand(x[1])), # cannot extend function 
    ("basic-functions-cscd", x -> cscd(x[1])),
    ("basic-functions-secd", x -> secd(x[1])),
    ("basic-functions-cotd", x -> cotd(x[1])),
    # ("basic-functions-atand", x-> atand(x[1])), # cannot extend function 
    # ("basic-functions-acotd", x-> acotd(x[1])), # cannot extend function 
    ("basic-functions-sinh", x -> sinh(x[1])),
    ("basic-functions-asinh", x -> asinh(x[1])),
    ("basic-functions-cosh", x -> cosh(x[1])),
    ("basic-functions-acosh", x -> acosh(x[1] + 1)),
    ("basic-functions-tanh", x -> tanh(x[1])),
    ("basic-functions-csch", x -> csch(x[1])),
    ("basic-functions-sech", x -> sech(x[1])),
    ("basic-functions-coth", x -> coth(x[1])),
    ("basic-functions-atanh", x -> atanh(x[1])),
    # ("basic-functions-acoth", x-> acoth(x[1])), # range issue
    ("basic-functions-:+", x -> +(x[1], x[2])),
    ("basic-functions-:-", x -> -(x[1], x[2])),
    ("basic-functions-:*", x -> *(x[1], x[2])),
    ("basic-functions-:^", x -> ^(x[1], x[2])),
    ("basic-functions-:/", x -> /(x[1], x[2])),
    # ("basic-functions-:<=", x-> <=(x[1], x[2])), # not implemented 
    # ("basic-functions-:>=", x-> >=(x[1], x[2])), # not implemented 
    # ("basic-functions-:(==),", x-> (==)(x[1], x[2])), # not implemented 
    # ("basic-functions-:<", x-> <(x[1], x[2])), # not implemented 
    # ("basic-functions-:>", x-> >(x[1], x[2])), # not implemented 
    ("special-functions-erfi", x -> erfi(x[1])),
    ("special-functions-erfcinv", x -> erfcinv(x[1])),
    ("special-functions-erfcx", x -> erfcx(x[1])),
    ("special-functions-invdigamma", x -> invdigamma(x[1])),
    ("special-functions-bessely1", x -> bessely1(x[1])),
    ("special-functions-besselj1", x -> besselj1(x[1])),
    ("special-functions-dawson", x -> dawson(x[1])),
    ("special-functions-airyaiprime", x -> airyaiprime(x[1])),
    ("special-functions-erf", x -> erf(x[1])),
    ("special-functions-trigamma", x -> trigamma(x[1])),
    ("special-functions-gamma", x -> gamma(x[1])),
    ("special-functions-airyaiprime", x -> airyaiprime(x[1])),
    ("special-functions-airybiprime", x -> airybiprime(x[1])),
    ("special-functions-erfinv", x -> erfinv(x[1])),
    ("special-functions-bessely0", x -> bessely0(x[1])),
    ("special-functions-erfc", x -> erfc(x[1])),
    ("special-functions-trigamma", x -> trigamma(x[1])),
    ("special-functions-airybiprime", x -> airybiprime(x[1])),
    ("special-functions-besselj0", x -> besselj0(x[1])),
    ("special-functions-beta", x -> beta(x[1], x[2])),
    ("special-functions-logbeta", x -> logbeta(x[1], x[2])),
    (
        "composite-functions-1-1",
        x -> beta(erf(x[1] / x[2] / 3.0) + 3.0 * x[2], erf(x[9])^2),
    ),
    ("composite-functions-1-2", x -> 0 * x[1]),
    (
        "composite-functions-1-3",
        x -> beta(cos(log(abs2(inv(inv(x[1]))) + 1.0)), erfc(tanh(0 * x[1]))),
    ),
    ("composite-functions-1-4", x -> (0 * x[1]^x[3]^1.0 + x[1]) / x[9] / x[10]),
    (
        "composite-functions-1-5",
        x -> exp(x[1] + 1.0)^x[2] * log(abs2(x[3]) + 3) / tanh(x[2]),
    ),
    ("composite-functions-1-6", x -> beta(2 * logbeta(x[1], x[5]), beta(x[2], x[3]))),
    ("composite-functions-1-7", x -> besselj0(exp(erf(-x[1])))),
    ("composite-functions-1-8", x -> erfc(abs2(x[1]^2 / x[2])^x[9] / x[10])),
    ("composite-functions-1-9", x -> erfc(x[1])^erf(2.5x[2])),
    ("composite-functions-1-10", x -> sin(1 / x[1])),
    ("composite-functions-1-11", x -> exp(x[2]) / cos(x[1])^2 + sin(x[1]^2)),
    ("composite-functions-1-12", x -> sin(x[9]inv(x[1]) - x[8]inv(x[2]))),
    ("composite-functions-1-13", x -> x[1] / log(x[2]^2 + 9.0)),
    (
        "composite-functions-1-14",
        x -> beta(beta(tan(beta(x[1], 1) + 2.0), cos(sin(x[2]))), x[3]),
    ),
]

const PARAMETER_FUNCTIONS = [
    ("parameter-basic-1", (x, θ) -> x[1] + θ[1]),
    ("parameter-basic-2", (x, θ) -> x[1] * θ[1]),
    ("parameter-basic-3", (x, θ) -> x[1]^2 + θ[1] * x[2]),
    ("parameter-basic-4", (x, θ) -> sin(x[1]) + cos(θ[1])),
    ("parameter-basic-5", (x, θ) -> exp(x[1] + θ[1])),
    ("parameter-basic-6", (x, θ) -> log(x[1]^2 + θ[1]^2)),
    ("parameter-basic-7", (x, θ) -> x[1] / (1 + θ[1])),
    ("parameter-basic-8", (x, θ) -> θ[1] * sin(x[1]) + θ[2] * cos(x[2])),
    ("parameter-composite-1", (x, θ) -> exp(x[1] * θ[1]) + sin(x[2] + θ[2])),
    ("parameter-composite-2", (x, θ) -> beta(x[1] + θ[1], x[2] + θ[2])),
    ("parameter-composite-3", (x, θ) -> sqrt(x[1]^2 + θ[1]^2) * log(x[2] + θ[2] + 1)),
    ("parameter-composite-4", (x, θ) -> gamma(x[1] + 1) * θ[1] + erf(x[2] * θ[2])),
]

function gradient(f, x)
    T = eltype(x)
    y = fill!(similar(x), zero(T))
    ExaModels.gradient!(y, (p, x, θ) -> f(x), x, nothing, nothing, one(T))
    return y
end

function gradient_with_params(f, x, θ)
    T = eltype(x)
    y = fill!(similar(x), zero(T))
    ExaModels.gradient!(y, (p, x, θ) -> f(x, θ), x, θ, nothing, one(T))
    return y
end

function sgradient(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(nothing), nothing)
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    a1 = unique(y1)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Tuple{Int,Int}, n)

    ExaModels.sgradient!(buffer_I, ff, nothing, nothing, nothing, comp, 0, NaN)
    ExaModels.sgradient!(buffer, ff, nothing, x, nothing, comp, 0, one(T))

    y = zeros(length(x))
    y[collect(i for (i, j) in buffer_I)] += buffer

    return y
end

function sgradient_with_params(f, x, θ)
    T = eltype(x)

    ff = f(ExaModels.VarSource(), ExaModels.ParameterSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(nothing), nothing)
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    a1 = unique(y1)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Tuple{Int,Int}, n)

    ExaModels.sgradient!(buffer_I, ff, nothing, nothing, θ, comp, 0, NaN)
    ExaModels.sgradient!(buffer, ff, nothing, x, θ, comp, 0, one(T))

    y = zeros(length(x))
    y[collect(i for (i, j) in buffer_I)] += buffer

    return y
end

function sjacobian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(nothing), nothing)
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    a1 = unique(y1)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.sjacobian!(buffer_I, buffer_J, ff, nothing, nothing, nothing, comp, 0, 0, NaN)
    ExaModels.sjacobian!(buffer, nothing, ff, nothing, x, nothing, comp, 0, 0, one(T))

    y = zeros(length(x))
    y[buffer_J] += buffer

    return y
end

function sjacobian_with_params(f, x, θ)
    T = eltype(x)

    ff = f(ExaModels.VarSource(), ExaModels.ParameterSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(nothing), nothing)
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    a1 = unique(y1)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.sjacobian!(buffer_I, buffer_J, ff, nothing, nothing, θ, comp, 0, 0, NaN)
    ExaModels.sjacobian!(buffer, nothing, ff, nothing, x, θ, comp, 0, 0, one(T))

    y = zeros(length(x))
    y[buffer_J] += buffer

    return y
end

function shessian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(nothing), nothing)
    y2 = []
    ExaModels.hrpass0(t, nothing, y2, nothing, nothing, 0, NaN, NaN)

    a2 = unique(y2)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

    n = length(a2)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.shessian!(
        buffer_I,
        buffer_J,
        ff,
        nothing,
        nothing,
        nothing,
        comp,
        0,
        NaN,
        NaN,
    )
    ExaModels.shessian!(buffer, nothing, ff, nothing, x, nothing, comp, 0, one(T), zero(T))

    y = zeros(length(x), length(x))
    for (k, (i, j)) in enumerate(zip(buffer_I, buffer_J))
        if i == j
            y[i, j] += buffer[k]
        else
            y[i, j] += buffer[k]
            y[j, i] += buffer[k]
        end
    end
    return y
end

function shessian_with_params(f, x, θ)
    T = eltype(x)

    ff = f(ExaModels.VarSource(), ExaModels.ParameterSource())
    t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(nothing), nothing)
    y2 = []
    ExaModels.hrpass0(t, nothing, y2, nothing, nothing, 0, NaN, NaN)

    a2 = unique(y2)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

    n = length(a2)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.shessian!(buffer_I, buffer_J, ff, nothing, nothing, θ, comp, 0, NaN, NaN)
    ExaModels.shessian!(buffer, nothing, ff, nothing, x, θ, comp, 0, one(T), zero(T))

    y = zeros(length(x), length(x))
    for (k, (i, j)) in enumerate(zip(buffer_I, buffer_J))
        if i == j
            y[i, j] += buffer[k]
        else
            y[i, j] += buffer[k]
            y[j, i] += buffer[k]
        end
    end
    return y
end

function runtests()
    @testset "AD test" begin
        test_expression()
        for (name, f) in FUNCTIONS
            x0 = rand(10)
            @testset "$name" begin
                g = ForwardDiff.gradient(f, x0)
                h = ForwardDiff.hessian(f, x0)
                @test gradient(f, x0) ≈ g atol = 1e-6
                @test sgradient(f, x0) ≈ g atol = 1e-6
                @test sjacobian(f, x0) ≈ g atol = 1e-6
                @test shessian(f, x0) ≈ h atol = 1e-6
            end
        end
        @testset "Parameter tests" begin
            for (name, f) in PARAMETER_FUNCTIONS
                x0 = rand(10)
                θ0 = rand(5)
                @testset "$name" begin
                    f_fixed_θ = x -> f(x, θ0)
                    g = ForwardDiff.gradient(f_fixed_θ, x0)
                    h = ForwardDiff.hessian(f_fixed_θ, x0)
                    @test gradient_with_params(f, x0, θ0) ≈ g atol = 1e-6
                    @test sgradient_with_params(f, x0, θ0) ≈ g atol = 1e-6
                    @test sjacobian_with_params(f, x0, θ0) ≈ g atol = 1e-6
                    @test shessian_with_params(f, x0, θ0) ≈ h atol = 1e-6
                end
            end
        end
    end
end

end #module
