module ADTest

using ExaModels
using Test, ForwardDiff, SpecialFunctions

# Each unique anonymous function creates a distinct type, triggering recompilation of the
# entire AD pipeline (gradient, sgradient, sjacobian, shessian). Individual unary/binary
# function derivatives are already validated by verify_derivative_tables(). Here we test
# the full AD pipeline with a small number of composites that cover all operator categories.
const FUNCTIONS = [
    # Unary basics: sin, cos, exp, log, sqrt, abs, abs2, inv, cbrt, tanh, atan
    ("basic-unary",
     x -> sin(x[1]) + cos(x[2]) + exp(x[3]) + log(x[4] + 1) + sqrt(abs2(x[5]) + 1) +
          tanh(x[6]) + atan(x[7]) + cbrt(x[8]) + abs(x[9]) + inv(x[10])),
    # Binary ops: +, -, *, ^, /
    ("basic-binary",
     x -> (x[1] + x[2]) * (x[3] - x[4]) + x[5]^x[6] + x[7] / x[8]),
    # Special functions: erf, erfc, gamma, beta, bessel, trigamma, dawson
    ("special-fns",
     x -> erf(x[1]) + erfc(x[2]) + gamma(x[3] + 1) + beta(x[4] + 1, x[5] + 1) +
          besselj0(x[6]) + dawson(x[7]) + trigamma(x[8] + 1)),
    # Deep nesting with erf and beta
    ("composite-deep",
     x -> beta(erf(x[1] / x[2] / 3.0) + 3.0 * x[2], erf(x[9])^2)),
    # Edge cases: zero multiplication, inv chains, zero-power
    ("composite-edge",
     x -> (0 * x[1]) + beta(cos(log(abs2(inv(inv(x[1]))) + 1.0)), erfc(tanh(0 * x[1]))) +
          (0 * x[1]^x[3]^1.0 + x[1]) / x[9] / x[10]),
    # Mixed ops: exp-power, log, trig squares, simple division
    ("composite-mixed",
     x -> exp(x[1] + 1.0)^x[2] * log(abs2(x[3]) + 3) / tanh(x[2]) +
          exp(x[2]) / cos(x[1])^2 + sin(x[1]^2) + x[1] / log(x[2]^2 + 9.0)),
    # Special function compositions: bessel-exp-erf, erfc-power-erf, sin-inv
    ("composite-special",
     x -> besselj0(exp(erf(-x[1]))) + erfc(x[1])^erf(2.5x[2]) +
          erfc(abs2(x[1]^2 / x[2])^x[9] / x[10]) + sin(x[9]inv(x[1]) - x[8]inv(x[2]))),
    # Nested beta and logbeta
    ("composite-beta",
     x -> beta(2 * logbeta(x[1], x[5]), beta(x[2], x[3])) +
          beta(beta(tan(beta(x[1], 1) + 2.0), cos(sin(x[2]))), x[3])),
]

# Consolidated parameter functions: each unique closure type triggers recompilation,
# so we combine into fewer composites while covering the same operator categories.
const PARAMETER_FUNCTIONS = [
    # Basic parameter ops: +, *, ^2, sin, cos, exp, log, /
    ("parameter-basic",
     (x, θ) -> (x[1] + θ[1]) + x[1] * θ[1] + x[1]^2 + θ[1] * x[2] +
               sin(x[1]) + cos(θ[1]) + exp(x[1] + θ[1]) +
               log(x[1]^2 + θ[1]^2 + 1) + x[1] / (1 + θ[1])),
    # Multi-parameter with θ[1] and θ[2]
    ("parameter-multi",
     (x, θ) -> θ[1] * sin(x[1]) + θ[2] * cos(x[2]) + exp(x[1] * θ[1]) + sin(x[2] + θ[2])),
    # Composite with special functions
    ("parameter-composite",
     (x, θ) -> sqrt(x[1]^2 + θ[1]^2 + 1) * log(x[2] + θ[2] + 2) +
               gamma(x[1] + 2) * θ[1] + erf(x[2] * θ[2]) +
               beta(x[1] + θ[1] + 1, x[2] + θ[2] + 1)),
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
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y1, a1 = ExaModels.grpass(d, nothing, nothing, ExaModels.NaNSource{T}(), ((),()), T(NaN))

    comp = ExaModels.Compressor(y1)

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Tuple{Int,Int}, n)

    ExaModels.sgradient!(buffer_I, ff, nothing, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), comp, 0, T(NaN))
    ExaModels.sgradient!(buffer, ff, nothing, x, nothing, comp, 0, one(T))

    y = zeros(length(x))
    y[collect(i for (i, j) in buffer_I)] += buffer

    return y
end

function sgradient_with_params(f, x, θ)
    T = eltype(x)

    ff = f(ExaModels.VarSource(), ExaModels.ParameterSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y1, a1 = ExaModels.grpass(d, nothing, nothing, ExaModels.NaNSource{T}(), ((),()), T(NaN))

    comp = ExaModels.Compressor(y1)

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Tuple{Int,Int}, n)

    ExaModels.sgradient!(buffer_I, ff, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), θ, comp, 0, T(NaN))
    ExaModels.sgradient!(buffer, ff, nothing, x, θ, comp, 0, one(T))

    y = zeros(length(x))
    y[collect(i for (i, j) in buffer_I)] += buffer

    return y
end

function sjacobian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y1, a1 = ExaModels.grpass(d, nothing, nothing, ExaModels.NaNSource{T}(), ((),()), T(NaN))

    comp = ExaModels.Compressor(y1)

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.sjacobian!(buffer_I, buffer_J, ff, nothing, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), comp, 0, 0, T(NaN))
    ExaModels.sjacobian!(buffer, nothing, ff, nothing, x, nothing, comp, 0, 0, one(T))

    y = zeros(length(x))
    y[buffer_J] += buffer

    return y
end

function sjacobian_with_params(f, x, θ)
    T = eltype(x)

    ff = f(ExaModels.VarSource(), ExaModels.ParameterSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y1, a1 = ExaModels.grpass(d, nothing, nothing, nothing, ((),()), T(NaN))

    comp = ExaModels.Compressor(y1)

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.sjacobian!(buffer_I, buffer_J, ff, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), θ, comp, 0, 0, T(NaN))
    ExaModels.sjacobian!(buffer, nothing, ff, nothing, x, θ, comp, 0, 0, one(T))

    y = zeros(length(x))
    y[buffer_J] += buffer

    return y
end

function shessian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y2, a2 = ExaModels.hrpass0(t, nothing, nothing, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), ((),()), T(NaN), T(NaN))

    comp = ExaModels.Compressor(y2)

    n = length(a2)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.shessian!(
        buffer_I,
        buffer_J,
        ff,
        nothing,
        ExaModels.NaNSource{T}(),
        ExaModels.NaNSource{T}(),
        comp,
        0,
        T(NaN),
        T(NaN)
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
    t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(ExaModels.NaNSource{T}()), ExaModels.NaNSource{T}())
    y2, a2 = ExaModels.hrpass0(t, nothing, nothing, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), ((),()), T(NaN), T(NaN))

    comp = ExaModels.Compressor(y2)

    n = length(a2)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.shessian!(buffer_I, buffer_J, ff, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), θ, comp, 0, T(NaN), T(NaN))
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
# Verify derivative tables against finite differences.
# Step sizes: h1 for first derivatives (central diff truncation O(h^2)),
# h2 for second derivatives (optimal h ~ eps^(1/4) ≈ 1e-4 to balance truncation vs rounding).
function verify_derivative_tables()
    h1 = 1.0e-7
    h2 = 1.0e-4
    @testset "Univariate derivative tables" begin
        # Special test points for domain-sensitive functions
        special_x = Dict(
            :acosh => 1.7, :acoth => 1.7, :atanh => 0.3,
            # Degree-based trig: use degrees as test points
            :sind => 15.0, :cosd => 15.0, :tand => 15.0,
            :cscd => 15.0, :secd => 15.0, :cotd => 15.0,
            :atand => 15.0, :acotd => 15.0,
        )
        for (fname, df, ddf) in ExaModels._UNIVARIATES
            f = getfield(Base, fname)
            test_x = get(special_x, fname, 0.7)
            @testset "$fname" begin
                fd_df = (f(test_x + h1) - f(test_x - h1)) / (2h1)
                @test df(test_x) ≈ fd_df atol = 1.0e-6 rtol = 1.0e-5
                ddf_val = ddf(test_x)
                if !isnan(ddf_val)
                    fd_ddf = (f(test_x + h2) - 2 * f(test_x) + f(test_x - h2)) / h2^2
                    @test ddf_val ≈ fd_ddf atol = 1.0e-6 rtol = 1.0e-4
                end
            end
        end
    end
    return @testset "Bivariate derivative tables" begin
        for (fname, df1, df2, ddf11, ddf12, ddf22) in ExaModels._BIVARIATES
            f = getfield(Base, fname)
            x1, x2 = 0.7, 1.3
            @testset "$fname" begin
                fd_df1 = (f(x1 + h1, x2) - f(x1 - h1, x2)) / (2h1)
                fd_df2 = (f(x1, x2 + h1) - f(x1, x2 - h1)) / (2h1)
                @test df1(x1, x2) ≈ fd_df1 rtol = 1.0e-5
                @test df2(x1, x2) ≈ fd_df2 rtol = 1.0e-5
                fd_ddf11 = (f(x1 + h2, x2) - 2 * f(x1, x2) + f(x1 - h2, x2)) / h2^2
                fd_ddf22 = (f(x1, x2 + h2) - 2 * f(x1, x2) + f(x1, x2 - h2)) / h2^2
                fd_ddf12 = (f(x1 + h2, x2 + h2) - f(x1 + h2, x2 - h2) - f(x1 - h2, x2 + h2) + f(x1 - h2, x2 - h2)) / (4 * h2^2)
                @test ddf11(x1, x2) ≈ fd_ddf11 atol = 1.0e-6 rtol = 1.0e-4
                @test ddf12(x1, x2) ≈ fd_ddf12 atol = 1.0e-6 rtol = 1.0e-4
                @test ddf22(x1, x2) ≈ fd_ddf22 atol = 1.0e-6 rtol = 1.0e-4
            end
        end
    end
end

function runtests()
    @testset "AD test" begin
        verify_derivative_tables()
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
