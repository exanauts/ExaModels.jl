#!/usr/bin/env julia
#
# Development-time script that uses Symbolics.jl to auto-generate derivative
# rules for ExaModels' ExaModelsSpecialFunctions extension.
#
# Usage (from repo root):
#   julia --project=deps deps/generate_specialfunctions.jl          # print to stdout
#   julia --project=deps deps/generate_specialfunctions.jl --write   # overwrite ext/ExaModelsSpecialFunctions.jl
#
# First-time setup:
#   julia --project=deps -e 'import Pkg; Pkg.instantiate()'
#
# This script is NOT a runtime dependency — it generates static Julia code.

using Symbolics, SpecialFunctions

@variables x x1 x2

# ============================================================================
# Symbolics helpers (shared with generate_functionlist.jl pattern)
# ============================================================================

function sym_derivs_univ(f_sym)
    expr = f_sym(x)
    df = Symbolics.simplify(Symbolics.derivative(expr, x))
    ddf = Symbolics.simplify(Symbolics.derivative(df, x))
    return df, ddf
end

function sym_derivs_biv(f_sym)
    expr = f_sym(x1, x2)
    df1 = Symbolics.simplify(Symbolics.derivative(expr, x1))
    df2 = Symbolics.simplify(Symbolics.derivative(expr, x2))
    ddf11 = Symbolics.simplify(Symbolics.derivative(df1, x1))
    ddf12 = Symbolics.simplify(Symbolics.derivative(df1, x2))
    ddf22 = Symbolics.simplify(Symbolics.derivative(df2, x2))
    return df1, df2, ddf11, ddf12, ddf22
end

function _runic_mul_spaces(s::String)
    for _ in 1:2
        s = replace(s, r"(\S)\*(\S)" => s"\1 * \2")
    end
    return s
end

# ============================================================================
# Type-generic constant replacement
# ============================================================================

# Float64 constants emitted by Symbolics.jl for SpecialFunctions derivatives.
# All replaced with type-generic helper calls to support Float32.
const SF_FLOAT_CONSTANT_REPLACEMENTS = [
    # 1/sqrt(π) ≈ 0.5641895835477563
    (string(1 / sqrt(Float64(π))), "_cinvsqrtpi({V})"),
    # 2/sqrt(π) ≈ 1.1283791670955126
    (string(2 / sqrt(Float64(π))), "(2 * _cinvsqrtpi({V}))"),
    # 4/sqrt(π) ≈ 2.256758334191025
    (string(4 / sqrt(Float64(π))), "(4 * _cinvsqrtpi({V}))"),
    # sqrt(π)/2 ≈ 0.886226925452758
    (string(sqrt(Float64(π)) / 2), "_csqrtpihalf({V})"),
    # π/2 ≈ 1.5707963267948968
    (string(Float64(π) / 2), "(_cpi({V}) / 2)"),
    # π ≈ 3.141592653589793
    (string(Float64(π)), "_cpi({V})"),
]

function replace_sf_float_constants(s::String, typevar::String)
    for (literal, template) in SF_FLOAT_CONSTANT_REPLACEMENTS
        repl = replace(template, "{V}" => typevar)
        escaped = replace(literal, "." => "\\.")
        s = replace(s, Regex(escaped * raw"(?=[a-zA-Z_(])") => repl * " * ")
        s = replace(s, literal => repl)
    end
    return s
end

function replace_integer_constants(s::String, typevar::String)
    if occursin(r"^-?\d+$", s)
        n = parse(Int, s)
        n == 0 && return "zero($typevar)"
        n == 1 && return "one($typevar)"
        n == -1 && return "-one($typevar)"
        return "oftype($typevar, $s)"
    end
    return s
end

function sym2lambda_sf(expr, vars::Symbol...)
    s = string(expr)
    typevar = string(vars[1])
    # Qualify bare SpecialFunctions names that Symbolics sometimes unqualifies
    s = replace_sf_float_constants(s, typevar)
    s = replace_integer_constants(s, typevar)
    s = _runic_mul_spaces(s)
    if length(vars) == 1
        return "$(vars[1]) -> $s"
    else
        vstr = join(vars, ", ")
        return "($vstr) -> $s"
    end
end

# ============================================================================
# Function tables
# ============================================================================

# Univariates auto-generated via Symbolics.jl
const AUTO_UNIVARIATES = [
    :erf,
    :erfc,
    :erfi,
    :erfcx,
    :erfcinv,
    :erfinv,    # will be overridden by manual entry below if needed
    :digamma,
    :trigamma,
    :invdigamma,
    :gamma,
    :airyai,
    :airybi,
    :airyaiprime,
    :airybiprime,
    :besselj0,
    :bessely0,
    :besselj1,
    :bessely1,
    :dawson,
]

# Manual overrides: (fname, df_string, ddf_string) — used when Symbolics fails or
# when the auto-generated output contains hardcoded Float64 literals that the constant
# replacement regex cannot match due to Symbolics.jl printing fewer significant digits.
const MANUAL_UNIVARIATES = [
    (
        :erfinv,
        "x -> _csqrtpihalf(x) * exp(SpecialFunctions.erfinv(x)^2)",
        "x -> _csqrtpihalf(x) * exp(SpecialFunctions.erfinv(x)^2) * 2 * SpecialFunctions.erfinv(x) * _csqrtpihalf(x) * exp(SpecialFunctions.erfinv(x)^2)",
    ),
    (
        # Symbolics.jl emits truncated Float64 literals (e.g. 0.886226925452758) that
        # don't match string(sqrt(π)/2) exactly, so we override with type-generic helpers.
        :erfcinv,
        "x -> -_csqrtpihalf(x) * exp(SpecialFunctions.erfcinv(x)^2)",
        "x -> (_cpi(x) / 2) * SpecialFunctions.erfcinv(x) * exp(2 * SpecialFunctions.erfcinv(x)^2)",
    ),
]
const MANUAL_NAMES = Set(first(t) for t in MANUAL_UNIVARIATES)

# Bivariates auto-generated via Symbolics.jl
const AUTO_BIVARIATES = [:beta, :logbeta]

# ============================================================================
# Code generation
# ============================================================================

function generate()
    io = IOBuffer()

    println(io, "module ExaModelsSpecialFunctions")
    println(io)
    println(io, "using ExaModels, SpecialFunctions")
    println(io)
    println(
        io,
        "# Type-generic constant helpers (avoid Float64 literals for Float32 compatibility)",
    )
    println(io, "@inline _cinvsqrtpi(x) = oftype(x, 1 / sqrt(π))")
    println(io, "@inline _csqrtpihalf(x) = oftype(x, sqrt(π) / 2)")
    println(io, "@inline _cpi(x) = oftype(x, π)")
    println(io)
    println(io, "# " * "="^74)
    println(io, "# Univariate SpecialFunctions")
    println(io, "# Auto-generated by deps/generate_specialfunctions.jl via Symbolics.jl")
    println(io, "# " * "="^74)
    println(io)

    for fname in AUTO_UNIVARIATES
        fname in MANUAL_NAMES && continue
        f = getfield(SpecialFunctions, fname)
        df_sym, ddf_sym = sym_derivs_univ(f)
        df_s = sym2lambda_sf(df_sym, :x)
        ddf_s = sym2lambda_sf(ddf_sym, :x)
        println(io, "ExaModels.@register_univariate(")
        println(io, "    SpecialFunctions.$fname,")
        println(io, "    $df_s,")
        println(io, "    $ddf_s,")
        println(io, ")")
    end

    if !isempty(MANUAL_UNIVARIATES)
        println(io)
        println(io, "# Manual entries (Symbolics.jl cannot auto-derive these)")
        for (fname, df_s, ddf_s) in MANUAL_UNIVARIATES
            println(io, "ExaModels.@register_univariate(")
            println(io, "    SpecialFunctions.$fname,")
            println(io, "    $df_s,")
            println(io, "    $ddf_s,")
            println(io, ")")
        end
    end

    println(io)
    println(io, "# " * "="^74)
    println(io, "# Bivariate SpecialFunctions")
    println(io, "# Auto-generated by deps/generate_specialfunctions.jl via Symbolics.jl")
    println(io, "# " * "="^74)
    println(io)

    for fname in AUTO_BIVARIATES
        f = getfield(SpecialFunctions, fname)
        df1_sym, df2_sym, ddf11_sym, ddf12_sym, ddf22_sym = sym_derivs_biv(f)
        df1_s = sym2lambda_sf(df1_sym, :x1, :x2)
        df2_s = sym2lambda_sf(df2_sym, :x1, :x2)
        ddf11_s = sym2lambda_sf(ddf11_sym, :x1, :x2)
        ddf12_s = sym2lambda_sf(ddf12_sym, :x1, :x2)
        ddf22_s = sym2lambda_sf(ddf22_sym, :x1, :x2)
        println(io, "ExaModels.@register_bivariate(")
        println(io, "    SpecialFunctions.$fname,")
        println(io, "    $df1_s,")
        println(io, "    $df2_s,")
        println(io, "    $ddf11_s,")
        println(io, "    $ddf12_s,")
        println(io, "    $ddf22_s,")
        println(io, ")")
    end

    println(io)
    println(io, "end # module ExaModelsSpecialFunctions")

    return String(take!(io))
end

# ============================================================================
# Main
# ============================================================================

function main()
    content = generate()

    if "--write" in ARGS
        outfile = joinpath(dirname(@__DIR__), "ext", "ExaModelsSpecialFunctions.jl")
        write(outfile, content)
        println("Wrote $(length(content)) bytes to $outfile")
    else
        print(content)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
