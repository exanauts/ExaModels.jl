# Shared utilities for code-generation scripts in deps/.
# Included (not imported) by generate_functionlist.jl and generate_specialfunctions.jl.

using Symbolics

@variables x x1 x2

# ============================================================================
# Symbolics helpers
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

"""
Ensure spaces around `*` for Runic formatting compliance.
Symbolics.jl emits `a*b`; Runic expects `a * b`.
Two passes handle chains like `a*b*c`.
"""
function _runic_mul_spaces(s::String)
    for _ in 1:2
        s = replace(s, r"(\S)\*(\S)" => s"\1 * \2")
    end
    return s
end

# ============================================================================
# Type-generic constant replacement (Float32 compatibility)
# ============================================================================

"""
Replace known Float64 literal constants in an expression string with
type-generic helper calls, handling implicit multiplication.
"""
function replace_float_constants(s::String, typevar::String, replacements)
    for (literal, template) in replacements
        repl = replace(template, "{V}" => typevar)
        escaped = replace(literal, "." => "\\.")
        # Literal followed by word char or '(' needs explicit ' * '
        s = replace(s, Regex(escaped * raw"(?=[a-zA-Z_(])") => repl * " * ")
        # Remaining occurrences (before operators, ), end of string)
        s = replace(s, literal => repl)
    end
    return s
end

"""
Replace bare integer expression bodies and ifelse integer arguments
with type-generic versions for Float32 compatibility.
"""
function replace_integer_constants(s::String, typevar::String)
    # Bare integer body -> typed constant
    if occursin(r"^-?\d+$", s)
        return _typed_int(s, typevar)
    end
    # Integer arguments in ifelse(cond, int, int) calls
    s = replace(
        s, r"ifelse\((.+?), (-?\d+), (-?\d+)\)" => m -> begin
            caps = match(r"ifelse\((.+?), (-?\d+), (-?\d+)\)", m).captures
            cond, v1, v2 = caps
            "ifelse($(cond), $(_typed_int(v1, typevar)), $(_typed_int(v2, typevar)))"
        end
    )
    return s
end

function _typed_int(s::AbstractString, typevar::String)
    n = parse(Int, s)
    n == 0 && return "zero($typevar)"
    n == 1 && return "one($typevar)"
    n == -1 && return "-one($typevar)"
    return "oftype($typevar, $s)"
end

function sym2lambda(expr, float_replacements, vars::Symbol...)
    s = string(expr)
    typevar = string(vars[1])
    s = replace_float_constants(s, typevar, float_replacements)
    s = replace_integer_constants(s, typevar)
    s = _runic_mul_spaces(s)
    if length(vars) == 1
        return "$(vars[1]) -> $s"
    else
        vstr = join(vars, ", ")
        return "($vstr) -> $s"
    end
end
