# Extension points used by ExaModelsMOI and ExaModelsGenOpt extensions

"""
    copy_extra_constraints!(c, moim, var_to_idx, con_to_idx, T)

Hook for extensions to add extra constraint types after standard MOI constraints
are processed. Default is a no-op, defined in ExaModelsMOI.
"""
function copy_extra_constraints! end

"""
    is_extension_type(::Type{F}) -> Bool

Return `true` if `F` is a function type handled by an extension.
Used by `check_supported` and `supports_constraint` to whitelist extension types.
"""
function is_extension_type end
is_extension_type(::Type) = false

"""
    exafy_extension_obj_arg(m, var_to_idx) -> Union{Nothing, Tuple}

Try to convert an objective function argument `m` to an `(expr, pars)` tuple
for ExaModels. `var_to_idx` maps `MOI.VariableIndex` to `(type, idx)` named tuples.
Returns `nothing` if the type is not handled by any extension.
"""
function exafy_extension_obj_arg end
exafy_extension_obj_arg(m, var_to_idx) = nothing

"""
    op(s::Symbol)

Map a Symbol to the corresponding Julia function. Used by both ExaModelsMOI
and ExaModelsGenOpt for expression tree conversion.
"""
function op(s::Symbol)
    # uni/multi
    if s === :+
        return +
    elseif s === :-
        return -
        # multi
    elseif s === :*
        return *
    elseif s === :^
        return ^
    elseif s === :/
        return /
        # uni
    elseif s === :abs
        return abs
    elseif s === :sign
        error("sign not supported")
    elseif s === :sqrt
        return sqrt
    elseif s === :cbrt
        return cbrt
    elseif s === :abs2
        return abs2
    elseif s === :inv
        return inv
    elseif s === :log
        return log
    elseif s === :log10
        return log10
    elseif s === :log2
        return log2
    elseif s === :log1p
        return log1p
    elseif s === :exp
        return exp
    elseif s === :exp2
        return exp2
    elseif s === :expm1
        error("expm1 not supported")
        # trig
    elseif s === :sin
        return sin
    elseif s === :cos
        return cos
    elseif s === :tan
        return tan
    elseif s === :sec
        return sec
    elseif s === :csc
        return csc
    elseif s === :cot
        return cot
    elseif s === :sind
        return sind
    elseif s === :cosd
        return cosd
    elseif s === :tand
        return tand
    elseif s === :secd
        return secd
    elseif s === :cscd
        return cscd
    elseif s === :cotd
        return cotd
    elseif s === :asin
        return asin
    elseif s === :acos
        return acos
    elseif s === :atan
        return atan
    elseif s === :asec
        error("asec not supported")
    elseif s === :acsc
        error("acsc not supported")
    elseif s === :acot
        return acot
    elseif s === :asind
        error("asind not supported")
    elseif s === :acosd
        error("acosd not supported")
    elseif s === :atand
        return atand
    elseif s === :asecd
        error("aced not supported")
    elseif s === :acscd
        error("acscd not supported")
    elseif s === :acotd
        return acotd
    elseif s === :sinh
        return sinh
    elseif s === :cosh
        return cosh
    elseif s === :tanh
        return tanh
    elseif s === :sech
        return sech
    elseif s === :csch
        return csch
    elseif s === :coth
        return coth
    elseif s === :asinh
        return asinh
    elseif s === :acosh
        return acosh
    elseif s === :atanh
        return atanh
    elseif s === :asech
        error("asech not supported")
    elseif s === :acsch
        error("acsch not supported")
    elseif s === :acoth
        return acoth
        # special
    elseif s === :deg2rad
        error("deg2rad not supported")
    elseif s === :rad2deg
        error("rad2deg not supported")
    elseif s === :lgamma
        error("lgamma not supported")
        # not in MOI
    elseif s === :exp10
        return exp10
    elseif s === :beta
        return beta
    elseif s === :logbeta
        return logbeta
    else
        return eval(s)
    end
end
