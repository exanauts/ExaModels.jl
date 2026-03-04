module ExaModelsIpopt

import ExaModels
import GenOpt

function copy_generator_constraints!(c, moim, cis, var_to_idx, con_to_idx, T, ::Type{F}, ::Type{S}) where {F<:GenOpt.FunctionGenerator}
    cis = MOI.get(moim, MOI.ListOfConstraintIndices{F,S}())
    # FIXME we assume that `var_to_idx` is the identity
    for ci in cis
        func = MOI.get(moim, MOI.ConstraintFunction(), ci)
        set = MOI.get(moim, MOI.ConstraintSet(), ci)
        con_to_idx[ci] = c.ncon
        expr, pars = _exagen(func.func, func.iterators)
        ExaModels.constraint(c, expr, pars; lcon = _lower_bounds(set, T), ucon = _upper_bounds(set, T))
    end
end

end
