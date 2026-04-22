module ExaModelsGenOpt

import ExaModels
import GenOpt
import GenOpt: FunctionGenerator, SumGenerator, ContiguousArrayOfVariables, IteratorIndex, Iterator
import MathOptInterface as MOI

# Mark GenOpt function types as extension types
ExaModels.is_extension_type(::Type{<:FunctionGenerator}) = true
ExaModels.is_extension_type(::Type{<:SumGenerator}) = true

# Handle SumGenerator in objective expressions
function ExaModels.exafy_extension_obj_arg(m::SumGenerator, var_to_idx)
    return _exagen(m.func, m.iterators, var_to_idx)
end

# Hook to process FunctionGenerator constraints after standard constraints
function ExaModels.copy_extra_constraints!(c, moim, var_to_idx, con_to_idx, T)
    con_types = MOI.get(moim, MOI.ListOfConstraintTypesPresent())
    for (F, S) in con_types
        F <: FunctionGenerator || continue
        cis = MOI.get(moim, MOI.ListOfConstraintIndices{F, S}())
        c = _copy_generator_constraints!(c, moim, cis, var_to_idx, con_to_idx, T, S)
    end
    return c
end

function _copy_generator_constraints!(c, moim, cis, var_to_idx, con_to_idx, T, ::Type{S}) where {S}
    for ci in cis
        func = MOI.get(moim, MOI.ConstraintFunction(), ci)
        set = MOI.get(moim, MOI.ConstraintSet(), ci)
        con_to_idx[ci] = c.ncon
        expr, pars = _exagen(func.func, func.iterators, var_to_idx)
        c, _ = ExaModels.add_con(c, expr for p in pars; lcon = _lower_bounds(set, T), ucon = _upper_bounds(set, T))
    end
    return c
end

# Convert GenOpt expression trees to ExaModels format

exagen(α::Number, _, _) = α

function exagen(f::MOI.ScalarNonlinearFunction, offsets, var_to_idx)
    if f.head == :getindex
        v = f.args[1]
        if v isa ContiguousArrayOfVariables
            idx = exagen(f.args[2], offsets, var_to_idx)
            # Translate MOI-space offset to ExaModels-space offset using var_to_idx
            first_moi_vi = MOI.VariableIndex(v.offset + 1)
            exa_offset = var_to_idx[first_moi_vi].idx - 1
            if !iszero(exa_offset)
                idx = exa_offset + idx
            end
            cp = cumprod(v.size)
            for i in 3:length(f.args)
                idx += cp[i - 2] * (exagen(f.args[i], offsets, var_to_idx) - 1)
            end
            return ExaModels.Var(idx)
        elseif v isa IteratorIndex
            @assert length(f.args) == 2
            @assert f.args[2] isa Integer
            if isnothing(offsets)
                @assert isone(f.args[2])
                return ExaModels.DataSource()
            else
                return ExaModels.DataIndexed(ExaModels.DataSource(), offsets[v.value] + f.args[2])
            end
        else
            error("Unexpected the first operand of `getindex` to be of type `$(typeof(v))`")
        end
    else
        return ExaModels.op(f.head)((exagen(e, offsets, var_to_idx) for e in f.args)...)
    end
end

function _exagen(func::MOI.ScalarNonlinearFunction, iterators, var_to_idx)
    lengths = map(it -> length(first(it.values)), iterators)
    if length(lengths) == 1 && lengths[] == 1
        cs = nothing
        pars = only.(iterators[].values)
    else
        cs = [0; cumsum(lengths)[1:(end - 1)]]
        pars = vec(
            map(Base.Iterators.ProductIterator(ntuple(i -> iterators[i].values, length(iterators)))) do I
                reduce((i, j) -> tuple(i..., j...), I)
            end
        )
    end
    expr = exagen(func, cs, var_to_idx)
    return expr, pars
end

# Bound helpers for vector sets used by FunctionGenerator constraints
_lower_bounds(::Union{MOI.Zeros, MOI.Nonnegatives}, T) = zero(T)
_lower_bounds(::MOI.Nonpositives, T) = typemin(T)
_upper_bounds(::Union{MOI.Zeros, MOI.Nonpositives}, T) = zero(T)
_upper_bounds(::MOI.Nonnegatives, T) = typemax(T)

end # module
