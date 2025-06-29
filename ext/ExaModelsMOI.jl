module ExaModelsMOI

import ExaModels: ExaModels, NLPModels, SolverCore

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const MOIB = MathOptInterface.Bridges

const SUPPORTED_FUNC_TYPE = [
    MOI.ScalarAffineFunction,
    MOI.ScalarQuadraticFunction,
    MOI.ScalarNonlinearFunction,
    MOI.VariableIndex,
]
const SUPPORTED_SET_TYPE = [
    MOI.GreaterThan,
    MOI.LessThan,
    MOI.EqualTo,
    MOI.Interval,
]
const SUPPORTED_FUNC_TYPE_UNION = Union{SUPPORTED_FUNC_TYPE...}
const SUPPORTED_SET_TYPE_UNION = Union{SUPPORTED_SET_TYPE...}

"""
    Abstract data structure for storing expression tree and data arrays
"""
abstract type AbstractBin end

struct Bin{E,P,I} <: AbstractBin
    head::E
    data::P
    inner::I
end

struct BinNull <: AbstractBin end

function update_bin!(bin, e, p)
    if _update_bin!(bin, e, p) # if update succeeded, return the original bin
        return bin
    else # if update has failed, return a new bin
        return Bin(e, [p], bin)
    end
end
function _update_bin!(bin::Bin{E,P,I}, e, p) where {E,P,I}
    if e == bin.head && p isa eltype(bin.data)
        push!(bin.data, p)
        return true
    else
        return _update_bin!(bin.inner, e, p)
    end
end
function _update_bin!(::BinNull, e, p)
    return false
end

float_type(::Type{MOI.GreaterThan{T}}) where {T} = T
float_type(::Type{MOI.LessThan{T}}) where {T} = T
float_type(::Type{MOI.EqualTo{T}}) where {T} = T
float_type(::Type{MOI.Interval{T}}) where {T} = T

function ExaModels.ExaModel(moim::MOI.ModelLike; backend = nothing, prod = false)
    minimize = MOI.get(moim, MOI.ObjectiveSense()) === MOI.MIN_SENSE

    variables = MOI.get(moim, MOI.ListOfVariableIndices())
    con_types = MOI.get(moim, MOI.ListOfConstraintTypesPresent())

    set_float_types = Set()
    for (F,S) in con_types
        if !(F <: SUPPORTED_FUNC_TYPE_UNION)
            error("Found unsupported function type $F.")
        end
        if !(S <: SUPPORTED_SET_TYPE_UNION)
            error("Found unsupported set type $S.")
        end
        push!(set_float_types, float_type(S))
    end

    T = if isempty(set_float_types)
        error("Cannot deduce float type from the constraints")
    elseif length(set_float_types) > 1
        error("All constraints must have the same float type. Found $set_float_types")
    else
        first(set_float_types)
    end

    # create exacore
    c = ExaModels.ExaCore(T; backend = backend, minimize = minimize)

    # variables
    nvar = length(variables)
    x0 = zeros(T, nvar)
    lvar = fill(-T(Inf), nvar)
    uvar = fill(T(Inf), nvar)

    for ci in MOI.get(moim, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{T}}())
        vi = MOI.get(moim, MOI.ConstraintFunction(), ci)
        @assert vi isa MOI.VariableIndex
        lvar[vi.value] = MOI.get(moim, MOI.ConstraintSet(), ci).lower
    end
    for ci in MOI.get(moim, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{T}}())
        vi = MOI.get(moim, MOI.ConstraintFunction(), ci)
        @assert vi isa MOI.VariableIndex
        uvar[vi.value] = MOI.get(moim, MOI.ConstraintSet(), ci).upper
    end

    if !isempty(filter(
        FS -> (FS[1] <: MOI.VariableIndex) && !(
            FS[2] <: Union{MOI.GreaterThan{T},MOI.LessThan{T}}
        ), con_types
    ))
        # this is where parameters would error
        error("Found unsupported variable index constraint")
    end

    for vi in variables
        start = MOI.get(moim, MOI.VariablePrimalStart(), vi)
        isnothing(start) && continue
        x0[vi.value] = start
    end

    v = ExaModels.variable(c, nvar; start = x0, lvar = lvar, uvar = uvar)

    # objective
    obj_type = MOI.get(moim, MOI.ObjectiveFunctionType())
    if !(obj_type <: SUPPORTED_FUNC_TYPE_UNION)
        error("Objective function of type $obj_type is not supported")
    end

    obj_bin = exafy_obj(
        MOI.get(moim, MOI.ObjectiveFunction{obj_type}()),
        BinNull()
    )

    build_objective(c, obj_bin)

    # constraint
    bin = BinNull()
    offset = 0
    lcon = similar(x0, 0)
    ucon = similar(x0, 0)

    for F in SUPPORTED_FUNC_TYPE
        F <: MOI.VariableIndex && continue
        FT = F <: MOI.ScalarNonlinearFunction ? F : F{T}
        for S in SUPPORTED_SET_TYPE
            ST = S{T}
            cis = MOI.get(moim, MOI.ListOfConstraintIndices{FT,ST}())
            bin, offset = exafy_con(moim, cis, bin, offset, lcon, ucon)
        end
    end

    y0 = fill!(similar(lcon), zero(T))
    cons = ExaModels.constraint(c, offset; start = y0, lcon = lcon, ucon = ucon)
    build_constraint!(c, cons, bin)

    return ExaModels.ExaModel(c; prod = prod)
end

function _exafy_con(i, c::C, bin, offset; pos = true) where {C<:MOI.ScalarAffineFunction}
    for mm in c.terms
        e, p = _exafy(mm)
        e = pos ? e : -e
        bin = update_bin!(
            bin,
            ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) => e,
            (p..., offset + i.value),
        ) # augment data with constraint index
    end
    bin = update_bin!(bin, ExaModels.Null(c.constant), (1,))
    return bin, offset
end
function _exafy_con(i, c::C, bin, offset; pos = true) where {C<:MOI.ScalarQuadraticFunction}
    for mm in c.affine_terms
        e, p = _exafy(mm)
        e = pos ? e : -e
        bin = update_bin!(
            bin,
            ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) => e,
            (p..., offset + i.value),
        ) # augment data with constraint index
    end
    for mm in c.quadratic_terms
        e, p = _exafy(mm)
        e = pos ? e : -e
        bin = update_bin!(
            bin,
            ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) => e,
            (p..., offset + i.value),
        ) # augment data with constraint index
    end
    bin = update_bin!(bin, ExaModels.Null(c.constant), (1,))
    return bin, offset
end
function _exafy_con(i, c::C, bin, offset; pos = true) where {C<:MOI.ScalarNonlinearFunction}
    if c.head == :+
        for mm in c.args
            bin, offset = _exafy_con(i, mm, bin, offset)
        end
        # elseif c.head == :-
        #     bin, offset = _exafy_con(i, c.args[1], bin, offset)
        #     bin, offset = _exafy_con(i, c.args[2], bin, offset; pos = false)
    else
        e, p = _exafy(c)
        e = pos ? e : -e
        bin = update_bin!(
            bin,
            ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) => e,
            (p..., offset + i.value),
        ) # augment data with constraint index
    end
    return bin, offset
end
function _exafy_con(i, c::C, bin, offset; pos = true) where {C<:Real}
    e =
        pos ? ExaModels.ParIndexed(ExaModels.ParSource(), 1) :
        -ExaModels.ParIndexed(ExaModels.ParSource(), 1)
    bin = update_bin!(
        bin,
        ExaModels.ParIndexed(ExaModels.ParSource(), 2) => 0 * ExaModels.Var(1) + e,
        (c, offset + i.value),
    )

    return bin, offset
end

function exafy_con(moim, cons::V, bin, offset, lcon, ucon) where {V<:Vector{<:MOI.ConstraintIndex}}
    l = length(cons)

    resize!(lcon, offset + l)
    resize!(ucon, offset + l)
    for ci in cons
        func = MOI.get(moim, MOI.ConstraintFunction(), ci)
        set = MOI.get(moim, MOI.ConstraintSet(), ci)
        _exafy_con_update_vector(ci, set, lcon, ucon, offset)
        bin, offset = _exafy_con(ci, func, bin, offset)
    end
    return bin, (offset += l)
end

function _exafy_con_update_vector(i, e::MOI.Interval{T}, lcon, ucon, offset) where {T}
    lcon[offset+i.value] = e.lower
    ucon[offset+i.value] = e.upper
end

function _exafy_con_update_vector(i, e::MOI.LessThan{T}, lcon, ucon, offset) where {T}
    lcon[offset+i.value] = -Inf
    ucon[offset+i.value] = e.upper
end

function _exafy_con_update_vector(i, e::MOI.GreaterThan{T}, lcon, ucon, offset) where {T}
    ucon[offset+i.value] = Inf
    lcon[offset+i.value] = e.lower
end

function _exafy_con_update_vector(i, e::MOI.EqualTo{T}, lcon, ucon, offset) where {T}
    lcon[offset+i.value] = e.value
    ucon[offset+i.value] = e.value
end


function build_constraint!(c, cons, bin)
    build_constraint!(c, cons, bin.inner)
    ExaModels.constraint!(c, cons, bin.head, bin.data)
end

function build_constraint!(c, cons, ::BinNull) end

function build_objective(c, bin)
    build_objective(c, bin.inner)
    ExaModels.objective(c, bin.head, bin.data)
end

function build_objective(c, ::BinNull) end

function exafy_obj(o::Nothing, bin)
    return bin
end

function exafy_obj(o::MOI.VariableIndex, bin)
    e, p = _exafy(o)
    return update_bin!(bin, e, p)
end

function exafy_obj(o::MOI.ScalarQuadraticFunction{T}, bin) where {T}
    for m in o.affine_terms
        e, p = _exafy(m)
        bin = update_bin!(bin, e, p)
    end
    for m in o.quadratic_terms
        e, p = _exafy(m)
        bin = update_bin!(bin, e, p)
    end

    return update_bin!(bin, ExaModels.Null(o.constant), (1,))
end

function exafy_obj(o::MOI.ScalarAffineFunction{T}, bin) where {T}
    for m in o.terms
        e, p = _exafy(m)
        bin = update_bin!(bin, e, p)
    end

    return update_bin!(bin, ExaModels.Null(o.constant), (1,))
end

function exafy_obj(o::MOI.ScalarNonlinearFunction, bin)
    constant = 0.0
    if o.head == :+
        for m in o.args
            if m isa MOI.ScalarAffineFunction
                for mm in m.affine_terms
                    e, p = _exafy(mm)
                    bin = update_bin!(bin, e, p)
                end
            elseif m isa MOI.ScalarQuadraticFunction
                for mm in m.affine_terms
                    e, p = _exafy(mm)
                    bin = update_bin!(bin, e, p)
                end
                for mm in m.quadratic_terms
                    e, p = _exafy(mm)
                    bin = update_bin!(bin, e, p)
                end
                constant += m.constant
            else
                e, p = _exafy(m)
                bin = update_bin!(bin, e, p)
            end
        end
    else
        e, p = _exafy(o)
        bin = update_bin!(bin, e, p)
    end

    return update_bin!(bin, ExaModels.Null(constant), (1,)) # TODO see if this can be empty tuple
end

function _exafy(v::MOI.VariableIndex, p = ())
    i = ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)
    return ExaModels.Var(i), (p..., v.value)
end

function _exafy(i::R, p) where {R<:Real}
    return ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1), (p..., i)
end

function _exafy(e::MOI.ScalarNonlinearFunction, p = ())
    return op(e.head)((
        begin
            c, p = _exafy(e, p)
            c
        end for e in e.args
    )...), p
end

function _exafy(e::MOI.ScalarAffineFunction{T}, p = ()) where {T}
    ec = if !isempty(e.terms)
        sum(begin
            c1, p = _exafy(term, p)
            c1
        end for term in e.terms) +
        ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)
    else
        ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)
    end

    return ec, (p..., e.constant)
end

function _exafy(e::MOI.ScalarAffineTerm{T}, p = ()) where {T}
    c1, p = _exafy(e.variable, p)
    return *(c1, ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)),
    (p..., e.coefficient)
end

function _exafy(e::MOI.ScalarQuadraticFunction{T}, p = ()) where {T}
    t = ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)
    p = (p..., e.constant)

    if !isempty(e.affine_terms)
        t += sum(begin
            c1, p = _exafy(term, p)
            c1
        end for term in e.affine_terms)
    end

    if !isempty(e.quadratic_terms)
        t += sum(begin
            c1, p = _exafy(term, p)
            c1
        end for term in e.quadratic_terms)
    end

    return t, p
end

function _exafy(e::MOI.ScalarQuadraticTerm{T}, p = ()) where {T}

    if e.variable_1 == e.variable_2
        v, p = _exafy(e.variable_1, p)
        return ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) * abs2(v),
        (p..., e.coefficient / 2) # it seems that MOI assumes this by default
    else
        v1, p = _exafy(e.variable_1, p)
        v2, p = _exafy(e.variable_2, p)
        return ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) * v1 * v2,
        (p..., e.coefficient)
    end
end

# eval can be a performance killer -- we want to explicitly include symbols for frequently used operations.
function op(s::Symbol)
    # uni/multi
    if     s === :+            return + #
    elseif s === :-            return - #
    # multi
    elseif s === :*            return *
    elseif s === :^            return ^
    elseif s === :/            return /
    # uni
    elseif s === :abs          return abs #
    elseif s === :sign         return sign
    elseif s === :sqrt         return sqrt #
    elseif s === :cbrt         return cbrt #
    elseif s === :abs2         return abs2 #
    elseif s === :inv          return inv #
    elseif s === :log          return log #
    elseif s === :log10        return log10 #
    elseif s === :log2         return log2 #
    elseif s === :log1p        return log1p #
    elseif s === :exp          return exp #
    elseif s === :exp2         return exp2 #
    elseif s === :expm1        error("expm1 not supported")
    # trig
    elseif s === :sin          return sin #
    elseif s === :cos          return cos #
    elseif s === :tan          return tan #
    elseif s === :sec          return sec #
    elseif s === :csc          return csc #
    elseif s === :cot          return cot #
    elseif s === :sind         return sind #
    elseif s === :cosd         return cosd #
    elseif s === :tand         return tand #
    elseif s === :secd         return secd #
    elseif s === :cscd         return cscd #
    elseif s === :cotd         return cotd #
    elseif s === :asin         return asin #
    elseif s === :acos         return acos #
    elseif s === :atan         return atan #
    elseif s === :asec         error("asec not supported")
    elseif s === :acsc         error("acsc not supported")
    elseif s === :acot         return acot #
    elseif s === :asind        return asind
    elseif s === :acosd        return acosd
    elseif s === :atand        return atand #
    elseif s === :asecd        return asecd
    elseif s === :acscd        return acscd
    elseif s === :acotd        return acotd #
    elseif s === :sinh         return sinh #
    elseif s === :cosh         return cosh #
    elseif s === :tanh         return tanh #
    elseif s === :sech         return sech #
    elseif s === :csch         return csch #
    elseif s === :coth         return coth #
    elseif s === :asinh        return asinh #
    elseif s === :acosh        return acosh #
    elseif s === :atanh        return atanh #
    elseif s === :asech        error("asech not supported")
    elseif s === :acsch        error("acsch not supported")
    elseif s === :acoth        return acoth #
    # special (commented will use `eval` which would succeed if SpecialFunctions is loaded)
    elseif s === :deg2rad      error("deg2rad not supported")
    elseif s === :rad2deg      error("rad2deg not supported")
    # elseif s === :erf          error("erf not supported")
    # elseif s === :erfinv       error("erfinv not supported")
    # elseif s === :erfc         error("erfc not supported")
    # elseif s === :erfcinv      error("erfcinv not supported")
    # elseif s === :erfi         error("erfi not supported")
    # elseif s === :gamma        error("gamma not supported")
    elseif s === :lgamma       error("lgamma not supported")
    # elseif s === :digamma      error("digamma not supported")
    # elseif s === :invdigamma   error("invdigamma not supported")
    # elseif s === :trigamma     error("trigamma not supported")
    # elseif s === :airyai       error("airyai not supported")
    # elseif s === :airybi       error("airybi not supported")
    # elseif s === :airyaiprime  error("airyaiprime not supported")
    # elseif s === :airybiprime  error("airybiprime not supported")
    # elseif s === :besselj0     error("besselj0 not supported")
    # elseif s === :besselj1     error("besselj1 not supported")
    # elseif s === :bessely0     error("bessely0 not supported")
    # elseif s === :bessely1     error("bessely1 not supported")
    # elseif s === :erfcx        error("erfcx not supported")
    # elseif s === :dawson       error("dawson not supported")
    
    # not in MOI
    elseif s === :exp10        return exp10
    elseif s === :beta         return beta
    elseif s === :logbeta      return logbeta
    else
        return eval(s)
    end
end


# struct EmptyOptimizer{B}
#     backend::B
# end
mutable struct Optimizer{B,S} <: MOI.AbstractOptimizer
    solver::S
    backend::B
    model::Union{Nothing,ExaModels.ExaModel}
    result::Any
    solve_time::Float64
    options::Dict{Symbol,Any}
end

MOI.is_empty(model::Optimizer) = isnothing(model.model)

const _FUNCTIONS = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
}
const _SETS = Union{MOI.GreaterThan{Float64},MOI.LessThan{Float64},MOI.EqualTo{Float64}}
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VariableIndex,_FUNCTIONS}},
    ::Type{<:_SETS},
)
    return true
end
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex,<:_FUNCTIONS}},
)
    return true
end
function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

function ExaModels.Optimizer(solver, backend = nothing; kwargs...)
    return Optimizer(solver, backend, nothing, nothing, 0.0, Dict{Symbol,Any}(kwargs...))
end

function MOI.empty!(model::ExaModelsMOI.Optimizer)
    model.model = nothing
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    dest.model = ExaModels.ExaModel(src; backend = dest.backend)
    return MOIU.identity_index_map(src)
end

function MOI.optimize!(optimizer::Optimizer)
    optimizer.solve_time = @elapsed begin
        result = optimizer.solver(optimizer.model; optimizer.options...)
        optimizer.result = (
            objective = result.objective,
            solution = Array(result.solution),
            multipliers = Array(result.multipliers),
            multipliers_L = Array(result.multipliers_L),
            multipliers_U = Array(result.multipliers_U),
            status = result.status,
        )
    end

    return optimizer
end

MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus) =
    ExaModels.termination_status_translator(optimizer.solver, optimizer.result.status)
MOI.get(model::Optimizer, attr::Union{MOI.PrimalStatus,MOI.DualStatus}) =
    ExaModels.result_status_translator(model.solver, model.result.status)

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    # MOI.throw_if_not_valid(model, vi)
    # if _is_parameter(vi)
    #     p = model.parameters[vi]
    #     return model.nlp_model[p]
    # end
    return model.result.solution[vi.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    # MOI.throw_if_not_valid(model, ci)
    s = -1.0
    return s * model.result.multipliers[ci.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    # MOI.throw_if_not_valid(model, ci)
    rc = model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
    return min(0.0, rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    # MOI.throw_if_not_valid(model, ci)
    rc = model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
    return max(0.0, rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    # MOI.throw_if_not_valid(model, ci)
    rc = model.result.multipliers_L[ci.value] - model.result.multipliers_U[ci.value]
    return rc
end


function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.result !== nothing) ? 1 : 0
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    # scale = (model.sense == MOI.MAX_SENSE) ? -1 : 1
    # return scale * model.result.objective
    return model.result.objective
end

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time
MOI.get(
    model::Optimizer,
    ::MOI.SolverName,
) = "$(string(model.solver)) running with ExaModels"

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[Symbol(p.name)] = value
    # No need to reset model.solver because this gets handled in optimize!.
    return
end

end # module
