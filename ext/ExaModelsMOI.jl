module ExaModelsMOI

import ExaModels: ExaModels, NLPModels, SolverCore

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const MOIB = MathOptInterface.Bridges

const SUPPORTED_FUNC_TYPE{T} = Union{
    MOI.ScalarAffineFunction{T},
    MOI.ScalarQuadraticFunction{T},
    MOI.ScalarNonlinearFunction,
}
const SUPPORTED_FUNC_TYPE_WITH_VAR{T} = Union{SUPPORTED_FUNC_TYPE{T},MOI.VariableIndex}
const SUPPORTED_FUNC_SET_TYPE{T} =
    Union{MOI.GreaterThan{T},MOI.LessThan{T},MOI.EqualTo{T},MOI.Interval{T}}
const SUPPORTED_VAR_SET_TYPE{T} =
    Union{MOI.GreaterThan{T},MOI.LessThan{T},MOI.EqualTo{T},MOI.Parameter{T}}
const PARAMETER_INDEX_THRESHOLD = Int64(4_611_686_018_427_387_904) # div(typemax(Int64),2)+1
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

function check_supported(T, moim)
    con_types = MOI.get(moim, MOI.ListOfConstraintTypesPresent())
    for (F, S) in con_types
        !(F <: SUPPORTED_FUNC_TYPE_WITH_VAR) && error("Unsupported function type $F.")
        if F <: MOI.VariableIndex
            !(S <: SUPPORTED_VAR_SET_TYPE) &&
                error("Unsupported variable index constraint $F in $S")
        else
            !(S <: SUPPORTED_FUNC_SET_TYPE) && error("Unsupported set type $S")
        end
    end

    obj_type = MOI.get(moim, MOI.ObjectiveFunctionType())
    !(obj_type <: SUPPORTED_FUNC_TYPE_WITH_VAR) &&
        error("Unsupported objective function type $obj_type.")

    obj_sense = MOI.get(moim, MOI.ObjectiveSense())
    !(obj_sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)) &&
        error("Unsupported objective sense $obj_sense.")
    return obj_sense === MOI.MIN_SENSE
end

function ExaModels.ExaModel(
    moim::MOI.ModelLike;
    backend = nothing,
    prod = false,
    T = Float64,
)
    c, _ = to_exacore(moim; backend = backend, T = T)
    return ExaModels.ExaModel(c; prod = prod)
end

function to_exacore(moim::MOI.ModelLike; backend = nothing, T = Float64)
    minimize = check_supported(T, moim)

    c = ExaModels.ExaCore(T; backend = backend, minimize = minimize)

    var_to_idx = copy_variables!(c, moim, T)
    con_to_idx = copy_constraints!(c, moim, var_to_idx, T)
    copy_objective!(c, moim, var_to_idx)

    return c, (var_to_idx, con_to_idx)
end

function fill_variable_bounds!(moim, lvar, uvar, var_to_idx, T)
    for ci in
        MOI.get(moim, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{T}}())
        vi = MOI.get(moim, MOI.ConstraintFunction(), ci)
        lvar[var_to_idx[vi]] = MOI.get(moim, MOI.ConstraintSet(), ci).lower
    end
    for ci in
        MOI.get(moim, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{T}}())
        vi = MOI.get(moim, MOI.ConstraintFunction(), ci)
        uvar[var_to_idx[vi]] = MOI.get(moim, MOI.ConstraintSet(), ci).upper
    end
    for ci in MOI.get(moim, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.EqualTo{T}}())
        vi = MOI.get(moim, MOI.ConstraintFunction(), ci)
        fixed_val = MOI.get(moim, MOI.ConstraintSet(), ci).value
        lvar[var_to_idx[vi]] = fixed_val
        uvar[var_to_idx[vi]] = fixed_val
    end
end

function fill_variable_start!(moim, x0, param_vis)
    var_to_idx = Dict{MOI.VariableIndex,Int}()
    for (i, vi) in enumerate(MOI.get(moim, MOI.ListOfVariableIndices()))
        vi ∈ param_vis && continue
        var_to_idx[vi] = i
        start = MOI.get(moim, MOI.VariablePrimalStart(), vi)
        isnothing(start) && continue
        x0[i] = start
    end
    return var_to_idx
end

function _get_parameters(moim::MOI.ModelLike, T)
    cis = MOI.get(moim, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Parameter{T}}())
    parameters = Vector{Tuple{MOI.VariableIndex,MOI.Parameter{T}}}()
    for ci in cis
        vi = MOI.get(moim, MOI.ConstraintFunction(), ci)
        set = MOI.get(moim, MOI.ConstraintSet(), ci)
        push!(parameters, (vi, set))
    end
    sort!(parameters, by = x -> x[1].value)
    return parameters
end


function copy_variables!(c, moim, T)
    nvarpar = MOI.get(moim, MOI.NumberOfVariables())
    parameters = _get_parameters(moim, T)
    npar = length(parameters)
    nvar = nvarpar - npar

    x0 = zeros(T, nvar)
    var_to_idx = fill_variable_start!(moim, x0, first.(parameters))

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    fill_variable_bounds!(moim, lvar, uvar, var_to_idx, T)

    ExaModels.variable(c, nvar; start = x0, lvar = lvar, uvar = uvar)

    varpar_to_idx = Dict()
    for (vi, i) in var_to_idx
        varpar_to_idx[vi] = (type = :variable, idx = i)
    end

    if npar > 0
        p0 = zeros(T, npar)
        for (i, (vi, set)) in enumerate(parameters)
            p0[i] = T(set.value)
            varpar_to_idx[vi] = (type = :parameter, idx = i)
        end
        ExaModels.parameter(c, p0)
    end

    return varpar_to_idx
end

function copy_objective!(c, moim, var_to_idx)
    obj_type = MOI.get(moim, MOI.ObjectiveFunctionType())

    bin = BinNull()
    bin = exafy_obj(MOI.get(moim, MOI.ObjectiveFunction{obj_type}()), bin, var_to_idx)

    build_objective!(c, bin)
end

function copy_constraints!(c, moim, var_to_idx, T)
    bin = BinNull()
    offset = 0
    lcon = zeros(T, 0)
    ucon = zeros(T, 0)
    y0 = zeros(T, 0)
    con_to_idx = Dict{MOI.ConstraintIndex,Int}()

    con_types = MOI.get(moim, MOI.ListOfConstraintTypesPresent())
    for (F, S) in con_types
        F <: MOI.VariableIndex && continue
        cis = MOI.get(moim, MOI.ListOfConstraintIndices{F,S}())
        bin, offset =
            exafy_con(moim, cis, bin, offset, lcon, ucon, y0, var_to_idx, con_to_idx)
    end
    cons = ExaModels.constraint(c, offset; start = y0, lcon = lcon, ucon = ucon)
    build_constraint!(c, cons, bin)

    return con_to_idx
end

function _exafy_con(
    i,
    c::C,
    bin,
    var_to_idx,
    con_to_idx;
    pos = true,
) where {C<:MOI.ScalarAffineFunction}
    for mm in c.terms
        e, p = _exafy(mm, var_to_idx)
        e = pos ? e : -e
        bin = update_bin!(
            bin,
            ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) => e,
            (p..., con_to_idx[i]),
        ) # augment data with constraint index
    end
    bin = update_bin!(bin, ExaModels.Null(c.constant), (1,))
    return bin
end
function _exafy_con(
    i,
    c::C,
    bin,
    var_to_idx,
    con_to_idx;
    pos = true,
) where {C<:MOI.ScalarQuadraticFunction}
    for mm in c.affine_terms
        e, p = _exafy(mm, var_to_idx)
        e = pos ? e : -e
        bin = update_bin!(
            bin,
            ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) => e,
            (p..., con_to_idx[i]),
        ) # augment data with constraint index
    end
    for mm in c.quadratic_terms
        e, p = _exafy(mm, var_to_idx)
        e = pos ? e : -e
        bin = update_bin!(
            bin,
            ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) => e,
            (p..., con_to_idx[i]),
        ) # augment data with constraint index
    end
    bin = update_bin!(bin, ExaModels.Null(c.constant), (1,))
    return bin
end
function _exafy_con(
    i,
    c::C,
    bin,
    var_to_idx,
    con_to_idx;
    pos = true,
) where {C<:MOI.ScalarNonlinearFunction}
    if c.head == :+
        for mm in c.args
            bin = _exafy_con(i, mm, bin, var_to_idx, con_to_idx)
        end
        # elseif c.head == :-
        #     bin, offset = _exafy_con(i, c.args[1], bin, offset)
        #     bin, offset = _exafy_con(i, c.args[2], bin, offset; pos = false)
    else
        e, p = _exafy(c, var_to_idx)
        e = pos ? e : -e
        bin = update_bin!(
            bin,
            ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) => e,
            (p..., con_to_idx[i]),
        ) # augment data with constraint index
    end
    return bin
end
function _exafy_con(i, c::C, bin, var_to_idx, con_to_idx; pos = true) where {C<:Real}
    e =
        pos ? ExaModels.ParIndexed(ExaModels.ParSource(), 1) :
        -ExaModels.ParIndexed(ExaModels.ParSource(), 1)
    bin = update_bin!(
        bin,
        ExaModels.ParIndexed(ExaModels.ParSource(), 2) => 0 * ExaModels.Var(1) + e,
        (c, con_to_idx[i]),
    )

    return bin
end

function exafy_con(
    moim,
    cons::V,
    bin,
    offset,
    lcon,
    ucon,
    y0,
    var_to_idx,
    con_to_idx,
) where {V<:Vector{<:MOI.ConstraintIndex}}
    l = length(cons)

    resize!(lcon, offset + l)
    resize!(ucon, offset + l)
    resize!(y0, offset + l)
    for (i, ci) in enumerate(cons)
        func = MOI.get(moim, MOI.ConstraintFunction(), ci)
        set = MOI.get(moim, MOI.ConstraintSet(), ci)
        start = MOI.get(moim, MOI.ConstraintPrimalStart(), ci)
        con_to_idx[ci] = offset + i
        _exafy_con_update_start(ci, start, y0, con_to_idx)
        _exafy_con_update_vector(ci, set, lcon, ucon, con_to_idx)
        bin = _exafy_con(ci, func, bin, var_to_idx, con_to_idx)
    end
    return bin, (offset += l)
end

function _exafy_con_update_start(i, start, y0, con_to_idx)
    y0[con_to_idx[i]] = start
end

function _exafy_con_update_start(i, ::Nothing, y0, con_to_idx)
    y0[con_to_idx[i]] = zero(eltype(y0))
end

function _exafy_con_update_vector(i, e::MOI.Interval{T}, lcon, ucon, con_to_idx) where {T}
    lcon[con_to_idx[i]] = e.lower
    ucon[con_to_idx[i]] = e.upper
end

function _exafy_con_update_vector(i, e::MOI.LessThan{T}, lcon, ucon, con_to_idx) where {T}
    lcon[con_to_idx[i]] = -Inf
    ucon[con_to_idx[i]] = e.upper
end

function _exafy_con_update_vector(
    i,
    e::MOI.GreaterThan{T},
    lcon,
    ucon,
    con_to_idx,
) where {T}
    ucon[con_to_idx[i]] = Inf
    lcon[con_to_idx[i]] = e.lower
end

function _exafy_con_update_vector(i, e::MOI.EqualTo{T}, lcon, ucon, con_to_idx) where {T}
    lcon[con_to_idx[i]] = e.value
    ucon[con_to_idx[i]] = e.value
end


function build_constraint!(c, cons, bin)
    build_constraint!(c, cons, bin.inner)
    ExaModels.constraint!(c, cons, bin.head, bin.data)
end

function build_constraint!(c, cons, ::BinNull) end

function build_objective!(c, bin)
    build_objective!(c, bin.inner)
    ExaModels.objective(c, bin.head, bin.data)
end

function build_objective!(c, ::BinNull) end

function exafy_obj(o::Nothing, bin, var_to_idx)
    return bin
end

function exafy_obj(o::MOI.VariableIndex, bin, var_to_idx)
    e, p = _exafy(o, var_to_idx)
    return update_bin!(bin, e, p)
end

function exafy_obj(o::MOI.ScalarQuadraticFunction{T}, bin, var_to_idx) where {T}
    for m in o.affine_terms
        e, p = _exafy(m, var_to_idx)
        bin = update_bin!(bin, e, p)
    end
    for m in o.quadratic_terms
        e, p = _exafy(m, var_to_idx)
        bin = update_bin!(bin, e, p)
    end

    return update_bin!(bin, ExaModels.Null(o.constant), (1,))
end

function exafy_obj(o::MOI.ScalarAffineFunction{T}, bin, var_to_idx) where {T}
    for m in o.terms
        e, p = _exafy(m, var_to_idx)
        bin = update_bin!(bin, e, p)
    end

    return update_bin!(bin, ExaModels.Null(o.constant), (1,))
end

function exafy_obj(o::MOI.ScalarNonlinearFunction, bin, var_to_idx)
    constant = 0.0
    if o.head == :+
        for m in o.args
            if m isa MOI.ScalarAffineFunction
                for mm in m.terms
                    e, p = _exafy(mm, var_to_idx)
                    bin = update_bin!(bin, e, p)
                end
            elseif m isa MOI.ScalarQuadraticFunction
                for mm in m.affine_terms
                    e, p = _exafy(mm, var_to_idx)
                    bin = update_bin!(bin, e, p)
                end
                for mm in m.quadratic_terms
                    e, p = _exafy(mm, var_to_idx)
                    bin = update_bin!(bin, e, p)
                end
                constant += m.constant
            else
                e, p = _exafy(m, var_to_idx)
                bin = update_bin!(bin, e, p)
            end
        end
    else
        e, p = _exafy(o, var_to_idx)
        bin = update_bin!(bin, e, p)
    end

    return update_bin!(bin, ExaModels.Null(constant), (1,)) # TODO see if this can be empty tuple
end

function _exafy(v::MOI.VariableIndex, var_to_idx, p = ())
    i = ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)
    vartype, idx = var_to_idx[v]
    if vartype === :variable
        return ExaModels.Var(i), (p..., idx)
    elseif vartype === :parameter
        return ExaModels.ParameterNode(i), (p..., idx)
    else
        error("Unknown variable type: $vartype")
    end
end

function _exafy(i::R, var_to_idx, p) where {R<:Real}
    return ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1), (p..., i)
end

function _exafy(e::MOI.ScalarNonlinearFunction, var_to_idx, p = ())
    return op(e.head)((
        begin
            c, p = _exafy(e, var_to_idx, p)
            c
        end for e in e.args
    )...), p
end

function _exafy(e::MOI.ScalarAffineFunction{T}, var_to_idx, p = ()) where {T}
    ec = if !isempty(e.terms)
        sum(begin
            c1, p = _exafy(term, var_to_idx, p)
            c1
        end for term in e.terms) +
        ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)
    else
        ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)
    end

    return ec, (p..., e.constant)
end

function _exafy(e::MOI.ScalarAffineTerm{T}, var_to_idx, p = ()) where {T}
    c1, p = _exafy(e.variable, var_to_idx, p)
    return *(c1, ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)),
    (p..., e.coefficient)
end

function _exafy(e::MOI.ScalarQuadraticFunction{T}, var_to_idx, p = ()) where {T}
    t = ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1)
    p = (p..., e.constant)

    if !isempty(e.affine_terms)
        t += sum(begin
            c1, p = _exafy(term, var_to_idx, p)
            c1
        end for term in e.affine_terms)
    end

    if !isempty(e.quadratic_terms)
        t += sum(begin
            c1, p = _exafy(term, var_to_idx, p)
            c1
        end for term in e.quadratic_terms)
    end

    return t, p
end

function _exafy(e::MOI.ScalarQuadraticTerm{T}, var_to_idx, p = ()) where {T}

    if e.variable_1 == e.variable_2
        v, p = _exafy(e.variable_1, var_to_idx, p)
        return ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) * abs2(v),
        (p..., e.coefficient / 2) # it seems that MOI assumes this by default
    else
        v1, p = _exafy(e.variable_1, var_to_idx, p)
        v2, p = _exafy(e.variable_2, var_to_idx, p)
        return ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) * v1 * v2,
        (p..., e.coefficient)
    end
end

# eval can be a performance killer -- we want to explicitly include symbols for frequently used operations.
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
        # special (commented will use `eval` which would succeed if SpecialFunctions is loaded)
    elseif s === :deg2rad
        error("deg2rad not supported")
    elseif s === :rad2deg
        error("rad2deg not supported")
        # elseif s === :erf          error("erf not supported")
        # elseif s === :erfinv       error("erfinv not supported")
        # elseif s === :erfc         error("erfc not supported")
        # elseif s === :erfcinv      error("erfcinv not supported")
        # elseif s === :erfi         error("erfi not supported")
        # elseif s === :gamma        error("gamma not supported")
    elseif s === :lgamma
        error("lgamma not supported")
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

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:SUPPORTED_FUNC_TYPE},
    ::Type{<:SUPPORTED_FUNC_SET_TYPE},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:SUPPORTED_VAR_SET_TYPE},
)
    return true
end
function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{<:SUPPORTED_FUNC_TYPE_WITH_VAR})
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
    core, maps = to_exacore(src; backend = dest.backend)
    dest.model = ExaModels.ExaModel(core; prod = true)
    return _make_index_map(src, maps)
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
    if vi.value > PARAMETER_INDEX_THRESHOLD
        return model.model.θ[vi.value-PARAMETER_INDEX_THRESHOLD]
    else
        return model.result.solution[vi.value]
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:SUPPORTED_FUNC_TYPE,<:SUPPORTED_FUNC_SET_TYPE},
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


_make_index_map(model::MOI.ModelLike, maps) = _make_index_map(model, maps[1], maps[2])
function _make_index_map(model::MOI.ModelLike, var_to_idx, con_to_idx)
    variables = MOI.get(model, MOI.ListOfVariableIndices())
    map = MOI.Utilities.IndexMap()
    for x in variables
        vartype, rawidx = var_to_idx[x]
        if vartype === :variable
            map[x] = typeof(x)(rawidx)
        elseif vartype === :parameter
            map[x] = typeof(x)(rawidx + PARAMETER_INDEX_THRESHOLD)
        else
            error("Unknown variable type $vartype")
        end
    end
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        _make_constraints_map(model, map.con_map[F, S], con_to_idx)
    end
    return map
end
function _make_constraints_map(
    model,
    map::MOI.Utilities.DoubleDicts.IndexDoubleDictInner{F,S},
    con_to_idx,
) where {F,S}
    for c in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
        map[c] = typeof(c)(con_to_idx[c])
    end
    return
end

end # module
