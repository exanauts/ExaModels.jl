module ExaModelsMOI

import ExaModels
import MathOptInterface as MOI

function __init__()
    setglobal!(ExaModels, :Optimizer, Optimizer)
    return
end

const PARAMETER_INDEX_THRESHOLD = div(typemax(Int64), 2) + 1

"""
    struct Bin{E,P}
        head::E
        data::Vector{P}
    end

This struct represents `head(d) for d in data`

`head` will be one of two things:

 1) `DataIndexed() => e`: this means that the generator is a constraint
    augmentation. It maps the row index to an expression. The `e` is a symbolic
    function. This de-duplicates structural constraints, so two constraints with
    the same symbolic form will get automatically added as elements to the
    vector `data`.

 2) `e::ExaModels.AbstractNode`: this means that the generator is part of a
    summation. This is used for the objective function.
"""
struct Bin{E,P}
    head::E
    data::Vector{P}
end

"""
    update_bin!(bin::Vector{Bin}, head, data)

This function loops thorugh the list of `bin` looking for a matching `head`. If
found, it updates the bin in place. Othersise, it appends a new bin.
"""
function update_bin!(
    bins::Vector{Bin},
    head::Union{
        ExaModels.AbstractNode,
        Pair{<:ExaModels.AbstractNode,<:ExaModels.AbstractNode},
    },
    data::Tuple,
)
    for bin in bins
        if _update_bin!(bin, head, data)
            return
        end
    end
    push!(bins, Bin(head, [data]))
    return bins
end

# The types match for `head(data)` to be appended to this bin. We check if the
# head's match with `==`, otherwise we recurse to the next bin.
function _update_bin!(bin::Bin{E,P}, head::E, data::P) where {E,P}
    if head == bin.head
        push!(bin.data, data)
        return true
    end
    return false
end

# The head does not match. We can't update this bin in-place.
_update_bin!(::Bin, ::Any, ::Any) = false

# A method for the objective function. First convert the MOI function `f` into
# an `ExaModels.AbstractNode`, then add that.
function update_bin!(
    bins::Vector{Bin},
    f,
    var_to_idx::Dict{MOI.VariableIndex,Int},
)
    head, data = _exafy(f, (), var_to_idx, nothing)
    return update_bin!(bins, head, data)
end

# A method for adding to a constraint. First convert the MOI function `f` into
# an `ExaModels.AbstractNode`, then add that.
function update_bin!(
    bins::Vector{Bin},
    (row, f)::Pair{Int,F},
    var_to_idx::Dict{MOI.VariableIndex,Int},
) where {F}
    head, data = _exafy(f, (), var_to_idx, nothing)
    e = ExaModels.DataIndexed(ExaModels.DataSource(), length(data) + 1)
    return update_bin!(bins, e => head, (data..., row))
end

# This is a type that lets us dispatch on the difference between `row => expr`
# and `expr`.
abstract type AbstractBin end

# This is an `expr`. It gets added to the objective.
struct ObjectiveBin <:AbstractBin end

# Things are passed through unchanged.
(::ObjectiveBin)(f) = f

# Except objective constants are passed as `Null`.
(::ObjectiveBin)(f::Real) = ExaModels.Null(f)

# This is a `row => expr`. We keep the row in a closure.
struct ConstraintBin <: AbstractBin
    row::Int
end

# When passing, we convert to a pair.
(bin::ConstraintBin)(f) = bin.row => f

# VariableIndices are handled directly.
function update_bin!(
    bins::Vector{Bin},
    fn::AbstractBin,
    f::Union{Real,MOI.VariableIndex},
    var_to_idx::Dict{MOI.VariableIndex,Int},
)
    return update_bin!(bins, fn(f), var_to_idx)
end

# Add the additive terms separately, instead of creating a single +(args...)
# expression.
function update_bin!(
    bins::Vector{Bin},
    fn::AbstractBin,
    f::MOI.ScalarAffineFunction,
    var_to_idx::Dict{MOI.VariableIndex,Int},
)
    for term in f.terms
        update_bin!(bins, fn(term), var_to_idx)
    end
    if !iszero(f.constant)
        update_bin!(bins, fn(f.constant), var_to_idx)
    end
    return bins
end

# Add the additive terms separately, instead of creating a single +(args...)
# expression.
function update_bin!(
    bins::Vector{Bin},
    fn::AbstractBin,
    f::MOI.ScalarQuadraticFunction,
    var_to_idx::Dict{MOI.VariableIndex,Int},
)
    for term in f.affine_terms
        update_bin!(bins, fn(term), var_to_idx)
    end
    for term in f.quadratic_terms
        update_bin!(bins, fn(term), var_to_idx)
    end
    if !iszero(f.constant)
        update_bin!(bins, fn(f.constant), var_to_idx)
    end
    return bins
end

_is_zero(x::Real) = iszero(x)

_is_zero(::Any) = false

function update_bin!(
    bins::Vector{Bin},
    fn::AbstractBin,
    f::MOI.ScalarNonlinearFunction,
    var_to_idx::Dict{MOI.VariableIndex,Int},
)
    if f.head == :- && length(f.args) == 2
        # Optimization: :(x - y) -> :(+(x, -y))
        # This allows additive terms in the left-hand side to be added
        # separately. This is a common case in JuMP because
        # `@constraint(model, lhs <= rhs)` normalizes to `lhs - rhs <= 0`.
        update_bin!(bins, fn, f.args[1], var_to_idx)
        if !_is_zero(f.args[2])
            rhs = MOI.Utilities.operate(-, Float64, f.args[2])
            update_bin!(bins, fn(rhs), var_to_idx)
        end
        return bins
    elseif f.head != :+
        return update_bin!(bins, fn(f), var_to_idx)
    end
    # Optimization: if the expression is a `:+`, add the child arguments as
    # separate terms. This keeps the size of the expressions small for ExaModels.
    constant = 0.0
    for arg in f.args
        if arg isa MOI.ScalarAffineFunction
            for term in arg.terms
                update_bin!(bins, fn(term), var_to_idx)
            end
            constant += arg.constant
        elseif arg isa MOI.ScalarQuadraticFunction
            for term in arg.affine_terms
                update_bin!(bins, fn(term), var_to_idx)
            end
            for term in arg.quadratic_terms
                update_bin!(bins, fn(term), var_to_idx)
            end
            constant += arg.constant
        else
            # This is NOT fn(arg) here because we want to be able to lift any
            # nested `+(+(args...), args....)`.
            update_bin!(bins, fn, arg, var_to_idx)
        end
    end
    if !iszero(constant)
        update_bin!(bins, fn(constant), var_to_idx)
    end
    return bins
end

# _exafy

# This method is used for objective constants.
_exafy(f::ExaModels.Null, data::Tuple, ::Any, ::Any) = f, data

# This method is used when a constant appears in a function.
function _exafy(f::Real, data::Tuple, ::Any, ::Any)
    e = ExaModels.DataIndexed(ExaModels.DataSource(), length(data) + 1)
    return e, (data..., f)
end

function _exafy(
    f::MOI.VariableIndex,
    data::Tuple,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    idx = var_to_idx[f]
    if idx < 0  # It's a parameter
        e = ExaModels.DataIndexed(ExaModels.DataSource(), length(data) + 1)
        return ExaModels.ParameterNode(e), (data..., -idx)
    end
    if var_to_data !== nothing
        # An optimization: the variable `f` may already appear in the tuple
        # `data`. If so, we want to re-use the slot instead of appending a new
        # element to `data`. (ExaModels could be clever here and check for
        # duplicates.)
        #
        # We don't have this optimization for ParameterNode's because the main
        # problem with duplicates are they they show up as duplicated elements
        # in the Jacobian and Hessian.
        if (pidx = get(var_to_data, idx, nothing)) !== nothing
            p_cache = ExaModels.DataIndexed(ExaModels.DataSource(), pidx)
            return ExaModels.Var(p_cache), data
        end
        var_to_data[idx] = length(data) + 1
    end
    e = ExaModels.DataIndexed(ExaModels.DataSource(), length(data) + 1)
    return ExaModels.Var(e), (data..., idx)
end

function _exafy(
    f::MOI.ScalarAffineTerm,
    data::Tuple,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    x_head, data = _exafy(f.variable, data, var_to_idx, var_to_data)
    c_head, data = _exafy(f.coefficient, data, var_to_idx, var_to_data)
    return c_head * x_head, data
end

# This method is used when a ScalarAffineFunction appears inside a
# ScalarNonlinearFunction. For that reason we don't do anything clever with the
# additive terms.
function _exafy(
    f::MOI.ScalarAffineFunction,
    data::Tuple,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    head, data = _exafy(f.constant, data, var_to_idx, var_to_data)
    if !isempty(f.terms)
        y = sum(begin
            c1, data = _exafy(term, data, var_to_idx, var_to_data)
            c1
        end for term in f.terms)
        head += y
    end
    return head, data
end

function _exafy(
    f::MOI.ScalarQuadraticTerm,
    data::Tuple,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    if f.variable_1 == f.variable_2
        x_head, data = _exafy(f.variable_1, data, var_to_idx, var_to_data)
        c_head, data = _exafy(f.coefficient / 2, data, var_to_idx, var_to_data)
        return c_head * abs2(x_head), data
    end
    x1_head, data = _exafy(f.variable_1, data, var_to_idx, var_to_data)
    x2_head, data = _exafy(f.variable_2, data, var_to_idx, var_to_data)
    c_head, data = _exafy(f.coefficient, data, var_to_idx, var_to_data)
    return c_head * x1_head * x2_head, data
end

# This method is used when a ScalarQuadraticFunction appears inside a
# ScalarNonlinearFunction. For that reason we don't do anything clever with the
# additive terms.
function _exafy(
    f::MOI.ScalarQuadraticFunction,
    data::Tuple,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    var_to_data::Union{Nothing,Dict{Int,Int}},
)
    head, data = _exafy(f.constant, data, var_to_idx, var_to_data)
    if !isempty(f.affine_terms)
        head += sum(begin
            c1, data = _exafy(term, data, var_to_idx, var_to_data)
            c1
        end for term in f.affine_terms)
    end
    if !isempty(f.quadratic_terms)
        head += sum(begin
            c1, data = _exafy(term, data, var_to_idx, var_to_data)
            c1
        end for term in f.quadratic_terms)
    end
    return head, data
end

function _exafy(
    f::MOI.ScalarNonlinearFunction,
    data::Tuple,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    ::Nothing,
)
    # Replace the incoming `var_to_data === nothing` with a dictionary that maps
    # the variable index with the element in `data`. This is used when there are
    # repeated variable indices in `f`. See `_exafy(::VariableIndex, args...)`.
    return _exafy(f, data, var_to_idx, Dict{Int,Int}())
end

function _exafy(
    f::MOI.ScalarNonlinearFunction,
    data::Tuple,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    var_to_data::Dict{Int,Int},
)
    # This assumes that we support only the default functions in `MOI.Nonlinear`
    op = getfield(MOI.Nonlinear, f.head)
    if length(f.args) == 1
        # A special case when there is one argument.
        arg, data = _exafy(only(f.args), data, var_to_idx, var_to_data)
        return op(arg), data
    elseif length(f.args) == 2
        # A special case when there are two arguments
        arg1, data = _exafy(f.args[1], data, var_to_idx, var_to_data)
        arg2, data = _exafy(f.args[2], data, var_to_idx, var_to_data)
        return op(arg1, arg2), data
    end
    args = ()
    for arg in f.args
        head, data = _exafy(arg, data, var_to_idx, var_to_data)
        args = (args..., head)
    end
    return op(args...), data
end

"""
    ExaModels.ExaModel(
        src::MOI.ModelLike;
        backend = nothing,
        prod::Bool = false,
        T = ExaModels.default_T(backend),
    )

Convert `src` to an `ExaModel`.
"""
function ExaModels.ExaModel(
    src::MOI.ModelLike;
    backend = nothing,
    prod::Bool = false,
    T = ExaModels.default_T(backend),
)
    c, _, _ = to_exacore(src, T, backend)
    return ExaModels.ExaModel(c; prod)
end

function to_exacore(src::MOI.ModelLike, ::Type{T}, backend) where {T}
    minimize = MOI.get(src, MOI.ObjectiveSense()) != MOI.MAX_SENSE
    c = ExaModels.ExaCore(T; backend, minimize, concrete = Val(true))
    var_to_idx = Dict{MOI.VariableIndex,Int}()
    con_to_idx = Dict{MOI.ConstraintIndex,Int}()
    c = copy_variables!(c, src, var_to_idx, con_to_idx)
    # copy constraints
    offset = 0
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !_supports_constraint(F, S)
            throw(MOI.UnsupportedConstraint{F,S}())
        elseif F <: MOI.VariableIndex
            continue
        end
        c, offset = exafy_con(c, src, F, S, offset, var_to_idx, con_to_idx)
    end
    # copy the objective
    F = MOI.get(src, MOI.ObjectiveFunctionType())
    for bin in exafy_obj(MOI.get(src, MOI.ObjectiveFunction{F}()), var_to_idx)
        c, _ = ExaModels.add_obj(c, bin.head, bin.data)
    end
    return c, var_to_idx, con_to_idx
end

function copy_variables!(
    c::ExaModels.ExaCore{T},
    src::MOI.ModelLike,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    con_to_idx,
) where {T}
    # Deal with the parameters
    attr = MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Parameter{T}}()
    params = MOI.get(src, attr)
    if !isempty(params)
        p0 = [MOI.get(src, MOI.ConstraintSet(), ci).value for ci in params]
        c, _ = ExaModels.add_par(c, p0)
        for (i, ci) in enumerate(params)
            # Parameters are indexed {-1, ..., -P}
            var_to_idx[MOI.VariableIndex(ci.value)] = con_to_idx[ci] = -i
        end
    end
    # Now deal with the decision variables
    nvar = MOI.get(src, MOI.NumberOfVariables()) - length(params)
    lvar, uvar = fill(typemin(T), nvar), fill(typemax(T), nvar)
    start = zeros(T, nvar)
    i = 0
    supports = MOI.supports(src, MOI.VariablePrimalStart(), MOI.VariableIndex)
    for x in MOI.get(src, MOI.ListOfVariableIndices())
        if haskey(var_to_idx, x)
            continue # x is a Parameter
        end
        i += 1
        var_to_idx[x] = i
        if supports
            x0 = MOI.get(src, MOI.VariablePrimalStart(), x)::Union{Nothing,T}
            start[i] = something(x0, zero(T))
        end
    end
    _update_bounds(src, var_to_idx, con_to_idx, lvar, uvar, MOI.GreaterThan{T})
    _update_bounds(src, var_to_idx, con_to_idx, lvar, uvar, MOI.LessThan{T})
    _update_bounds(src, var_to_idx, con_to_idx, lvar, uvar, MOI.EqualTo{T})
    _update_bounds(src, var_to_idx, con_to_idx, lvar, uvar, MOI.Interval{T})
    c, _ = ExaModels.add_var(c, nvar; start, lvar, uvar)
    return c
end

function _update_bounds(
    src,
    var_to_idx,
    con_to_idx,
    lvar,
    uvar,
    ::Type{S},
) where {S}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
        col = con_to_idx[ci] = var_to_idx[MOI.VariableIndex(ci.value)]
        _update_bound(lvar, uvar, col => MOI.get(src, MOI.ConstraintSet(), ci))
    end
    return
end

function _update_bound(lvar, uvar, (col, set)::Pair{Int,<:MOI.GreaterThan})
    lvar[col] = set.lower
    return
end

function _update_bound(lvar, uvar, (col, set)::Pair{Int,<:MOI.LessThan})
    uvar[col] = set.upper
    return
end

function _update_bound(lvar, uvar, (col, set)::Pair{Int,<:MOI.EqualTo})
    lvar[col] = uvar[col] = set.value
    return
end

function _update_bound(lvar, uvar, (col, set)::Pair{Int,<:MOI.Interval})
    lvar[col], uvar[col] = set.lower, set.upper
    return
end

function exafy_con(
    c::ExaModels.ExaCore{T},
    src::MOI.ModelLike,
    ::Type{F},
    ::Type{S},
    offset::Int,
    var_to_idx::Dict{MOI.VariableIndex,Int},
    con_to_idx::Dict,
)::Tuple{ExaModels.ExaCore{T},Int} where {T,F,S}
    cis = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
    l = length(cis)
    bins = Bin[]
    lcon, ucon, start = fill(typemin(T), l), fill(typemax(T), l), zeros(T, l)
    has_start =
        MOI.supports(src, MOI.ConstraintPrimalStart(), MOI.ConstraintIndex{F,S})
    for (i, ci) in enumerate(cis)
        con_to_idx[ci] = offset + i
        # Update function
        func = MOI.get(src, MOI.ConstraintFunction(), ci)
        update_bin!(bins, ConstraintBin(i), func, var_to_idx)
        # Update set
        lcon[i], ucon[i] = _bounds(MOI.get(src, MOI.ConstraintSet(), ci))
        # ConstraintPrimalStart
        if has_start
            c0 = MOI.get(src, MOI.ConstraintPrimalStart(), ci)
            start[i] = something(c0, zero(T))
        end
    end
    c, cons = ExaModels.add_con(c, l; start, lcon, ucon)
    for bin in bins
        c, _ = ExaModels.add_con!(c, cons, (bin.head for _ in bin.data))
    end
    return c, offset + l
end

_bounds(s::MOI.Interval) = (s.lower, s.upper)

_bounds(s::MOI.EqualTo) = (s.value, s.value)

_bounds(s::MOI.GreaterThan{T}) where {T} = (s.lower, typemax(T))

_bounds(s::MOI.LessThan{T}) where {T} = (typemin(T), s.upper)

# exafy_obj

function exafy_obj(::MOI.AbstractFunction, var_to_idx)
    return throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{typeof(f)}()))
end

exafy_obj(::Nothing, var_to_idx) = Bin[]

function exafy_obj(
    f::Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
        MOI.ScalarNonlinearFunction,
    },
    var_to_idx,
)
    return update_bin!(Bin[], ObjectiveBin(), f, var_to_idx)
end

# Now comes the MOI interface

"""
    ExaModels.Optimizer(solver, backend = nothing)

Create a new ExaModels.Optimizer object.

## Examples

```julia-repl
julia> import ExaModels, NLPModelsIpopt, KernelAbstractions

julia> optimizer = () -> ExaModels.Optimizer(NLPModelsIpopt.ipopt);

julia> optimizer = () -> ExaModels.Optimizer(NLPModelsIpopt.ipopt, KernelAbstractions.CPU());
```
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    solver::Any
    backend::Any
    model::Union{Nothing,ExaModels.ExaModel}
    result::Any
    solve_time::Float64
    options::Dict{Symbol,Any}

    function Optimizer(solver, backend = nothing; kwargs...)
        options = Dict{Symbol,Any}(kwargs...)
        return new(solver, backend, nothing, nothing, 0.0, options)
    end
end

function MOI.empty!(model::ExaModelsMOI.Optimizer)
    model.model = nothing
    model.result = nothing
    model.solve_time = 0.0
    return
end

MOI.is_empty(model::Optimizer) = isnothing(model.model)

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return _supports_constraint(F, S)
end

 _supports_constraint(::Type{F}, ::Type{S}) where {F,S} = false

function _supports_constraint(
    ::Type{F},
    ::Type{S},
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
        MOI.ScalarNonlinearFunction,
    },
    S<:Union{MOI.GreaterThan,MOI.LessThan,MOI.EqualTo,MOI.Interval},
}
    return true
end

_supports_constraint(::Type{MOI.VariableIndex}, ::Type{<:MOI.Parameter}) = true

function MOI.supports_add_constrained_variable(
    ::Optimizer,
    ::Type{<:MOI.Parameter},
)
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{F},
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
        MOI.ScalarNonlinearFunction,
    },
}
    return true
end

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

# MOI.copy_to

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    # TODO(odow): support other element types
    c, var_to_idx, con_to_idx = to_exacore(src, Float64, dest.backend)
    dest.model = ExaModels.ExaModel(c; prod = true)
    map = MOI.Utilities.IndexMap()
    for x in MOI.get(src, MOI.ListOfVariableIndices())
        idx = var_to_idx[x]
        if idx < 0
            map[x] = MOI.VariableIndex(-idx + PARAMETER_INDEX_THRESHOLD)
        else
            map[x] = MOI.VariableIndex(idx)
        end
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        _make_constraints_map(src, map.con_map[F, S], con_to_idx)
    end
    return map
end

function _make_constraints_map(
    model,
    map::MOI.Utilities.DoubleDicts.IndexDoubleDictInner{F,S},
    con_to_idx,
) where {F,S}
    for c in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
        map[c] = MOI.ConstraintIndex{F,S}(con_to_idx[c])
    end
    return
end

# MOI.optimize!

function MOI.optimize!(model::Optimizer)
    start_time = time()
    result = model.solver(model.model; model.options...)
    model.result = (
        objective = result.objective,
        solution = Array(result.solution),
        multipliers = Array(result.multipliers),
        multipliers_L = Array(result.multipliers_L),
        multipliers_U = Array(result.multipliers_U),
        status = result.status,
    )
    model.solve_time = time() - start_time
    return
end

# MOI.TerminationStatus

# SolverCore returns a `Symbol` in `result.status` for any solver implementing
# the NLPModels callable interface (e.g. `madnlp(::AbstractNLPModel)`,
# `ipopt(::AbstractNLPModel)`). The vocabulary is defined by SolverCore.jl
# (see SolverCore/src/stats.jl). Solvers that emit something else can be
# supported by extending these dicts or overriding the two `MOI.get` methods.
const _TERMINATION_STATUS_CODES = Dict{Symbol, MOI.TerminationStatusCode}(
    :first_order => MOI.LOCALLY_SOLVED,
    :acceptable => MOI.ALMOST_LOCALLY_SOLVED,
    :small_step => MOI.SLOW_PROGRESS,
    :infeasible => MOI.INFEASIBLE,
    :max_iter => MOI.ITERATION_LIMIT,
    :max_time => MOI.TIME_LIMIT,
    :user => MOI.INTERRUPTED,
    :exception => MOI.OTHER_ERROR,
)

MOI.get(model::Optimizer, ::MOI.RawStatusString) = string(model.result.status)

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.result === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return get(_TERMINATION_STATUS_CODES, model.result.status, MOI.OTHER_ERROR)
end

# MOI.PrimalStatus, MOI.DualStatus

const _RESULT_STATUS_CODES = Dict{Symbol, MOI.ResultStatusCode}(
    :first_order => MOI.FEASIBLE_POINT,
    :acceptable => MOI.NEARLY_FEASIBLE_POINT,
    :infeasible => MOI.INFEASIBLE_POINT,
)

function MOI.get(model::Optimizer, attr::Union{MOI.PrimalStatus,MOI.DualStatus})
    if model.result === nothing || attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    return get(
        _RESULT_STATUS_CODES,
        model.result.status,
        MOI.UNKNOWN_RESULT_STATUS,
    )
end

# MOI.VariablePrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    if vi.value > PARAMETER_INDEX_THRESHOLD
        return model.model.θ[vi.value-PARAMETER_INDEX_THRESHOLD]
    end
    return model.result.solution[vi.value]
end

# MOI.ConstraintDual

_scale(model::Optimizer) = _scale(model.model)

_scale(model::ExaModels.ExaModel) = model.meta.minimize ? 1.0 : -1.0

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    return -_scale(model) * model.result.multipliers[ci.value]
end

function _reduced_cost(model, col)
    return model.result.multipliers_L[col] - model.result.multipliers_U[col]
end
function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.LessThan},
)
    MOI.check_result_index_bounds(model, attr)
    rc = _reduced_cost(model, ci.value)
    return min(zero(rc), _scale(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:MOI.GreaterThan},
)
    MOI.check_result_index_bounds(model, attr)
    rc = _reduced_cost(model, ci.value)
    return max(zero(rc), _scale(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
) where {S<:Union{MOI.Interval,MOI.EqualTo}}
    MOI.check_result_index_bounds(model, attr)
    return _scale(model) * _reduced_cost(model, ci.value)
end

# MOI.ResultCount

MOI.get(model::Optimizer, ::MOI.ResultCount) = model.result !== nothing ? 1 : 0

# MOI.ObjectiveValue

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.result.objective
end

# MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

# MOI.SolverName

function MOI.get(model::Optimizer, ::MOI.SolverName)
    return "$(string(model.solver)) running with ExaModels"
end

# MOI.RawOptimizerAttribute

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    model.options[Symbol(attr.name)] = value
    # No need to reset model.solver because this gets handled in optimize!.
    return
end

# MOI.NLPBlock

function MOI.set(::Optimizer, ::MOI.NLPBlock, ::MOI.NLPBlockData)
    return error(
        """
        The legacy nonlinear model interface is not supported.

        Please use the new MOI-based interface.
        """,
    )
end

end # module
