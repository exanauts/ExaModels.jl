module ExaModelsMOI

import ExaModels: ExaModels, NLPModels

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

const SUPPORTED_OBJ_TYPE = [
    :scalar_nonlinear,
    :scalar_affine,
    :scalar_quadratic,
    :single_variable
]
const UNSUPPORTED_OBJ_TYPE = [
    :vector_nonlinear,
    :vector_affine,
    :vector_quadratic,
    :vector_variables,
]
    
const SUPPORTED_CONS_TYPE = [
    :moi_scalarnonlinearfunction,
    :moi_scalaraffinefunction,
    :moi_scalarquadraticfunction
]
const UNSUPPORTED_CONS_TYPE = [
    :moi_vectoraffinefunction,
    :moi_vectornonlinearfunction,
    :moi_vectorquadraticfunction,
    :moi_vectorofvariables
]

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
    if _update_bin!(bin,e,p) # if update succeeded, return the original bin
        return bin
    else # if update has failed, return a new bin
        return Bin(e,[p],bin) 
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

float_type(::MOIU.Model{T}) where T = T

function ExaModels.ExaModel(moi_backend::MathOptInterface.Utilities.CachingOptimizer; backend = nothing)

    jm_cache = moi_backend.model_cache
    
    T = float_type(moi_backend.model_cache.model)    
    
    # create exacore;
    c = ExaModels.ExaCore(T, backend)

    # variables
    jvars = jm_cache.model.variables
    lvar = jvars.lower
    uvar = jvars.upper
    x0   = fill!(similar(lvar), 0.)
    nvar = length(lvar)
    if haskey(jm_cache.varattr, MOI.VariablePrimalStart())
        for (k,v) in jm_cache.varattr[MOI.VariablePrimalStart()]
            x0[k.value] = v
        end
    end
    v = ExaModels.variable(c, nvar; start = x0, lvar= lvar, uvar = uvar)

    # objective
    jobjs = jm_cache.model.objective
    
    bin = BinNull()

    for field in SUPPORTED_OBJ_TYPE
        bin = exafy_obj(getfield(jobjs, field), bin)
    end

    for field in UNSUPPORTED_OBJ_TYPE
        if getfield(jobjs, field) != nothing
            error("$field type objective is not supported")
        end
    end
    
    build_objective(c, bin)
    
    # constraint
    jcons = jm_cache.model.constraints

    bin = BinNull()
    offset = 0
    lcon = similar(x0, 0)
    ucon = similar(x0, 0)

    for field in SUPPORTED_CONS_TYPE
        bin, offset = exafy_con(getfield(jcons, field), bin, offset, lcon, ucon)
    end

    for field in UNSUPPORTED_CONS_TYPE
        if getfield(jcons, field) != nothing
            error("$field type constraint is not supported")
        end
    end

    y0 = fill!(similar(lcon), zero(T))
    cons = ExaModels.constraint(c, offset; start = y0, lcon = lcon, ucon = ucon)
    build_constraint!(c, cons, bin)
    
    return ExaModels.ExaModel(c)
end

function exafy_con(cons::Nothing, bin, offset, lcon, ucon)
    return bin, offset
end
function exafy_con(cons, bin, offset, lcon, ucon)
    bin, offset = _exafy_con(cons.moi_equalto, bin, offset, lcon, ucon)
    bin, offset = _exafy_con(cons.moi_greaterthan, bin, offset, lcon, ucon)
    bin, offset = _exafy_con(cons.moi_lessthan, bin, offset, lcon, ucon)
    bin, offset = _exafy_con(cons.moi_interval, bin, offset, lcon, ucon)
    return bin, offset
end

function _exafy_con(cons::MOIU.VectorOfConstraints, bin, offset, lcon, ucon)
    l = length(cons.constraints)
    
    resize!(lcon, offset + l)
    resize!(ucon, offset + l)
    
    for (i,(c,e)) in cons.constraints
        
        _exafy_con_update_vector(i, e, lcon, ucon, offset)

        if c isa MOI.ScalarAffineFunction
            for mm in c.terms
                e,p = _exafy(mm)
                bin = update_bin!(
                    bin,
                    ExaModels.ParIndexed(ExaModels.ParSource(), length(p)+1) => e,
                    (p..., offset + i.value)
                ) # augment data with constraint index
            end
            bin = update_bin!(bin, ExaModels.Null(c.constant), (1,))
        elseif c isa MOI.ScalarQuadraticFunction
            for mm in c.affine_terms
                e,p = _exafy(mm)
                bin = update_bin!(
                    bin,
                    ExaModels.ParIndexed(ExaModels.ParSource(), length(p)+1)=>e,
                    (p..., offset + i.value)
                ) # augment data with constraint index
            end
            for mm in c.quadratic_terms
                e,p = _exafy(mm)
                bin = update_bin!(
                    bin,
                    ExaModels.ParIndexed(ExaModels.ParSource(), length(p)+1)=>e,
                    (p..., offset + i.value)
                ) # augment data with constraint index
            end
            bin = update_bin!(bin, ExaModels.Null(c.constant), (1,))
        elseif c isa MOI.ScalarNonlinearFunction && c.head == :+; # TODO maybe: also * Real, -, etc.
            for mm in c.args
                e,p = _exafy(mm)
                bin = update_bin!(
                    bin,
                    ExaModels.ParIndexed(ExaModels.ParSource(), length(p)+1)=>e,
                    (p..., offset + i.value)
                ) # augment data with constraint index
            end
        else
            e, p = _exafy(c)
            bin = update_bin!(
                bin,
                ExaModels.ParIndexed(ExaModels.ParSource(), length(p)+1)=>e,
                (p..., offset + i.value)
            ) # augment data with constraint index
        end
    end

    return bin, (offset += l)
end


function _exafy_con(::Nothing, bin, offset, lcon, ucon)
    return bin, offset
end

function _exafy_con_update_vector(i, e::MOI.Interval{T}, lcon, ucon, offset) where T
    lcon[offset + i.value] = e.lower
    ucon[offset + i.value] = e.upper
end

function _exafy_con_update_vector(i, e::MOI.LessThan{T}, lcon, ucon, offset) where T
    lcon[offset + i.value] = -Inf
    ucon[offset + i.value] = e.upper
end

function _exafy_con_update_vector(i, e::MOI.GreaterThan{T}, lcon, ucon, offset) where T
    ucon[offset + i.value] = Inf
    lcon[offset + i.value] = e.lower
end

function _exafy_con_update_vector(i, e::MOI.EqualTo{T}, lcon, ucon, offset) where T
    lcon[offset + i.value] = e.value
    ucon[offset + i.value] = e.value
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
    e,p = _exafy(o)    
    return update_bin!(bin, e, p)
end

function exafy_obj(o::MOI.ScalarQuadraticFunction{T}, bin) where T
    for m in o.affine_terms
        e,p = _exafy(m)
        bin = update_bin!(bin, e, p)
    end
    for m in o.quadratic_terms
        e,p = _exafy(m)
        bin = update_bin!(bin, e, p)
    end
    
    return update_bin!(bin, ExaModels.Null(o.constant), (1,))
end
    
function exafy_obj(o::MOI.ScalarAffineFunction{T}, bin) where T
    for m in o.terms
        e,p = _exafy(m)
        bin = update_bin!(bin, e, p)
    end
    
    return update_bin!(bin, ExaModels.Null(o.constant), (1,))
end

function exafy_obj(o::MOI.ScalarNonlinearFunction, bin) 
    constant = 0.
    if o.head == :+;
        for m in o.args
            if m isa MOI.ScalarAffineFunction
                for mm in m.affine_terms
                    e,p = _exafy(mm)
                    bin = update_bin!(bin, e, p)
                end
            elseif m isa MOI.ScalarQuadraticFunction
                for mm in m.affine_terms
                    e,p = _exafy(mm)
                    bin = update_bin!(bin, e, p)
                end
                for mm in m.quadratic_terms
                    e,p = _exafy(mm)
                    bin = update_bin!(bin, e, p)
                end
                constant += m.constant
            else
                e,p = _exafy(m)
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

function _exafy(i::R, p) where R <: Real
    return ExaModels.ParIndexed( ExaModels.ParSource(), length(p) + 1), (p..., i)
end

function _exafy(e::MOI.ScalarNonlinearFunction, p = ())
    return op(e.head)(
        (
            begin
                c, p = _exafy(e,p)
                c
            end
            for e in e.args
        )... 
    ), p
end

function _exafy(e::MOI.ScalarAffineFunction{T}, p = ()) where T
    return sum(
        begin
            c1, p = _exafy(term, p) 
            c1
        end
        for term in e.terms) + ExaModels.ParIndexed( ExaModels.ParSource(), length(p) + 1), (p..., e.constant)    
end

function _exafy(e::MOI.ScalarAffineTerm{T}, p = ()) where T
    c1, p = _exafy(e.variable, p)
    return *(c1, ExaModels.ParIndexed( ExaModels.ParSource(), length(p) + 1)), (p..., e.coefficient)
end

function _exafy(e::MOI.ScalarQuadraticFunction{T}, p = ()) where T
    t = ExaModels.ParIndexed( ExaModels.ParSource(), length(p) + 1)
    p = (p..., e.constant)

    if !isempty(e.affine_terms)
        t += sum(
            begin
                c1, p = _exafy(term, p) 
                c1
            end
            for term in e.affine_terms)
    end
    
    if !isempty(e.quadratic_terms)
        t += sum(
            begin
                c1, p = _exafy(term, p) 
                c1
            end
            for term in e.quadratic_terms)
    end
    
    return t, p
end

function _exafy(e::MOI.ScalarQuadraticTerm{T}, p = ()) where T
    
    if e.variable_1 == e.variable_2
        v, p = _exafy(e.variable_1, p)
        return ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) * abs2(v), (p..., e.coefficient / 2) # it seems that MOI assumes this by default
    else
        v1, p = _exafy(e.variable_1, p)
        v2, p = _exafy(e.variable_2, p)
        return ExaModels.ParIndexed(ExaModels.ParSource(), length(p) + 1) * v1 * v2, (p..., e.coefficient)
    end
end

# eval can be a performance killer -- we want to explicitly include symbols for frequently used operations.
function op(s::Symbol)
    if s == :+;
        return +;
    elseif s == :-;
        return -;
    elseif s == :*;
        return *;
    elseif s == :/;
        return /;
    elseif s == :^;
        return ^;
    elseif s == :sin;
        return sin;
    elseif s == :cos;
        return cos;
    elseif s == :exp;
        return exp;
    else
        return eval(s)
    end
end

end # module
