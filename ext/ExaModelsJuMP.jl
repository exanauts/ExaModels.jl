module ExaModelsJuMP

import ExaModels, JuMP
import JuMP.MOI, ExaModels.NLPModels


function list!(bin, e, p)
    if _list!(bin,e,p)
        return bin
    else
        return (e,[p],bin)
    end
end
function _list!(bin::Tuple{E,P,I}, e, p) where {E,P,I}
    
    if e == bin[1] && p isa eltype(bin[2])
        push!(bin[2], p)
        return true
    else
        return _list!(bin[3], e, p)
    end
end

function _list!(::Nothing, e, p)
    return false
end

function ExaModels.ExaModel(jm::JuMP.GenericModel{T}) where T

    
    jm_cache = jm.moi_backend.model_cache

    # create exacore
    c = ExaModels.ExaCore()

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
    bin = nothing
    bin = exafy_obj(jobjs.scalar_nonlinear, bin)
    
    build_objective(c, bin)
    
    # constraint
    jcons = jm_cache.model.constraints

    neq = getncon(jcons.moi_scalarnonlinearfunction.moi_equalto)
    nge = getncon(jcons.moi_scalarnonlinearfunction.moi_greaterthan)
    nle = getncon(jcons.moi_scalarnonlinearfunction.moi_lessthan)

    ncon = neq + nge + nle

    bin = nothing

    # cons = ExaModels.constraint(c, ncons; start = y0, lcon= lcon, ucon = ucon)

    y0 = similar(x0, ncon)
    lcon = similar(x0, ncon)
    ucon = similar(x0, ncon)

    for (i,(c,e)) in jcons.moi_scalarnonlinearfunction.moi_equalto.constraints
        y0[i.value] = zero(eltype(x0))
        lcon[i.value] = e.value
        ucon[i.value] = e.value

        e,p = _exafy(c)
        bin = list!(bin, e, p)
    end
    
    # bin = exafy_con(jcons.moi_scalarnonlinearfunction, bin)
    f = ExaModels._simdfunction(bin[1], c.ncon, c.nnzj, c.nnzh)
    ExaModels._constraint(c, f, bin[2], y0, lcon, ucon)
    
    
    ExaModels.ExaModel(c)
end

getncon(cons) = length(cons.constraints)
getncon(::Nothing) = 0

function build_objective(c, bin)
    build_objective(c, bin[3])
    f = ExaModels._simdfunction(bin[1], c.nobj, c.nnzg, c.nnzh)
    ExaModels._objective(c, f, bin[2])
end

function build_objective(c, ::Nothing) end

function exafy_obj(o::MOI.ScalarNonlinearFunction, bin)
    offset = 0.
    if o.head == :+;
        for m in o.args
            if m isa MOI.ScalarAffineFunction
                for mm in m.affine_terms
                    e,p = _exafy(mm)
                    bin = list!(bin, e, p)
                end
            elseif m isa MOI.ScalarQuadraticFunction
                for mm in m.affine_terms
                    e,p = _exafy(mm)
                    bin = list!(bin, e, p)
                end
                for mm in m.quadratic_terms
                    e,p = _exafy(mm)
                    bin = list!(bin, e, p)
                end
                offset += m.constant
            else
                e,p = _exafy(m)
                bin = list!(bin, e, p)
            end
        end

    else
        e, p = _exafy(o)
        bin = list!(bin, e, p)
    end

    bin = list!(bin, ExaModels.Null(offset), (1,))

    return bin
end

function _exafy(v::MOI.VariableIndex, p = ())
    i = ExaModels.ParIndexed( ExaModels.ParSource(), length(p) + 1)
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
    # if length(e.args) == 1
    #     c1, p = _exafy(e.args[1], p)
    #     return op(e.head)(c1), p
    # elseif length(e.args) == 2
    #     c1, p1 = _exafy(e.args[1], p)
    #     c2, p2 = _exafy(e.args[2], p1)
    #     p = (p1...,p2...)
    #     return op(e.head)(c1, c2), p2
    # else 
    #     error("Performing $(op(e.head)) to $(length(e.args)) arguments is not suppported")
    # end
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
    t = sum(
        begin
            c1, p = _exafy(term, p) 
            c1
        end
        for term in e.quadratic_terms)
    if !isempty(e.affine_terms)
        t += sum(
            begin
                c1, p = _exafy(term, p) 
                c1
            end
            for term in e.affine_terms)
    end
    return t + ExaModels.ParIndexed( ExaModels.ParSource(), length(p) + 1), (p..., e.constant)    
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

# eval is performance killer here. We want to explicitly include symbols for frequently used operations.
function op(s::Symbol)
    if s == :+;
        return +;
    elseif s == :*;
        return *;
    else
        return eval(s)
    end
end

end # module ExaModelsMOI



