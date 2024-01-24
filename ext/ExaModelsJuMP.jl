module ExaModelsJuMP

import ExaModels: ExaModels, NLPModels
import JuMP: JuMP, MOI, MOIU


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
    bin = exafy_obj(jobjs.scalar_affine, bin)
    bin = exafy_obj(jobjs.scalar_quadratic, bin)
    
    build_objective(c, bin)
    
    # constraint
    jcons = jm_cache.model.constraints

    bin = nothing
    offset = 0
    
    y0 = similar(x0, 0)
    lcon = similar(x0, 0)
    ucon = similar(x0, 0)

    bin, offset = exafy_con(jcons.moi_scalarnonlinearfunction, bin, offset, y0, lcon, ucon)
    bin, offset = exafy_con(jcons.moi_scalaraffinefunction, bin, offset, y0, lcon, ucon)
    bin, offset = exafy_con(jcons.moi_scalarquadraticfunction, bin, offset, y0, lcon, ucon)

    build_constraint(c, bin, y0, lcon, ucon)
    

    m = ExaModels.ExaModel(c)


    m.meta.lcon .= lcon
    m.meta.ucon .= ucon
    
    return m
end

function exafy_con(cons, bin, offset, y0, lcon, ucon)
    bin, offset = _exafy_con(cons.moi_equalto, bin, offset, y0, lcon, ucon)
    bin, offset = _exafy_con(cons.moi_greaterthan, bin, offset, y0, lcon, ucon)
    bin, offset = _exafy_con(cons.moi_lessthan, bin, offset, y0, lcon, ucon)
    bin, offset = _exafy_con(cons.moi_interval, bin, offset, y0, lcon, ucon)
    return bin, offset
end

function _exafy_con(cons::MOIU.VectorOfConstraints, bin, offset, y0, lcon, ucon)
    l = length(cons.constraints)
    
    resize!(y0, offset + l)
    resize!(lcon, offset + l)
    resize!(ucon, offset + l)
    
    for (i,(c,e)) in cons.constraints
        
        _exafy_con_update_vector(i,e,y0, lcon, ucon, offset)
        
        bin = list!(bin, _exafy(c)...)
    end

    return bin, (offset += l)
end


function _exafy_con(::Nothing, bin, offset, y0, lcon, ucon)
    return bin, offset
end

function _exafy_con_update_vector(i, e::MOI.Interval{T}, y0, lcon, ucon, offset) where T
    y0[offset + i.value] = zero(T)
    lcon[offset + i.value] = e.lower
    ucon[offset + i.value] = e.upper
end

function _exafy_con_update_vector(i, e::MOI.LessThan{T}, y0, lcon, ucon, offset) where T
    y0[offset + i.value] = zero(T)
    lcon[offset + i.value] = -Inf
    ucon[offset + i.value] = e.upper
end

function _exafy_con_update_vector(i, e::MOI.GreaterThan{T}, y0, lcon, ucon, offset) where T
    y0[offset + i.value] = zero(T)
    ucon[offset + i.value] = Inf
    lcon[offset + i.value] = e.lower
end

function _exafy_con_update_vector(i, e::MOI.EqualTo{T}, y0, lcon, ucon, offset) where T
    y0[offset + i.value] = zero(T)
    lcon[offset + i.value] = e.value
    ucon[offset + i.value] = e.value
end



getncon(cons) = length(cons.constraints)
getncon(::Nothing) = 0

function build_constraint(c, bin, y0, lcon, ucon)
    build_constraint(c, bin[3], y0, lcon, ucon)
    f = ExaModels._simdfunction(bin[1], c.ncon, c.nnzj, c.nnzh)
    ExaModels._constraint(c, f, bin[2], y0, lcon, ucon)
end

function build_constraint(c, ::Nothing, y0, lcon, ucon) end

function build_objective(c, bin)
    build_objective(c, bin[3])
    f = ExaModels._simdfunction(bin[1], c.nobj, c.nnzg, c.nnzh)
    ExaModels._objective(c, f, bin[2])
end

function build_objective(c, ::Nothing) end

function exafy_obj(o::Nothing, bin) end

function exafy_obj(o::MOI.ScalarQuadraticFunction{T}, bin) where T
    for m in o.affine_terms
        e,p = _exafy(m)
        bin = list!(bin, e, p)
    end
    for m in o.quadratic_terms
        e,p = _exafy(m)
        bin = list!(bin, e, p)
    end
    
    return list!(bin, ExaModels.Null(o.constant), (1,))
end
    
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

    return list!(bin, ExaModels.Null(offset), (1,))
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



