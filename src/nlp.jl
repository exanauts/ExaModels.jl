abstract type AbstractVariable end
abstract type AbstractConstraint end
abstract type AbstractObjective end

struct VariableNull <: AbstractVariable end
struct ObjectiveNull <: AbstractObjective end
struct ConstraintNull <: AbstractConstraint end

struct Variable{S,O} <: AbstractVariable
    # itr::I
    size::S
    offset::O
end
struct Objective{R,F,I} <: AbstractObjective
    inner::R
    f::F
    itr::I
end
struct Constraint{R,F,I} <: AbstractConstraint
    inner::R
    f::F
    itr::I
end
struct ConstraintAug{R,F,I} <: AbstractConstraint
    inner::R
    f::F
    itr::I
    oa::Int
end

@kwdef mutable struct Counters
    neval_obj::Int = 0
    neval_cons::Int = 0
    neval_grad::Int = 0
    neval_jac::Int = 0
    neval_hess::Int = 0
    teval_obj::Float64 = 0.
    teval_cons::Float64 = 0.
    teval_grad::Float64 = 0.
    teval_jac::Float64 = 0.
    teval_hess::Float64 = 0.
end


"""
SIMDiff.Core

A core data object used for creating `SIMDiff.Model`.

SIMDiff.Core()

Returns `SIMDiff.Core` for creating `SIMDiff.Model{Float64,Vector{Float64}}`
"""
Base.@kwdef mutable struct Core{T, VT <: AbstractVector{T}, B}
    obj::AbstractObjective = ObjectiveNull()
    con::AbstractConstraint = ConstraintNull()
    nvar::Int = 0
    ncon::Int = 0
    nconaug::Int = 0
    nobj::Int = 0
    nnzc::Int = 0
    nnzg::Int = 0
    nnzj::Int = 0
    nnzh::Int = 0
    x0::VT = zeros(0)
    lvar::VT = similar(x0)
    uvar::VT = similar(x0)
    y0::VT = similar(x0)
    lcon::VT = similar(x0)
    ucon::VT = similar(x0)
    backend::B = nothing
end


"""
SIMDiff.Core(S::Type)

Returns `SIMDiff.Core` for creating `SIMDiff.Model{T,VT}`, where VT <: S
"""
Core(::Nothing) = Core()

"""
SIMDiff.Model <: NLPModels.AbstractNLPModel

An NLP model with SIMDiff backend
"""
struct Model{T,VT,E,O,C} <: NLPModels.AbstractNLPModel{T,VT}
    objs::O
    cons::C
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::Counters
    ext::E
end

"""
SIMDiff.Model(core)

"""
function Model(c::C) where C <: Core
    return Model(
        c.obj, c.con, 
        NLPModels.NLPModelMeta(
            c.nvar,
            ncon = c.ncon,
            nnzj = c.nnzj,
            nnzh = c.nnzh,
            x0 = c.x0,
            lvar = c.lvar,
            uvar = c.uvar,
            y0 = c.y0,
            lcon = c.lcon,
            ucon = c.ucon
        ),
        Counters(),
        extension(c) 
    )
end

@inline Base.getindex(v::V,i) where V <: Variable = Var(i + (v.offset - _start(v.size[1]) + 1))
@inline Base.getindex(v::V,i,j) where V <: Variable = Var(
    i
    + j * _length(v.size[1])
    + (
        v.offset  - _start(v.size[1]) + 1  - _start(v.size[2])  * _length(v.size[1])
    )
)

function myappend!(a,b::Base.Generator)
    la = length(a);
    lb = length(b);
    resize!(a, la+lb);
    map!(b.f, view(a,(la+1):(la+lb)) , b.iter)
    return a
end

function myappend!(a,b::AbstractArray)
    la = length(a);
    lb = length(b);
    resize!(a, la+lb);
    map!(identity, view(a,(la+1):(la+lb)) , b)
    return a
end


total(ns) = prod(_length(n) for n in ns)
_length(n::Int) = n
_length(n::UnitRange) = length(n)
size(ns) = Tuple(_length(n) for n in ns)
_start(n::Int) = 1
_start(n::UnitRange) = n.start

# redo(x::AbstractArray,ns) = x
# redo(x::Real,ns) = fill(x,size(ns))

function data(
    c::Core{T,VT,B}, gen
    ) where {T, VT, B}
    return collect(gen)
end

function variable(
    c::C, ns...;
    start = (zero(T) for i in 1:total(ns)),
    lvar = (T(-Inf) for i in 1:total(ns)),
    uvar = (T( Inf) for i in 1:total(ns))
    ) where {T, C <: Core{T}}

    # start = redo(start, ns)
    # lvar = redo(lvar, ns)
    # uvar = redo(uvar, ns)

    o = c.nvar
    c.nvar += total(ns)
    c.x0 = myappend!(c.x0, start)
    c.lvar = myappend!(c.lvar, lvar)
    c.uvar = myappend!(c.uvar, uvar)
    
    return Variable(ns,o)
    
end

function objective(c::C,gen) where C <: Core
    f = Func(
        gen, c.nobj, c.nnzg, c.nnzh
    )

    nitr = length(gen.iter)
    c.nobj += nitr
    c.nnzg += nitr * f.o1step
    c.nnzh += nitr * f.o2step

    c.obj = Objective(
        c.obj, f, gen.iter
    )
end

function constraint(
    c::C,
    gen::Base.Generator;
    start = (zero(T) for i in 1:length(gen)),
    lcon = (zero(T) for i in 1:length(gen)),
    ucon = (zero(T) for i in 1:length(gen))
    ) where {T, C <: Core{T}}

    f = Func(
        gen, c.ncon, c.nnzj, c.nnzh
    )
    nitr = length(gen.iter)
    c.ncon += nitr
    c.nnzj += nitr * f.o1step
    c.nnzh += nitr * f.o2step
    
    c.y0 = myappend!(c.y0, start)
    c.lcon = myappend!(c.lcon, lcon)
    c.ucon = myappend!(c.ucon, ucon)
    
    c.con = Constraint(
        c.con, f, gen.iter
    )
end

function constraint!(c::C,c1,gen) where C <: Core
    f = Func(
        gen, offset0(c1,0), c.nnzj, c.nnzh
    )
    oa = c.nconaug

    nitr = length(gen.iter)

    c.nconaug += nitr
    c.nnzj += nitr * f.o1step
    c.nnzh += nitr * f.o2step

    c.con = ConstraintAug(
        c.con, f, gen.iter, oa
    )
end


function extension(args...) end

function NLPModels.jac_structure!(
    m::M,
    rows::V,
    cols::V
    ) where {M <: Model, V <: AbstractVector}
    
    _jac_structure!(m.cons, rows, cols)
end

_jac_structure!(cons::ConstraintNull, rows, cols) = nothing
function _jac_structure!(cons, rows, cols)
    _jac_structure!(cons.inner, rows, cols)
    sjacobian!(rows, cols, cons, nothing, NaN16)
end

function NLPModels.hess_structure!(
    m::M,
    rows::V,
    cols::V
    ) where {M <: Model, V <: AbstractVector}

    _obj_hess_structure!(m.objs, rows, cols)
    _con_hess_structure!(m.cons, rows, cols)
end

_obj_hess_structure!(objs::ObjectiveNull, rows, cols) = nothing
function _obj_hess_structure!(objs, rows, cols)
    _obj_hess_structure!(objs.inner, rows, cols)
    shessian!(rows,cols, objs, nothing, NaN16, NaN16)
end

_con_hess_structure!(cons::ConstraintNull, rows, cols) = nothing
function _con_hess_structure!(cons, rows, cols)
    _con_hess_structure!(cons.inner, rows, cols)
    shessian!(rows,cols, cons, nothing, NaN16, NaN16)
end

function NLPModels.obj(
    m::M,
    x::V
    ) where {M <: Model, V <: AbstractVector}

    _obj(m.objs,x)
    
end

_obj(objs,x) = _obj(objs.inner,x) + sum(objs.f.f(k,x) for k in objs.itr)
_obj(objs::ObjectiveNull,x) = zero(eltype(x))

function NLPModels.cons!(
    m::M,
    x::V,
    g::V
    ) where {M <: Model, V <: AbstractVector}

    fill!(g, zero(eltype(g)))
    _cons!(m.cons,x,g)
end

function _cons!(cons,x,g)
    _cons!(cons.inner,x,g)
    @simd for i in eachindex(cons.itr)
        g[offset0(cons,i)] += cons.f.f(cons.itr[i],x)
    end
end
_cons!(cons::ConstraintNull,x,g) = nothing



function NLPModels.grad!(
    m::M,
    x::V,
    f::V
    ) where {M <: Model, V <: AbstractVector}

    fill!(f,zero(eltype(f)))
    _grad!(m.objs,x,f)
end

function _grad!(objs,x,f)
    _grad!(objs.inner,x,f)
    gradient!(f, objs, x, one(eltype(f)))
end
_grad!(objs::ObjectiveNull,x,f) = nothing

function NLPModels.jac_coord!(
    m::M,
    x::V,
    jac::V
    ) where {M <: Model, V <: AbstractVector}

    fill!(jac,zero(eltype(jac)))
    _jac_coord!(m.cons,x,jac)
end

_jac_coord!(cons::ConstraintNull,x,jac) = nothing
function _jac_coord!(cons,x,jac)
    _jac_coord!(cons.inner, x, jac)
    sjacobian!(jac, nothing, cons, x, one(eltype(jac)))
end

function NLPModels.hess_coord!(
    m::M,
    x::V,
    y::V,
    hess::V;
    obj_weight = one(eltype(x))
    ) where {M <: Model, V <: AbstractVector}


    fill!(hess,zero(eltype(hess)))
    
    _obj_hess_coord!(m.objs, x,y,hess,obj_weight)
    _con_hess_coord!(m.cons, x,y,hess,obj_weight)
end
_obj_hess_coord!(objs::ObjectiveNull, x,y,hess,obj_weight) = nothing
function _obj_hess_coord!(objs, x,y,hess,obj_weight)
    _obj_hess_coord!(objs.inner, x,y,hess,obj_weight)
    shessian!(hess, nothing, objs, x, obj_weight, zero(eltype(hess)))
end

_con_hess_coord!(cons::ConstraintNull, x,y,hess,obj_weight) = nothing
function _con_hess_coord!(cons, x,y,hess,obj_weight)
    _con_hess_coord!(cons.inner, x,y,hess,obj_weight)
    shessian!(hess, nothing, cons, x, y, zero(eltype(hess)))
end

@inbounds @inline offset0(a,i) = offset0(a.f, i)
@inbounds @inline offset1(a,i) = offset1(a.f, i)
@inbounds @inline offset2(a,i) = offset2(a.f, i)
@inbounds @inline offset0(f,itr,i) = offset0(f,i)
@inbounds @inline offset0(f::F,i) where F <: Func = f.o0 + i
@inbounds @inline offset1(f::F,i) where F <: Func = f.o1 + f.o1step * (i-1)
@inbounds @inline offset2(f::F,i) where F <: Func = f.o2 + f.o2step * (i-1)
@inbounds @inline offset0(a::C,i) where C <: ConstraintAug = offset0(a.f,a.itr,i)
@inbounds @inline offset0(f::F,itr,i) where {P <: Pair, F <: Func{P}} = f.o0 + f.f.first(itr[i],nothing)
