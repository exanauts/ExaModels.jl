abstract type AbstractVariable end
abstract type AbstractConstraint end
abstract type AbstractObjective end

struct VariableNull <: AbstractVariable end
struct ObjectiveNull <: AbstractObjective end
struct ConstraintNull <: AbstractConstraint end

struct Variable{S,O} <: AbstractVariable
    size::S
    length::O
    offset::O
end
Base.show(io::IO, v::Variable) = print(
    io,
    """
Variable

  x ∈ R^{$(join(size(v.size)," × "))}
""",
)


struct Objective{R,F,I} <: AbstractObjective
    inner::R
    f::F
    itr::I
end
Base.show(io::IO, v::Objective) = print(
    io,
    """
Objective

  min (...) + ∑_{p ∈ P} f(x,p)

  where |P| = $(length(v.itr))
""",
)


struct Constraint{R,F,I,O} <: AbstractConstraint
    inner::R
    f::F
    itr::I
    offset::O
end
Base.show(io::IO, v::Constraint) = print(
    io,
    """
Constraint

  s.t. (...)
       g♭ ≤ [g(x,p)]_{p ∈ P} ≤ g♯

  where |P| = $(length(v.itr))
""",
)


struct ConstraintAug{R,F,I} <: AbstractConstraint
    inner::R
    f::F
    itr::I
    oa::Int
end

Base.show(io::IO, v::ConstraintAug) = print(
    io,
    """

Constraint Augmentation

  s.t. (...)
       g♭ ≤ (...) + ∑_{p ∈ P} h(x,p) ≤ g♯

  where |P| = $(length(v.itr))
""",
)

"""
ExaCore([array_eltype::Type; backend = backend, minimize = true])

Returns an intermediate data object `ExaCore`, which later can be used for creating `ExaModel`

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore()
An ExaCore

  Float type: ...................... Float64
  Array type: ...................... Vector{Float64}
  Backend: ......................... Nothing

  number of objective patterns: .... 0
  number of constraint patterns: ... 0

julia> c = ExaCore(Float32)
An ExaCore

  Float type: ...................... Float32
  Array type: ...................... Vector{Float32}
  Backend: ......................... Nothing

  number of objective patterns: .... 0
  number of constraint patterns: ... 0

julia> using CUDA

julia> c = ExaCore(Float32; backend = CUDABackend())
An ExaCore

  Float type: ...................... Float32
  Array type: ...................... CUDA.CuArray{Float32, 1, CUDA.DeviceMemory}
  Backend: ......................... CUDA.CUDAKernels.CUDABackend

  number of objective patterns: .... 0
  number of constraint patterns: ... 0
```
"""
Base.@kwdef mutable struct ExaCore{T,VT<:AbstractVector{T},B}
    backend::B = nothing
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
    x0::VT = convert_array(zeros(0), backend)
    lvar::VT = similar(x0)
    uvar::VT = similar(x0)
    y0::VT = similar(x0)
    lcon::VT = similar(x0)
    ucon::VT = similar(x0)
    minimize::Bool = true
end

# Deprecated as of v0.7
function ExaCore(::Type{T}, backend) where {T<:AbstractFloat}
    @warn "ExaCore(T, backend) is deprecated. Use ExaCore(T; backend = backend) instead"
    return ExaCore(T; backend = backend)
end
function ExaCore(backend)
    @warn "ExaCore(backend) is deprecated. Use ExaCore(T; backend = backend) instead"
    return ExaCore(; backend = backend)
end

ExaCore(::Type{T}; backend = nothing, kwargs...) where {T<:AbstractFloat} =
    ExaCore(x0 = convert_array(zeros(T, 0), backend); backend = backend, kwargs...)

depth(a) = depth(a.inner) + 1
depth(a::ObjectiveNull) = 0
depth(a::ConstraintNull) = 0

Base.show(io::IO, c::ExaCore{T,VT,B}) where {T,VT,B} = print(
    io,
    """
An ExaCore

  Float type: ...................... $T
  Array type: ...................... $VT
  Backend: ......................... $B

  number of objective patterns: .... $(depth(c.obj))
  number of constraint patterns: ... $(depth(c.con))
""",
)


struct ExaModel{T,VT,E,O,C} <: NLPModels.AbstractNLPModel{T,VT}
    objs::O
    cons::C
    meta::NLPModels.NLPModelMeta{T,VT}
    counters::NLPModels.Counters
    ext::E
end

function Base.show(io::IO, c::ExaModel{T,VT}) where {T,VT}
    println(io, "An ExaModel{$T, $VT, ...}\n")
    Base.show(io, c.meta)
end

"""
    ExaModel(core)

Returns an `ExaModel` object, which can be solved by nonlinear
optimization solvers within `JuliaSmoothOptimizer` ecosystem, such as
`NLPModelsIpopt` or `MadNLP`.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();                      # create an ExaCore object

julia> x = variable(c, 1:10);              # create variables

julia> objective(c, x[i]^2 for i in 1:10); # set objective function

julia> m = ExaModel(c)                     # creat an ExaModel object
An ExaModel{Float64, Vector{Float64}, ...}

  Problem name: Generic
   All variables: ████████████████████ 10     All constraints: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            free: ████████████████████ 10                free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
            nnzh: ( 81.82% sparsity)   10              linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
                                                    nonlinear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0
                                                         nnzj: (------% sparsity)

julia> using NLPModelsIpopt

julia> result = ipopt(m; print_level=0)    # solve the problem
"Execution stats: first-order stationary"

```
"""
function ExaModel(c::C; prod = nothing) where {C<:ExaCore}
    return ExaModel(
        c.obj,
        c.con,
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
            ucon = c.ucon,
            minimize = c.minimize,
        ),
        NLPModels.Counters(),
        nothing,
    )
end

@inline function Base.getindex(v::V, i) where {V<:Variable}
    _bound_check(v.size, i)
    Var(i + (v.offset - _start(v.size[1]) + 1))
end
@inline function Base.getindex(v::V, is...) where {V<:Variable}
    @assert(length(is) == length(v.size), "Variable index dimension error")
    _bound_check(v.size, is)
    Var(v.offset + idxx(is .- (_start.(v.size) .- 1), _length.(v.size)))
end

function _bound_check(sizes, i::I) where {I<:Integer}
    __bound_check(sizes[1], i)
end
function _bound_check(sizes, is::NTuple{N,I}) where {I<:Integer,N}
    __bound_check(sizes[1], is[1])
    _bound_check(sizes[2:end], is[2:end])
end
_bound_check(sizes, is) = nothing
_bound_check(sizes, is::Tuple{}) = nothing

function __bound_check(a::I, b::I) where {I<:Integer}
    @assert(1 <= b <= a, "Variable index bound error")
end
function __bound_check(a::UnitRange{Int}, b::I) where {I<:Integer}
    @assert(b in a, "Variable index bound error")
end


function append!(backend, a, b::Base.Generator, lb)
    b = _adapt_gen(b)

    la = length(a)
    resize!(a, la + lb)
    map!(b.f, view(a, (la+1):(la+lb)), convert_array(b.iter, backend))
    return a
end

function append!(backend, a, b::Base.Generator{UnitRange{I}}, lb) where {I}

    la = length(a)
    resize!(a, la + lb)
    map!(b.f, view(a, (la+1):(la+lb)), b.iter)
    return a
end

function append!(backend, a, b::AbstractArray, lb)

    la = length(a)
    resize!(a, la + lb)
    map!(identity, view(a, (la+1):(la+lb)), convert_array(b, backend))
    return a
end

function append!(backend, a, b::Number, lb)

    la = length(a)
    resize!(a, la + lb)
    fill!(view(a, (la+1):(la+lb)), b)
    return a
end

total(ns) = prod(_length(n) for n in ns)
_length(n::Int) = n
_length(n::UnitRange) = length(n)
size(ns) = Tuple(_length(n) for n in ns)
_start(n::Int) = 1
_start(n::UnitRange) = n.start

"""
    variable(core, dims...; start = 0, lvar = -Inf, uvar = Inf)

Adds variables with dimensions specified by `dims` to `core`, and returns `Variable` object. `dims` can be either `Integer` or `UnitRange`.

## Keyword Arguments
- `start`: The initial guess of the solution. Can either be `Number`, `AbstractArray`, or `Generator`.
- `lvar` : The variable lower bound. Can either be `Number`, `AbstractArray`, or `Generator`.
- `uvar` : The variable upper bound. Can either be `Number`, `AbstractArray`, or `Generator`.


## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();

julia> x = variable(c, 10; start = (sin(i) for i=1:10))
Variable

  x ∈ R^{10}

julia> y = variable(c, 2:10, 3:5; lvar = zeros(9,3), uvar = ones(9,3))
Variable

  x ∈ R^{9 × 3}

```
"""
function variable(
    c::C,
    ns...;
    start = zero(T),
    lvar = T(-Inf),
    uvar = T(Inf),
) where {T,C<:ExaCore{T}}


    o = c.nvar
    len = total(ns)
    c.nvar += len
    c.x0 = append!(c.backend, c.x0, start, total(ns))
    c.lvar = append!(c.backend, c.lvar, lvar, total(ns))
    c.uvar = append!(c.backend, c.uvar, uvar, total(ns))

    return Variable(ns, len, o)

end

function variable(c::C; kwargs...) where {T,C<:ExaCore{T}}

    return variable(c, 1; kwargs...)[1]
end

"""
    objective(core::ExaCore, generator)

Adds objective terms specified by a `generator` to `core`, and returns an `Objective` object. Note: it is assumed that the terms are summed.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();

julia> x = variable(c, 10);

julia> objective(c, x[i]^2 for i=1:10)
Objective

  min (...) + ∑_{p ∈ P} f(x,p)

  where |P| = 10
```
"""
function objective(c::C, gen) where {C<:ExaCore}
    gen = _adapt_gen(gen)
    f = SIMDFunction(gen, c.nobj, c.nnzg, c.nnzh)
    pars = gen.iter

    _objective(c, f, pars)
end

"""
    objective(core::ExaCore, expr [, pars])

Adds objective terms specified by a `expr` and `pars` to `core`, and returns an `Objective` object.
"""
function objective(c::C, expr::N, pars = 1:1) where {C<:ExaCore,N<:AbstractNode}
    f = _simdfunction(expr, c.nobj, c.nnzg, c.nnzh)

    _objective(c, f, pars)
end

function _objective(c, f, pars)
    nitr = length(pars)
    c.nobj += nitr
    c.nnzg += nitr * f.o1step
    c.nnzh += nitr * f.o2step

    c.obj = Objective(c.obj, f, convert_array(pars, c.backend))
end

"""
    constraint(core, generator; start = 0, lcon = 0,  ucon = 0)

Adds constraints specified by a `generator` to `core`, and returns an `Constraint` object. 

## Keyword Arguments
- `start`: The initial guess of the solution. Can either be `Number`, `AbstractArray`, or `Generator`.
- `lcon` : The constraint lower bound. Can either be `Number`, `AbstractArray`, or `Generator`.
- `ucon` : The constraint upper bound. Can either be `Number`, `AbstractArray`, or `Generator`.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();

julia> x = variable(c, 10);

julia> constraint(c, x[i] + x[i+1] for i=1:9; lcon = -1, ucon = (1+i for i=1:9))
Constraint

  s.t. (...)
       g♭ ≤ [g(x,p)]_{p ∈ P} ≤ g♯

  where |P| = 9
```
"""
function constraint(
    c::C,
    gen::Base.Generator;
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
) where {T,C<:ExaCore{T}}

    gen = _adapt_gen(gen)
    f = SIMDFunction(gen, c.ncon, c.nnzj, c.nnzh)
    pars = gen.iter

    _constraint(c, f, pars, start, lcon, ucon)
end

"""
    constraint(core, expr [, pars]; start = 0, lcon = 0,  ucon = 0)

Adds constraints specified by a `expr` and `pars` to `core`, and returns an `Constraint` object. 
"""
function constraint(
    c::C,
    expr::N,
    pars = 1:1;
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
) where {T,C<:ExaCore{T},N<:AbstractNode}

    f = _simdfunction(expr, c.ncon, c.nnzj, c.nnzh)

    _constraint(c, f, pars, start, lcon, ucon)
end

"""
    constraint(core, n; start = 0, lcon = 0,  ucon = 0)

Adds empty constraints of dimension n, so that later the terms can be added with `constraint!`. 
"""
function constraint(
    c::C,
    n;
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
) where {T,C<:ExaCore{T}}

    f = _simdfunction(Null(), c.ncon, c.nnzj, c.nnzh)

    _constraint(c, f, 1:n, start, lcon, ucon)
end


function _constraint(c, f, pars, start, lcon, ucon)
    nitr = length(pars)
    o = c.ncon
    c.ncon += nitr
    c.nnzj += nitr * f.o1step
    c.nnzh += nitr * f.o2step

    c.y0 = append!(c.backend, c.y0, start, nitr)
    c.lcon = append!(c.backend, c.lcon, lcon, nitr)
    c.ucon = append!(c.backend, c.ucon, ucon, nitr)

    c.con = Constraint(c.con, f, convert_array(pars, c.backend), o)
end

"""
    constraint!(c::C, c1, gen::Base.Generator) where {C<:ExaCore}

Expands the existing constraint `c1` in `c` by adding additional constraint terms specified by a `generator`.

# Arguments
- `c::C`: The model to which the constraints are added.
- `c1`: An initial constraint value or expression.
- `gen::Base.Generator`: A generator that produces the pair of constraint index and term to be added.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();

julia> x = variable(c, 10);

julia> c1 = constraint(c, x[i] + x[i+1] for i=1:9; lcon = -1, ucon = (1+i for i=1:9));

julia> constraint!(c, c1, i => sin(x[i+1]) for i=4:6)
Constraint Augmentation

  s.t. (...)
       g♭ ≤ (...) + ∑_{p ∈ P} h(x,p) ≤ g♯

  where |P| = 3
```
"""
function constraint!(c::C, c1, gen::Base.Generator) where {C<:ExaCore}

    gen = _adapt_gen(gen)
    f = SIMDFunction(gen, offset0(c1, 0), c.nnzj, c.nnzh)
    pars = gen.iter

    _constraint!(c, f, pars)
end

"""
    constraint!(c, c1, expr, pars)

Expands the existing constraint `c1` in `c` by adding addtional constraints terms specified by `expr` and `pars`.
"""
function constraint!(c::C, c1, expr, pars) where {C<:ExaCore}
    f = _simdfunction(expr, offset0(c1, 0), c.nnzj, c.nnzh)

    _constraint!(c, f, pars)
end

function _constraint!(c, f, pars)
    oa = c.nconaug

    nitr = length(pars)

    c.nconaug += nitr
    c.nnzj += nitr * f.o1step
    c.nnzh += nitr * f.o2step

    c.con = ConstraintAug(c.con, f, convert_array(pars, c.backend), oa)
end


function jac_structure!(m::ExaModel, rows::AbstractVector, cols::AbstractVector)

    _jac_structure!(m.cons, rows, cols)
end

_jac_structure!(cons::ConstraintNull, rows, cols) = nothing
function _jac_structure!(cons, rows, cols)
    _jac_structure!(cons.inner, rows, cols)
    sjacobian!(rows, cols, cons, nothing, NaN)
end

function hess_structure!(m::ExaModel, rows::AbstractVector, cols::AbstractVector)

    _obj_hess_structure!(m.objs, rows, cols)
    _con_hess_structure!(m.cons, rows, cols)
end

_obj_hess_structure!(objs::ObjectiveNull, rows, cols) = nothing
function _obj_hess_structure!(objs, rows, cols)
    _obj_hess_structure!(objs.inner, rows, cols)
    shessian!(rows, cols, objs, nothing, NaN, NaN)
end

_con_hess_structure!(cons::ConstraintNull, rows, cols) = nothing
function _con_hess_structure!(cons, rows, cols)
    _con_hess_structure!(cons.inner, rows, cols)
    shessian!(rows, cols, cons, nothing, NaN, NaN)
end

function obj(m::ExaModel, x::AbstractVector)
    _obj(m.objs, x)
end

_obj(objs, x) = _obj(objs.inner, x) + sum(objs.f.f(k, x) for k in objs.itr)
_obj(objs::ObjectiveNull, x) = zero(eltype(x))

function cons_nln!(m::ExaModel, x::AbstractVector, g::AbstractVector)

    fill!(g, zero(eltype(g)))
    _cons_nln!(m.cons, x, g)
end

function _cons_nln!(cons, x, g)
    _cons_nln!(cons.inner, x, g)
    @simd for i in eachindex(cons.itr)
        g[offset0(cons, i)] += cons.f.f(cons.itr[i], x)
    end
end
_cons_nln!(cons::ConstraintNull, x, g) = nothing



function grad!(m::ExaModel, x::AbstractVector, f::AbstractVector)

    fill!(f, zero(eltype(f)))
    _grad!(m.objs, x, f)
    return f
end

function _grad!(objs, x, f)
    _grad!(objs.inner, x, f)
    gradient!(f, objs, x, one(eltype(f)))
end
_grad!(objs::ObjectiveNull, x, f) = nothing

function jac_coord!(m::ExaModel, x::AbstractVector, jac::AbstractVector)

    fill!(jac, zero(eltype(jac)))
    _jac_coord!(m.cons, x, jac)

end

_jac_coord!(cons::ConstraintNull, x, jac) = nothing
function _jac_coord!(cons, x, jac)
    _jac_coord!(cons.inner, x, jac)
    sjacobian!(jac, nothing, cons, x, one(eltype(jac)))
end

function jprod_nln!(m::ExaModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)

    fill!(Jv, zero(eltype(Jv)))
    _jprod_nln!(m.cons, x, v, Jv)

end

_jprod_nln!(cons::ConstraintNull, x, v, Jv) = nothing
function _jprod_nln!(cons, x, v, Jv)
    _jprod_nln!(cons.inner, x, v, Jv)
    sjacobian!((Jv, v), nothing, cons, x, one(eltype(Jv)))
end

function jtprod_nln!(m::ExaModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)

    fill!(Jtv, zero(eltype(Jtv)))
    _jtprod_nln!(m.cons, x, v, Jtv)

end

_jtprod_nln!(cons::ConstraintNull, x, v, Jtv) = nothing
function _jtprod_nln!(cons, x, v, Jtv)
    _jtprod_nln!(cons.inner, x, v, Jtv)
    sjacobian!(nothing, (Jtv, v), cons, x, one(eltype(Jtv)))
end

function hess_coord!(
    m::ExaModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)

    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.objs, x, y, hess, obj_weight)
    _con_hess_coord!(m.cons, x, y, hess, obj_weight)

    return hess
end
_obj_hess_coord!(objs::ObjectiveNull, x, y, hess, obj_weight) = nothing
function _obj_hess_coord!(objs, x, y, hess, obj_weight)
    _obj_hess_coord!(objs.inner, x, y, hess, obj_weight)
    shessian!(hess, nothing, objs, x, obj_weight, zero(eltype(hess)))
end

_con_hess_coord!(cons::ConstraintNull, x, y, hess, obj_weight) = nothing
function _con_hess_coord!(cons, x, y, hess, obj_weight)
    _con_hess_coord!(cons.inner, x, y, hess, obj_weight)
    shessian!(hess, nothing, cons, x, y, zero(eltype(hess)))
end

function hprod!(
    m::ExaModel,
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
)

    fill!(Hv, zero(eltype(Hv)))
    _obj_hprod!(m.objs, x, y, v, Hv, obj_weight)
    _con_hprod!(m.cons, x, y, v, Hv, obj_weight)

    return Hv
end

_obj_hprod!(objs::ObjectiveNull, x, y, v, Hv, obj_weight) = nothing
function _obj_hprod!(objs, x, y, v, Hv, obj_weight)
    _obj_hprod!(objs.inner, x, y, v, Hv, obj_weight)
    shessian!((Hv, v), nothing, objs, x, obj_weight, zero(eltype(Hv)))
end

_con_hprod!(cons::ConstraintNull, x, y, v, Hv, obj_weight) = nothing
function _con_hprod!(cons, x, y, v, Hv, obj_weight)
    _con_hprod!(cons.inner, x, y, v, Hv, obj_weight)
    shessian!((Hv, v), nothing, cons, x, y, zero(eltype(Hv)))
end

@inbounds @inline offset0(a, i) = offset0(a.f, i)
@inbounds @inline offset1(a, i) = offset1(a.f, i)
@inbounds @inline offset2(a, i) = offset2(a.f, i)
@inbounds @inline offset0(f, itr, i) = offset0(f, i)
@inbounds @inline offset0(f::F, i) where {F<:SIMDFunction} = f.o0 + i
@inbounds @inline offset1(f::F, i) where {F<:SIMDFunction} = f.o1 + f.o1step * (i - 1)
@inbounds @inline offset2(f::F, i) where {F<:SIMDFunction} = f.o2 + f.o2step * (i - 1)
@inbounds @inline offset0(a::C, i) where {C<:ConstraintAug} = offset0(a.f, a.itr, i)
@inbounds @inline offset0(f::F, itr, i) where {P<:Pair,F<:SIMDFunction{P}} =
    f.o0 + f.f.first(itr[i], nothing)
@inbounds @inline offset0(f::F, itr, i) where {T<:Tuple,P<:Pair{T},F<:SIMDFunction{P}} =
    f.o0 + idxx(coord(itr, i, f.f.first), Base.size(itr))

@inline idx(itr, I) = @inbounds itr[I]
@inline idx(itr::Base.Iterators.ProductIterator{V}, I) where {V} =
    _idx(I - 1, itr.iterators, Base.size(itr))
@inline function _idx(n, (vec1, vec...), (si1, si...))
    d, r = divrem(n, si1)
    return (vec1[r+1], _idx(d, vec, si)...)
end
@inline _idx(n, (vec,), ::Tuple{Int}) = @inbounds vec[n+1]

@inline idxx(coord, si) = _idxx(coord, si, 1) + 1
@inline _idxx(coord, si, a) = a * (coord[1] - 1) + _idxx(coord[2:end], si[2:end], a * si[1])
@inline _idxx(::Tuple{}, ::Tuple{}, a) = 0

@inline coord(itr, i, (f, fs...)) = (f(idx(itr, i), nothing), coord(itr, i, fs)...)
@inline coord(itr, i, ::Tuple{}) = ()

for (thing, val) in [(:solution, 1), (:multipliers_L, 0), (:multipliers_U, 2)]
    @eval begin
        """
            $(string($thing))(result, x)

        Returns the $(string($thing)) for variable `x` associated with `result`, obtained by solving the model.

        ## Example
        ```jldoctest
        julia> using ExaModels, NLPModelsIpopt

        julia> c = ExaCore();                     

        julia> x = variable(c, 1:10, lvar = -1, uvar = 1);

        julia> objective(c, (x[i]-2)^2 for i in 1:10);

        julia> m = ExaModel(c);                   

        julia> result = ipopt(m; print_level=0);

        julia> val = $(string($thing))(result, x);

        julia> isapprox(val, fill($(string($val)), 10), atol=sqrt(eps(Float64)), rtol=Inf)
        true
        ```
        """
        function $thing(result::SolverCore.AbstractExecutionStats, x)

            o = x.offset
            len = total(x.size)
            s = size(x.size)
            return reshape(view(result.$thing, o+1:o+len), s...)
        end
    end
end

solution(result::SolverCore.AbstractExecutionStats, x::Var{I}) where {I} =
    return result.solution[x.i]


"""
    multipliers(result, y)

Returns the multipliers for constraints `y` associated with `result`, obtained by solving the model.

## Example
```jldoctest
julia> using ExaModels, NLPModelsIpopt

julia> c = ExaCore();                     

julia> x = variable(c, 1:10, lvar = -1, uvar = 1);

julia> objective(c, (x[i]-2)^2 for i in 1:10);

julia> y = constraint(c, x[i] + x[i+1] for i=1:9; lcon = -1, ucon = (1+i for i=1:9));

julia> m = ExaModel(c);                   

julia> result = ipopt(m; print_level=0);

julia> val = multipliers(result, y);


julia> val[1] ≈ 0.81933930
true
```
"""
function multipliers(result::SolverCore.AbstractExecutionStats, y::Constraint)
    o = y.offset
    len = length(y.itr)
    return view(result.multipliers, o+1:o+len)
end


_adapt_gen(gen) = Base.Generator(gen.f, collect(gen.iter))
_adapt_gen(gen::Base.Generator{P}) where {P<:Union{AbstractArray,AbstractRange}} = gen
