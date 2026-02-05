abstract type AbstractVariable end
abstract type AbstractParameter end
abstract type AbstractConstraint end
abstract type AbstractExpression end
abstract type AbstractObjective end

struct VariableNull <: AbstractVariable end
struct ParameterNull <: AbstractParameter end
struct ObjectiveNull <: AbstractObjective end
struct ExpressionNull <: AbstractExpression end
struct ConstraintNull <: AbstractConstraint end

struct Variable{S, O} <: AbstractVariable
    size::S
    length::O
    offset::O
end
Base.show(io::IO, v::Variable) = print(
    io,
    """
    Variable

      x ∈ R^{$(join(size(v.size), " × "))}
    """,
)

struct Parameter{S, O} <: AbstractParameter
    size::S
    length::O
    offset::O
end
Base.show(io::IO, v::Parameter) = print(
    io,
    """
    Parameter

      θ ∈ R^{$(join(size(v.size), " × "))}
    """,
)
struct Objective{R, F, I} <: AbstractObjective
    inner::R
    f::F
    itr::I
end
Base.show(io::IO, v::Objective) = print(
    io,
    """
    Objective

      min (...) + ∑_{p ∈ P} f(x,θ,p)

      where |P| = $(length(v.itr))
    """,
)


struct Expression{R, F, I, O, S} <: AbstractExpression
    inner::R
    f::F
    itr::I
    offset::O
    size::S
end
Base.show(io::IO, v::Expression) = print(
    io,
    """
    Expression

      s.t. (...)
           g♭ ≤ [g(x,θ,p)]_{p ∈ P} ≤ g♯

      where |P| = $(length(v.itr))
    """,
)


struct Constraint{R, F, I, O} <: AbstractConstraint
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
           g♭ ≤ [g(x,θ,p)]_{p ∈ P} ≤ g♯

      where |P| = $(length(v.itr))
    """,
)

struct ExpressionAug{R, F, I} <: AbstractConstraint
    inner::R
    f::F
    itr::I
    oa::Int
end

struct ConstraintAug{R, F, I} <: AbstractConstraint
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
           g♭ ≤ (...) + ∑_{p ∈ P} h(x,θ,p) ≤ g♯

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
Base.@kwdef mutable struct ExaCore{
        T,
        VT <: AbstractVector{T},
        VI <: AbstractVector{UInt},
        B,
        MI <: AbstractVector{AbstractVector{Any}},
        VII <: AbstractVector{Tuple{UInt, UInt}},
    }
    backend::B = nothing
    obj::AbstractObjective = ObjectiveNull()
    con::AbstractConstraint = ConstraintNull()
    exp::AbstractExpression = ExpressionNull()
    # corresponds to y1 and y2 in _simdfunction()
    full_exp_refs1::MI = AbstractVector{Any}[]
    full_exp_refs2::MI = AbstractVector{Any}[]
    e1_starts::VII = convert_array(Tuple{UInt, UInt}[], backend)
    e2_starts::VII = convert_array(Tuple{UInt, UInt}[], backend)
    e1_cnts::VI = convert_array(UInt[], backend)
    e2_cnts::VI = convert_array(UInt[], backend)
    e1_len::Int = 0
    e2_len::Int = 0
    varis::Vector{UInt} = UInt[]
    isexp::VI = convert_array(UInt[], backend)
    nvar::Int = 0
    npar::Int = 0
    nexp::Int = 0
    ncon::Int = 0
    nconaug::Int = 0
    nobj::Int = 0
    nnzc::Int = 0
    nnzg::Int = 0
    nnzj::Int = 0
    nnzh::Int = 0
    x0::VT = convert_array(zeros(0), backend)
    θ::VT = similar(x0, 0)
    lvar::VT = similar(x0)
    uvar::VT = similar(x0)
    y0::VT = similar(x0)
    lcon::VT = similar(x0)
    ucon::VT = similar(x0)
    minimize::Bool = true
    # Parameter subexpression support
    nparam_subexpr::Int = 0
    param_subexpr_values::VT = similar(x0, 0)
    param_subexpr_fns::Vector{Any} = Any[]
end

# Deprecated as of v0.7
function ExaCore(::Type{T}, backend) where {T <: AbstractFloat}
    @warn "ExaCore(T, backend) is deprecated. Use ExaCore(T; backend = backend) instead"
    return ExaCore(T; backend = backend)
end
function ExaCore(backend)
    @warn "ExaCore(backend) is deprecated. Use ExaCore(T; backend = backend) instead"
    return ExaCore(; backend = backend)
end

ExaCore(::Type{T}; backend = nothing, kwargs...) where {T <: AbstractFloat} =
    ExaCore(x0 = convert_array(zeros(T, 0), backend); backend = backend, kwargs...)

depth(a) = depth(a.inner) + 1
depth(a::ObjectiveNull) = 0
depth(a::ConstraintNull) = 0
depth(a::ExpressionNull) = 0

Base.show(io::IO, c::ExaCore{T, VT, B}) where {T, VT, B} = print(
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

struct ExaModel{T, VT, VI, E, O, C, EX, VII} <: NLPModels.AbstractNLPModel{T, VT}
    objs::O
    cons::C
    exps::EX
    varis::Vector{UInt}
    isexp::VI
    nexp::Int
    e1_starts::VII
    e2_starts::VII
    # index referenced by e1_starts is length
    # next length elements are how much to increment cnt by
    # sum of all cnts in a range = full cnt of expr - 1
    e1_cnts::VI
    e2_cnts::VI
    e1::VT
    e2::VT
    θ::VT
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    ext::E
end

function Base.show(io::IO, c::ExaModel{T, VT}) where {T, VT}
    println(io, "An ExaModel{$T, $VT, ...}\n")
    return Base.show(io, c.meta)
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

julia> m = ExaModel(c)                     # create an ExaModel object
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
                                                     lin_nnzj: (------% sparsity)
                                                     nln_nnzj: (------% sparsity)

julia> using NLPModelsIpopt

julia> result = ipopt(m; print_level=0)    # solve the problem
"Execution stats: first-order stationary"

```
"""
function ExaModel(c::C; prod = nothing) where {C <: ExaCore}
    return ExaModel(
        c.obj,
        c.con,
        c.exp,
        c.varis,
        c.isexp,
        c.nexp,
        c.e1_starts,
        c.e2_starts,
        c.e1_cnts,
        c.e2_cnts,
        convert_array(zeros(c.e1_len), c.backend),
        convert_array(zeros(c.e2_len), c.backend),
        c.θ,
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

@inline function Base.getindex(v::V, i) where {V <: Variable}
    _bound_check(v.size, i)
    return Var(i + (v.offset - _start(v.size[1]) + 1))
end
@inline function Base.getindex(v::V, is...) where {V <: Variable}
    @assert(length(is) == length(v.size), "Variable index dimension error")
    _bound_check(v.size, is)
    return Var(v.offset + idxx(is .- (_start.(v.size) .- 1), _length.(v.size)))
end

@inline function Base.getindex(e::E, i) where {E <: Expression}
    _bound_check(e.size, i)
    return Var(i + (e.offset - _start(e.size[1]) + 1))
end
@inline function Base.getindex(e::E, is...) where {E <: Expression}
    @assert(length(is) == length(e.size), "Expression index dimension error. Got $(length(is)) dimensions, expected $(length(e.size)).")
    _bound_check(e.size, is)
    return Var(e.offset + idxx(is .- (_start.(e.size) .- 1), _length.(e.size)))
end

@inline function Base.getindex(p::P, i) where {P <: Parameter}
    _bound_check(p.size, i)
    return ParameterNode(i + (p.offset - _start(p.size[1]) + 1))
end
@inline function Base.getindex(p::P, is...) where {P <: Parameter}
    @assert(length(is) == length(p.size), "Parameter index dimension error")
    _bound_check(p.size, is)
    return ParameterNode(p.offset + idxx(is .- (_start.(p.size) .- 1), _length.(p.size)))
end


function _bound_check(sizes, i::I) where {I <: Integer}
    return __bound_check(sizes[1], i)
end
function _bound_check(sizes, is::NTuple{N, I}) where {I <: Integer, N}
    __bound_check(sizes[1], is[1])
    return _bound_check(sizes[2:end], is[2:end])
end
_bound_check(sizes, is) = nothing
_bound_check(sizes, is::Tuple{}) = nothing

function __bound_check(a::I, b::I) where {I <: Integer}
    return @assert(1 <= b <= a, "Variable index bound error")
end
function __bound_check(a::UnitRange{Int}, b::I) where {I <: Integer}
    return @assert(b in a, "Variable index bound error")
end

function append!(backend, a, b::Base.Generator, lb)
    b = _adapt_gen(b)

    la = length(a)
    resize!(a, la + lb)
    map!(b.f, view(a, (la + 1):(la + lb)), convert_array(b.iter, backend))
    return a
end

function append!(backend, a, b::Base.Generator{UnitRange{I}}, lb) where {I}

    la = length(a)
    resize!(a, la + lb)
    map!(b.f, view(a, (la + 1):(la + lb)), b.iter)
    return a
end

function append!(backend, a, b::AbstractArray, lb)

    la = length(a)
    resize!(a, la + lb)
    map!(identity, view(a, (la + 1):(la + lb)), convert_array(b, backend))
    return a
end

function append!(backend, a, b::Number, lb)

    la = length(a)
    resize!(a, la + lb)
    fill!(view(a, (la + 1):(la + lb)), b)
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
    ) where {T, C <: ExaCore{T}}
    o = c.nvar
    len = total(ns)
    c.nvar += len
    c.varis = vcat(c.varis, (o + 1):c.nvar)
    append!(c.backend, c.isexp, 0, len)
    append!(c.backend, c.e1_starts, [(0, 0) for _ in 1:len], len)
    append!(c.backend, c.e2_starts, [(0, 0) for _ in 1:len], len)
    c.x0 = append!(c.backend, c.x0, start, total(ns))
    c.lvar = append!(c.backend, c.lvar, lvar, total(ns))
    c.uvar = append!(c.backend, c.uvar, uvar, total(ns))

    return Variable(ns, len, o)
end

"""
    parameter(core, start::AbstractArray)

Adds parameters with initial values specified by `start`, and returns `Parameter` object.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();

julia> θ = parameter(c, ones(10))
Parameter

  θ ∈ R^{10}
```
"""
function parameter(c::C, start::AbstractArray) where {T, C <: ExaCore{T}}

    ns = Base.size(start)
    o = c.npar
    len = total(ns)
    c.npar += len
    c.θ = append!(c.backend, c.θ, start, len)
    return Parameter(ns, len, o)

end

"""
    set_parameter!(core, param, values)

Updates the values of parameters in the core.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();

julia> p = parameter(c, ones(5))
Parameter

  θ ∈ R^{5}

julia> set_parameter!(c, p, rand(5))  # Update with new values
```
"""
function set_parameter!(c::ExaCore, param::Parameter, values::AbstractArray)
    if Base.size(values) != param.size
        throw(
            DimensionMismatch(
                "Parameter size mismatch: expected $(param.size), got $(Base.size(values))",
            ),
        )
    end

    start_idx = param.offset + 1
    end_idx = param.offset + param.length

    copyto!(@view(c.θ[start_idx:end_idx]), values)

    # Re-evaluate parameter subexpressions that depend on θ
    _recompute_param_subexprs!(c)

    return nothing
end

"""
    _recompute_param_subexprs!(c::ExaCore)

Re-evaluates all parameter-only subexpressions and updates their cached values in θ.
Called automatically by `set_parameter!`.
"""
function _recompute_param_subexprs!(c::ExaCore)
    for ps in c.param_subexpr_fns
        # Re-evaluate the subexpression with current θ values
        # Note: θ contains both user parameters and param subexpr values
        # The eval function only reads user parameters (earlier in θ)
        new_values = ps.fn(c.θ)
        start_idx = ps.offset + 1
        end_idx = ps.offset + ps.length
        copyto!(@view(c.θ[start_idx:end_idx]), new_values)
    end
    return nothing
end

function variable(c::C; kwargs...) where {T, C <: ExaCore{T}}

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

  min (...) + ∑_{p ∈ P} f(x,θ,p)

  where |P| = 10
```
"""
function objective(c::C, gen) where {C <: ExaCore}
    gen = _adapt_gen(gen)
    f = SIMDFunction(gen, c.full_exp_refs1, c.full_exp_refs2, c.exp, c.isexp, c.nobj, c.nnzg, c.nnzh)
    pars = gen.iter

    return _objective(c, f, pars)
end

"""
    objective(core::ExaCore, expr [, pars])

Adds objective terms specified by a `expr` and `pars` to `core`, and returns an `Objective` object.
"""
function objective(c::C, expr::N, pars = 1:1) where {C <: ExaCore, N <: AbstractNode}
    f = _simdfunction(expr, c.full_exp_refs1, c.full_exp_refs2, c.exp, c.isexp, c.nobj, c.nnzg, c.nnzh)

    return _objective(c, f, pars)
end

function _objective(c, f, pars)
    nitr = length(pars)
    c.nobj += nitr
    c.nnzg += nitr * f.o1step
    c.nnzh += nitr * f.o2step

    return c.obj = Objective(c.obj, f, convert_array(pars, c.backend))
end

"""
    constraint(core, generator; start = 0, lcon = 0,  ucon = 0)

Adds constraints specified by a `generator` to `core`, and returns an `Constraint` object.

## Keyword Arguments
- `start`: The initial guess of the dual solution. Can either be `Number`, `AbstractArray`, or `Generator`.
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
       g♭ ≤ [g(x,θ,p)]_{p ∈ P} ≤ g♯

  where |P| = 9
```
"""
function constraint(
        c::C,
        gen::Base.Generator;
        start = zero(T),
        lcon = zero(T),
        ucon = zero(T),
    ) where {T, C <: ExaCore{T}}

    gen = _adapt_gen(gen)
    f = SIMDFunction(gen, c.full_exp_refs1, c.full_exp_refs2, c.exp, c.isexp, c.ncon, c.nnzj, c.nnzh)
    pars = gen.iter

    return _constraint(c, f, pars, start, lcon, ucon)
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
    ) where {T, C <: ExaCore{T}, N <: AbstractNode}

    f = _simdfunction(expr, c.full_exp_refs1, c.full_exp_refs2, c.exp, c.isexp, c.ncon, c.nnzj, c.nnzh)

    return _constraint(c, f, pars, start, lcon, ucon)
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
    ) where {T, C <: ExaCore{T}}

    f = _simdfunction(Null(), c.full_exp_refs1, c.full_exp_refs2, c.exp, c.isexp, c.ncon, c.nnzj, c.nnzh)

    return _constraint(c, f, 1:n, start, lcon, ucon)
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

    return c.con = Constraint(c.con, f, convert_array(pars, c.backend), o)
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
       g♭ ≤ (...) + ∑_{p ∈ P} h(x,θ,p) ≤ g♯

  where |P| = 3
```
"""
function constraint!(c::C, c1, gen::Base.Generator) where {C <: ExaCore}

    gen = _adapt_gen(gen)
    f = SIMDFunction(gen, c.full_exp_refs1, c.full_exp_refs2, c.exp, c.isexp, offset0(c1, 0), c.nnzj, c.nnzh)
    pars = gen.iter

    return _constraint!(c, f, pars)
end

"""
    constraint!(c, c1, expr, pars)

Expands the existing constraint `c1` in `c` by adding addtional constraints terms specified by `expr` and `pars`.
"""
function constraint!(c::C, c1, expr, pars) where {C <: ExaCore}
    f = _simdfunction(expr, c.full_exp_refs1, c.full_exp_refs2, c.exp, c.isexp, offset0(c1, 0), c.nnzj, c.nnzh)

    return _constraint!(c, f, pars)
end

function _constraint!(c, f, pars)
    oa = c.nconaug

    nitr = length(pars)

    c.nconaug += nitr
    c.nnzj += nitr * f.o1step
    c.nnzh += nitr * f.o2step

    return c.con = ConstraintAug(c.con, f, convert_array(pars, c.backend), oa)
end


"""
    subexpr(core, generator)

Adds epressions specified by a `generator` to `core`, and returns an `Expression` object.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();

julia> x = variable(c, 10);

julia> e = subexpr(c, x[i] for i=1:9)
Expression
    e ≤ [g(x,θ,p)]_{p ∈ P}

  where |P| = 9
```
"""
function subexpr(
        c::C,
        gen::I,
    ) where {T, C <: ExaCore{T}, I<:Base.Iterators.Flatten}
    ns=[]
    it = gen.it
    while typeof(it) <: Union{Base.Generator, Base.Iterators.Flatten}
        push!(ns, length(it))
        (it, _) = Base.iterate(it)
    end
    subexpr(c, (nsi for nsi in ns), gen)
end
subexpr(c::C, gen::G) where {T, C <: ExaCore{T}, G<:Base.Generator} = subexpr(c, (length(gen.iter),), gen)

function subexpr(
        c::C,
        ns::S,
        gen::Base.Generator,
    ) where {T, C <: ExaCore{T}, S}
    gen = _adapt_gen(gen)
    f = simd_expr(c, gen)
    pars = gen.iter
    nitr = length(pars)
    o = c.nvar
    c.nvar += nitr
    append!(c.backend, c.isexp, (1 + c.nexp):(nitr + c.nexp), nitr)
    c.nexp += nitr
    c.nconaug += nitr
    start = convert_array(zeros(nitr), c.backend)
    lvar = convert_array(zeros(nitr), c.backend)
    uvar = convert_array(zeros(nitr), c.backend)
    # TODO: this fails if lvar / uvar infinite and f is trig (for example)
    #@simd for i in 1:nitr
    #    start[i] = f.f(pars[i], c.x0, c.θ)
    #    lvar[i] = f.f(pars[i], c.lvar, c.θ)
    #    uvar[i] = f.f(pars[i], c.uvar, c.θ)
    #end
    c.x0 = append!(c.backend, c.x0, start, nitr)
    c.lvar = append!(c.backend, c.lvar, lvar, nitr)
    c.uvar = append!(c.backend, c.uvar, uvar, nitr)
    c.exp = Expression(c.exp, f, convert_array(pars, c.backend), o, ns)
    return c.exp
end

function simd_expr(c::ExaCore, gen)
    f = gen.f(ParSource())
    nitr = length(gen.iter)

    y1_raw = []
    d = f(Identity(), AdjointNodeSource(nothing, nothing), nothing)
    ExaModels.grpass(d, nothing, y1_raw, nothing, 0, NaN)
    y1 = get_full_exp_refs(c.full_exp_refs1, c.exp, c.isexp, y1_raw)
    push!(c.full_exp_refs1, y1)

    y2_raw = []
    t = f(Identity(), SecondAdjointNodeSource(nothing, nothing), nothing)
    ExaModels.hrpass(nothing, nothing, nothing, nothing, nothing, nothing, nothing, t, nothing, y2_raw, nothing, nothing, 0, NaN, NaN)
    y2 = get_full_exp_refs(c.full_exp_refs2, c.exp, c.isexp, y2_raw)
    push!(c.full_exp_refs2, y2)

    a1 = unique(y1)
    o1step = length(a1)
    e1_cnts = compress_ref_cnts(y1, a1)
    c1 = Compressor(Tuple(findfirst(isequal(di), a1) for di in y1))
    append!(
        c.backend, c.e1_starts, [
            (length(c.e1_cnts) + 1, (i - 1) * o1step + c.e1_len + 1) for i in 1:nitr
        ], nitr
    )
    o1 = c.e1_len
    c.e1_len += nitr * o1step
    push!(c.e1_cnts, o1step)
    append!(c.backend, c.e1_cnts, e1_cnts, o1step)

    a2 = unique(y2)
    e2_cnts = compress_ref_cnts(y2, a2)
    o2step = length(a2)
    c2 = Compressor(Tuple(findfirst(isequal(di), a2) for di in y2))
    append!(
        c.backend, c.e2_starts, [
            (length(c.e2_cnts) + 1, (i - 1) * o2step + c.e2_len + 1) for i in 1:nitr
        ], nitr
    )
    o2 = c.e2_len
    c.e2_len += nitr * o2step
    push!(c.e2_cnts, o2step)
    append!(c.backend, c.e2_cnts, e2_cnts, length(e2_cnts))

    return SIMDFunction(f, c1, c2, c.nvar, o1, o2, o1step, o2step)
end

expr!(m, x, θ) = _expr!(m.exps, m, x, θ)
function _expr!(expr, m, x, θ)
    _expr!(expr.inner, m, x, θ)
    return @simd for i in eachindex(expr.itr)
        x[offset0(expr, i)] = expr.f(expr.itr[i], x, θ)
    end
end
_expr!(expr::ExpressionNull, m, x, θ) = nothing

function jac_structure!(m::ExaModel, rows::AbstractVector, cols::AbstractVector)
    e1_uint = reinterpret(UInt, m.e1)
    fill!(e1_uint, zero(UInt))
    _jac_structure!(m.exps, m, e1_uint, nothing, e1_uint)
    _jac_structure!(m.cons, m, e1_uint, rows, cols)
    return rows, cols
end

_jac_structure!(cons::ExpressionNull, m, e1_uint, rows, cols) = nothing
_jac_structure!(cons::ConstraintNull, m, e1_uint, rows, cols) = nothing
function _jac_structure!(f, m, e1_uint, rows, cols)
    _jac_structure!(f.inner, m, e1_uint, rows, cols)
    return sjacobian!(e1_uint, m.e1_starts, m.e1_cnts, m.isexp, rows, cols, f, nothing, nothing, NaN)
end

function hess_structure!(m::ExaModel, rows::AbstractVector, cols::AbstractVector)
    e1_uint = reinterpret(UInt, m.e1)
    e2_uint = reinterpret(UInt, m.e2)
    fill!(e1_uint, zero(UInt))
    fill!(e2_uint, zero(UInt))
    # Expression structures are already computed during model construction
    # Just run structure pass on the expressions to populate e1_uint/e2_uint
    _jac_structure!(m.exps, m, e1_uint, nothing, e1_uint)  # Populates e1_uint indices
    _exp_hess_structure!(m.exps, m, e2_uint)  # Populates e2_uint indices
    _obj_hess_structure!(m.objs, m, rows, cols, e1_uint, e2_uint)
    _con_hess_structure!(m.cons, m, rows, cols, e1_uint, e2_uint)
    return rows, cols
end

_exp_hess_structure!(exps::ExpressionNull, m, e2_uint) = nothing
function _exp_hess_structure!(exps, m, e2_uint)
    _exp_hess_structure!(exps.inner, m, e2_uint)
    # For expression Hessian structure, we need to record the indices in e2_uint
    # similar to how jac_structure! records indices in e1_uint
    # This uses the same shessian! but with structure-mode output (Integer vectors)
    return shessian!(
        e2_uint, e2_uint, exps, nothing, nothing,
        reinterpret(UInt, m.e1), m.e1_starts, m.e1_cnts,
        e2_uint, m.e2_starts, m.e2_cnts,
        NaN, NaN, m.isexp
    )
end

_obj_hess_structure!(objs::ObjectiveNull, m, rows, cols, e1_uint, e2_uint) = nothing
function _obj_hess_structure!(objs, m, rows, cols, e1_uint, e2_uint)
    _obj_hess_structure!(objs.inner, m, rows, cols, e1_uint, e2_uint)
    return shessian!(rows, cols, objs, nothing, nothing, e1_uint, m.e1_starts, m.e1_cnts, e2_uint, m.e2_starts, m.e2_cnts, NaN, NaN, m.isexp)
end

_con_hess_structure!(cons::ConstraintNull, m, rows, cols, e1_uint, e2_uint) = nothing
function _con_hess_structure!(cons, m, rows, cols, e1_uint, e2_uint)
    _con_hess_structure!(cons.inner, m, rows, cols, e1_uint, e2_uint)
    return shessian!(rows, cols, cons, nothing, nothing, e1_uint, m.e1_starts, m.e1_cnts, e2_uint, m.e2_starts, m.e2_cnts, NaN, NaN, m.isexp)
end

function obj(m::ExaModel, x::AbstractVector)
    expr!(m, x, m.θ)
    return _obj(m.objs, x, m.θ)
end

_obj(objs, x, θ) =
    _obj(objs.inner, x, θ) +
    (isempty(objs.itr) ? zero(eltype(x)) : sum(objs.f(k, x, θ) for k in objs.itr))
_obj(objs::ObjectiveNull, x, θ) = zero(eltype(x))

function cons_nln!(m::ExaModel, x::AbstractVector, g::AbstractVector)
    expr!(m, x, m.θ)
    fill!(g, zero(eltype(g)))
    _cons_nln!(m.cons, x, m.θ, g)
    return g
end

function _cons_nln!(cons, x, θ, g)
    _cons_nln!(cons.inner, x, θ, g)
    return @simd for i in eachindex(cons.itr)
        g[offset0(cons, i)] += cons.f(cons.itr[i], x, θ)
    end
end
_cons_nln!(cons::ConstraintNull, x, θ, g) = nothing

function grad!(m::ExaModel, x::AbstractVector, out::AbstractVector)
    expr!(m, x, m.θ)
    fill!(out, zero(eltype(out)))
    _grad!(m.exps, m, x, out)
    _grad!(m.objs, m, x, out)
    return f
end

_grad!(f::ObjectiveNull, m, x, out) = nothing
_grad!(f::ExpressionNull, m, x, out) = nothing
function _grad!(f, m, x, out)
    _grad!(f.inner, m, x, out)
    return gradient!(m.isexp, m.e1, m.e1_starts, m.e1_cntsegrad, f, objs, x, θ, one(eltype(out)))
end

function jac_coord!(m::ExaModel, x::AbstractVector, jac::AbstractVector)
    expr!(m, x, m.θ)
    fill!(jac, zero(eltype(jac)))
    _jac_coord!(m.exps, x, m, m.e1)
    _jac_coord!(m.cons, x, m, jac)
    return jac
end

_jac_coord!(f::ConstraintNull, x, m, jac) = nothing
_jac_coord!(f::ExpressionNull, x, m, jac) = nothing
function _jac_coord!(f, x, m, jac)
    _jac_coord!(f.inner, x, m, jac)
    return sjacobian!(m.e1, m.e1_starts, m.e1_cnts, m.isexp, jac, nothing, f, x, m.θ, one(eltype(jac)))
end

function jprod_nln!(m::ExaModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    expr!(m, x, m.θ)
    fill!(Jv, zero(eltype(Jv)))
    _jprod_nln!(m.cons, m.isexp, m.eJv, nothing, x, m.θ, v, Jv)
    return Jv
end

_jprod_nln!(cons::ConstraintNull, isexp, ey1, ey2, x, θ, v, Jv) = nothing
function _jprod_nln!(cons, isexp, ey1, ey2, x, θ, v, Jv)
    _jprod_nln!(cons.inner, isexp, ey1, ey2, x, θ, v, Jv)
    return sjacobian!(isexp, ey1, ey2, (Jv, v), nothing, cons, x, θ, one(eltype(Jv)))
end

function jtprod_nln!(m::ExaModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    expr!(m, x, m.θ)
    fill!(Jtv, zero(eltype(Jtv)))
    _jtprod_nln!(m.cons, m.isexp, nothing, m.eJtv, x, m.θ, v, Jtv)
    return Jtv
end

_jtprod_nln!(cons::ConstraintNull, isexp, ey1, ey2, x, θ, v, Jtv) = nothing
function _jtprod_nln!(cons, isexp, ey1, ey2, x, θ, v, Jtv)
    _jtprod_nln!(cons.inner, isexp, ey1, ey2, x, θ, v, Jtv)
    return sjacobian!(isexp, ey1, ey2, nothing, (Jtv, v), cons, x, θ, one(eltype(Jtv)))
end

function hess_coord!(
        m::ExaModel,
        x::AbstractVector,
        hess::AbstractVector;
        obj_weight = one(eltype(x)),
    )
    fill!(hess, zero(eltype(hess)))
    fill!(m.e1, zero(eltype(m.e1)))
    fill!(m.e2, zero(eltype(m.e2)))
    expr!(m, x, m.θ)
    # First compute expression Jacobians (e1)
    _jac_coord!(m.exps, x, m, m.e1)
    # Then compute expression Hessians (e2)
    _exp_hess_coord!(m.exps, x, m)
    # Now compute objective Hessian
    _obj_hess_coord!(m.objs, x, m, hess, obj_weight)
    return hess
end

function hess_coord!(
        m::ExaModel,
        x::AbstractVector,
        y::AbstractVector,
        hess::AbstractVector;
        obj_weight = one(eltype(x)),
    )
    fill!(hess, zero(eltype(hess)))
    fill!(m.e1, zero(eltype(m.e1)))
    fill!(m.e2, zero(eltype(m.e2)))
    expr!(m, x, m.θ)
    # First compute expression Jacobians (e1)
    _jac_coord!(m.exps, x, m, m.e1)
    # Then compute expression Hessians (e2)
    _exp_hess_coord!(m.exps, x, m)
    # Now compute objective and constraint Hessians
    _obj_hess_coord!(m.objs, x, m, hess, obj_weight)
    _con_hess_coord!(m.cons, x, m, y, hess)
    return hess
end

_exp_hess_coord!(exps::ExpressionNull, x, m) = nothing
function _exp_hess_coord!(exps, x, m)
    _exp_hess_coord!(exps.inner, x, m)
    return shessian!(m.e2, nothing, exps, x, m.θ, m.e1, m.e1_starts, m.e1_cnts, m.e2, m.e2_starts, m.e2_cnts, one(eltype(m.e2)), zero(eltype(m.e2)), m.isexp)
end

_obj_hess_coord!(objs::ObjectiveNull, x, m, hess, w) = nothing
function _obj_hess_coord!(objs, x, m, hess, w)
    _obj_hess_coord!(objs.inner, x, m, hess, w)
    return shessian!(hess, nothing, objs, x, m.θ, m.e1, m.e1_starts, m.e1_cnts, m.e2, m.e2_starts, m.e2_cnts, w, zero(eltype(hess)), m.isexp)
end

_con_hess_coord!(cons::ConstraintNull, x, m, y, hess) = nothing
function _con_hess_coord!(cons, x, m, y, hess)
    _con_hess_coord!(cons.inner, x, m, y, hess)
    return shessian!(hess, nothing, cons, x, m.θ, m.e1, m.e1_starts, m.e1_cnts, m.e2, m.e2_starts, m.e2_cnts, y, zero(eltype(hess)), m.isexp)
end

function hprod!(
        m::ExaModel,
        x::AbstractVector,
        v::AbstractVector,
        Hv::AbstractVector;
        obj_weight = one(eltype(x)),
    )
    fill!(Hv, zero(eltype(Hv)))
    fill!(m.e1, zero(eltype(m.e1)))
    fill!(m.e2, zero(eltype(m.e2)))
    expr!(m, x, m.θ)
    _jac_coord!(m.exps, x, m, m.e1)
    _exp_hess_coord!(m.exps, x, m)
    _obj_hprod!(m.objs, x, m, v, Hv, obj_weight)
    return Hv
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
    fill!(m.e1, zero(eltype(m.e1)))
    fill!(m.e2, zero(eltype(m.e2)))
    expr!(m, x, m.θ)
    _jac_coord!(m.exps, x, m, m.e1)
    _exp_hess_coord!(m.exps, x, m)
    _obj_hprod!(m.objs, x, m, v, Hv, obj_weight)
    _con_hprod!(m.cons, x, m, y, v, Hv)
    return Hv
end

_obj_hprod!(objs::ObjectiveNull, x, m, v, Hv, obj_weight) = nothing
function _obj_hprod!(objs, x, m, v, Hv, obj_weight)
    _obj_hprod!(objs.inner, x, m, v, Hv, obj_weight)
    return shessian!((Hv, v), nothing, objs, x, m.θ, m.e1, m.e1_starts, m.e1_cnts, m.e2, m.e2_starts, m.e2_cnts, obj_weight, zero(eltype(Hv)), m.isexp)
end

_con_hprod!(cons::ConstraintNull, x, m, y, v, Hv) = nothing
function _con_hprod!(cons, x, m, y, v, Hv)
    _con_hprod!(cons.inner, x, m, y, v, Hv)
    return shessian!((Hv, v), nothing, cons, x, m.θ, m.e1, m.e1_starts, m.e1_cnts, m.e2, m.e2_starts, m.e2_cnts, y, zero(eltype(Hv)), m.isexp)
end

@inbounds @inline offset0(a, i) = offset0(a.f, i)
@inbounds @inline offset1(a, i) = offset1(a.f, i)
@inbounds @inline offset2(a, i) = offset2(a.f, i)
@inbounds @inline offset0(f, itr, i) = offset0(f, i)
@inbounds @inline offset0(f::F, i) where {F <: SIMDFunction} = f.o0 + i
@inbounds @inline offset1(f::F, i) where {F <: SIMDFunction} = f.o1 + f.o1step * (i - 1)
@inbounds @inline offset2(f::F, i) where {F <: SIMDFunction} = f.o2 + f.o2step * (i - 1)
@inbounds @inline offset0(a::C, i) where {C <: ConstraintAug} = offset0(a.f, a.itr, i)
@inbounds @inline offset0(f::F, itr, i) where {P <: Pair, F <: SIMDFunction{P}} =
    f.o0 + f.f.first(itr[i], nothing, nothing)
@inbounds @inline offset0(f::F, itr, i) where {I <: Integer, P <: Pair{I}, F <: SIMDFunction{P}} =
    f.o0 + f.f.first
@inbounds @inline offset0(f::F, itr, i) where {T <: Tuple, P <: Pair{T}, F <: SIMDFunction{P}} =
    f.o0 + idxx(coord(itr, i, f.f.first), Base.size(itr))

@inline idx(itr, I) = @inbounds itr[I]
@inline idx(itr::Base.Iterators.ProductIterator{V}, I) where {V} =
    _idx(I - 1, itr.iterators, Base.size(itr))
@inline function _idx(n, (vec1, vec...), (si1, si...))
    d, r = divrem(n, si1)
    return (vec1[r + 1], _idx(d, vec, si)...)
end
@inline _idx(n, (vec,), ::Tuple{Int}) = @inbounds vec[n + 1]

@inline idxx(coord, si) = _idxx(coord, si, 1) + 1
@inline _idxx(coord, si, a) = a * (coord[1] - 1) + _idxx(coord[2:end], si[2:end], a * si[1])
@inline _idxx(::Tuple{}, ::Tuple{}, a) = 0

@inline coord(itr, i, (f, fs...)) = (f(idx(itr, i), nothing, nothing), coord(itr, i, fs)...)
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
            return reshape(view(result.$thing, (o + 1):(o + len)), s...)
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
    return view(result.multipliers, (o + 1):(o + len))
end


_adapt_gen(gen) = Base.Generator(gen.f, collect(gen.iter))
_adapt_gen(gen::Base.Generator{P}) where {P <: Union{AbstractArray, AbstractRange}} = gen
