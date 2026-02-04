abstract type AbstractVariable end
abstract type AbstractParameter end
abstract type AbstractConstraint end
abstract type AbstractObjective end

struct VariableNull <: AbstractVariable end
struct ParameterNull <: AbstractParameter end
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

"""
    Subexpr

A subexpression that has been lifted to auxiliary variables with defining equality constraints.
Can be indexed like a Variable to get `Var` nodes for use in objectives and constraints.
"""
struct Subexpr{S, O, C} <: AbstractVariable
    size::S
    length::O
    offset::O
    constraint::C
end
Base.show(io::IO, s::Subexpr) = print(
    io,
    """
    Subexpression (lifted)

      s ∈ R^{$(join(size(s.size), " × "))}
    """,
)

"""
    ReducedSubexpr

A reduced-form subexpression that substitutes the expression directly when indexed.
No auxiliary variables or constraints are created - the expression is inlined.
"""
struct ReducedSubexpr{S, F, I}
    size::S
    length::Int
    f::F           # The generator function
    iter::I        # The collected iterator (for indexing)
end
Base.show(io::IO, s::ReducedSubexpr) = print(
    io,
    """
    Subexpression (reduced)

      s ∈ R^{$(join(size(s.size), " × "))}
    """,
)

"""
    ParameterSubexpr

A parameter-only subexpression whose values are computed once when parameters are set,
not at every function evaluation. Use this for expressions that depend only on parameters
(θ), not on variables (x). Values are automatically recomputed when `set_parameter!` is called.
"""
struct ParameterSubexpr{S, O}
    size::S
    length::O
    offset::O  # Index into ExaCore.param_subexpr_values
end
Base.show(io::IO, s::ParameterSubexpr) = print(
    io,
    """
    Subexpression (parameter-only)

      s ∈ R^{$(join(size(s.size), " × "))}
    """,
)

struct Parameter{S,O} <: AbstractParameter
    size::S
    length::O
    offset::O
end
Base.show(io::IO, v::Parameter) = print(
    io,
    """
Parameter

  θ ∈ R^{$(join(size(v.size)," × "))}
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

  min (...) + ∑_{p ∈ P} f(x,θ,p)

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
       g♭ ≤ [g(x,θ,p)]_{p ∈ P} ≤ g♯

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
Base.@kwdef mutable struct ExaCore{T,VT<:AbstractVector{T},B}
    backend::B = nothing
    obj::AbstractObjective = ObjectiveNull()
    con::AbstractConstraint = ConstraintNull()
    nvar::Int = 0
    npar::Int = 0
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
    θ::VT
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
function ExaModel(c::C; prod = nothing) where {C<:ExaCore}
    return ExaModel(
        c.obj,
        c.con,
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

@inline function Base.getindex(v::V, i) where {V<:Variable}
    _bound_check(v.size, i)
    Var(i + (v.offset - _start(v.size[1]) + 1))
end
@inline function Base.getindex(v::V, is...) where {V<:Variable}
    @assert(length(is) == length(v.size), "Variable index dimension error")
    _bound_check(v.size, is)
    Var(v.offset + idxx(is .- (_start.(v.size) .- 1), _length.(v.size)))
end

# Subexpr indexing - delegates to same logic as Variable
@inline function Base.getindex(s::Subexpr, i)
    _bound_check(s.size, i)
    return Var(i + (s.offset - _start(s.size[1]) + 1))
end
@inline function Base.getindex(s::Subexpr, is...)
    @assert(length(is) == length(s.size), "Subexpression index dimension error")
    _bound_check(s.size, is)
    return Var(s.offset + idxx(is .- (_start.(s.size) .- 1), _length.(s.size)))
end

# ReducedSubexpr indexing - evaluates the expression directly
# For concrete indices, look up the iterator element and apply f
# For symbolic indices (during expression building), create symbolic iterator elements
@inline function Base.getindex(s::ReducedSubexpr, i::I) where {I <: Integer}
    _bound_check(s.size, i)
    idx = i - _start(s.size[1]) + 1
    return s.f(s.iter[idx])
end
@inline function Base.getindex(s::ReducedSubexpr, i)
    # Symbolic index case - the symbolic index IS the iterator element
    # No adjustment needed; the index is used directly in expression building
    return s.f(i)
end
@inline function Base.getindex(s::ReducedSubexpr, is::Vararg{I, N}) where {I <: Integer, N}
    @assert(length(is) == length(s.size), "ReducedSubexpr index dimension error")
    _bound_check(s.size, is)
    idx = idxx(is .- (_start.(s.size) .- 1), _length.(s.size))
    return s.f(s.iter[idx])
end
@inline function Base.getindex(s::ReducedSubexpr, is...)
    # Symbolic indices case - the symbolic indices ARE the iterator elements
    # No adjustment needed; the indices are used directly in expression building
    @assert(length(is) == length(s.size), "ReducedSubexpr index dimension error")
    return s.f(is)
end

# ParameterSubexpr indexing - returns ParameterNode pointing to cached values in θ
@inline function Base.getindex(s::ParameterSubexpr, i::I) where {I <: Integer}
    _bound_check(s.size, i)
    idx = i - _start(s.size[1]) + 1
    return ParameterNode(s.offset + idx)
end
@inline function Base.getindex(s::ParameterSubexpr, i)
    # Symbolic index case - compute offset symbolically
    return ParameterNode(s.offset + (i - _start(s.size[1]) + 1))
end
@inline function Base.getindex(s::ParameterSubexpr, is::Vararg{I, N}) where {I <: Integer, N}
    @assert(length(is) == length(s.size), "ParameterSubexpr index dimension error")
    _bound_check(s.size, is)
    idx = idxx(is .- (_start.(s.size) .- 1), _length.(s.size))
    return ParameterNode(s.offset + idx)
end
@inline function Base.getindex(s::ParameterSubexpr, is...)
    # Symbolic indices case - compute offset symbolically
    @assert(length(is) == length(s.size), "ParameterSubexpr index dimension error")
    idx = idxx(is .- (_start.(s.size) .- 1), _length.(s.size))
    return ParameterNode(s.offset + idx)
end

@inline function Base.getindex(p::P, i) where {P<:Parameter}
    _bound_check(p.size, i)
    ParameterNode(i + (p.offset - _start(p.size[1]) + 1))
end
@inline function Base.getindex(p::P, is...) where {P<:Parameter}
    @assert(length(is) == length(p.size), "Parameter index dimension error")
    _bound_check(p.size, is)
    ParameterNode(p.offset + idxx(is .- (_start.(p.size) .- 1), _length.(p.size)))
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
function parameter(c::C, start::AbstractArray;) where {T,C<:ExaCore{T}}

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

  min (...) + ∑_{p ∈ P} f(x,θ,p)

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
       g♭ ≤ (...) + ∑_{p ∈ P} h(x,θ,p) ≤ g♯

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

# Helper to infer dimensions from iterator
_infer_subexpr_dims(itr::AbstractRange) = (itr,)
_infer_subexpr_dims(itr::AbstractArray) = (length(itr),)
_infer_subexpr_dims(itr::Base.Iterators.ProductIterator) = itr.iterators
_infer_subexpr_dims(itr) = (length(collect(itr)),)  # fallback

"""
    subexpr(core, generator; reduced=false, parameter_only=false)

Creates a subexpression that can be reused in objectives and constraints.

Three forms are available:

- **Lifted** (default, `reduced=false`): Creates auxiliary variables with defining equality
  constraints. This generates derivative code once and uses simple variable references thereafter.
  Adds variables and constraints to the problem.

- **Reduced** (`reduced=true`): Stores the expression for direct substitution when indexed.
  No auxiliary variables or constraints are created. The expression is inlined wherever used.

- **Parameter-only** (`parameter_only=true`): For expressions that depend only on parameters (θ),
  not variables (x). Values are computed once when parameters are set, not at every function
  evaluation. Automatically recomputed when `set_parameter!` is called.

Both lifted and reduced forms support SIMD-vectorized evaluation and can be nested.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore();

julia> x = variable(c, 10);

julia> s = subexpr(c, x[i]^2 for i in 1:10)
Subexpression (lifted)

  s ∈ R^{10}

julia> objective(c, s[i] + s[i+1] for i in 1:9);
```

## Reduced form (experimental)

!!! warning
    The reduced form (`reduced=true`) is experimental and may have issues with complex
    nested expressions. Use the default lifted form for production code.

```julia
c = ExaCore()
x = variable(c, 10)

# Reduced form - no extra variables/constraints
s = subexpr(c, x[i]^2 for i in 1:10; reduced=true)

# s[i] substitutes x[i]^2 directly into the expression
objective(c, s[i] + s[i+1] for i in 1:9)
```

## Parameter-only form

For expressions involving only parameters, use `parameter_only=true` to evaluate them
once when parameters change, rather than at every optimization iteration:

```julia
c = ExaCore()
θ = parameter(c, ones(10))
x = variable(c, 10)

# Parameter-only subexpression - computed once per parameter update
weights = subexpr(c, θ[i]^2 + θ[i+1] for i in 1:9; parameter_only=true)

# Use in objective - weights[i] returns cached value, not re-computed
objective(c, weights[i] * x[i]^2 for i in 1:9)
```

## Multi-dimensional example
```julia
c = ExaCore()
x = variable(c, 0:T, 0:N)

# Automatically infers 2D structure from Cartesian product
dx = subexpr(c, x[t, i] - x[t-1, i] for t in 1:T, i in 1:N)

# Now dx[t, i] can be used in constraints
constraint(c, dx[t, i] - something for t in 1:T, i in 1:N)
```
"""
function subexpr(c::C, gen::Base.Generator; reduced::Bool = false, parameter_only::Bool = false) where {T, C <: ExaCore{T}}
    # Infer dimensions before adapting (which may collect the iterator)
    ns = _infer_subexpr_dims(gen.iter)

    gen = _adapt_gen(gen)
    n = length(gen.iter)

    if parameter_only
        # Parameter-only form: evaluate once, cache values in θ, re-evaluate on parameter update
        # Store values at the end of θ (after user parameters)
        o = c.npar  # Offset into θ
        c.npar += n
        c.nparam_subexpr += n

        # Store the evaluation function for re-computation on parameter updates
        # This evaluation happens on CPU to avoid GPU scalar indexing issues,
        # but the results are stored in θ which may be on GPU.
        # This is acceptable since parameter updates are infrequent (not in optimization hot path).
        iter_collected = collect(gen.iter)
        eval_fn = function(θ_vec)
            # Convert to CPU for expression evaluation (handles GPU arrays)
            θ_cpu = θ_vec isa Array ? θ_vec : Array(θ_vec)
            dummy_x = T[]
            return T[gen.f(p)(Identity(), dummy_x, θ_cpu) for p in iter_collected]
        end
        push!(c.param_subexpr_fns, (offset = o, length = n, fn = eval_fn))

        # Evaluate immediately with current parameter values and append to θ
        # eval_fn returns CPU array, append! handles GPU conversion
        values = eval_fn(c.θ)
        c.θ = append!(c.backend, c.θ, values, n)

        return ParameterSubexpr(ns, n, o)
    end

    if reduced
        # Reduced form: store the function and iterator for direct substitution
        return ReducedSubexpr(ns, n, gen.f, collect(gen.iter))
    end

    # Lifted form: create auxiliary variables and defining constraints
    # Compute start values from expression evaluated at variable start values
    # Uses CPU evaluation to handle GPU arrays (same approach as parameter_only)
    iter_collected = collect(gen.iter)
    x0_cpu = c.x0 isa Array ? c.x0 : Array(c.x0)
    θ_cpu = c.θ isa Array ? c.θ : Array(c.θ)
    start_values = T[gen.f(p)(Identity(), x0_cpu, θ_cpu) for p in iter_collected]

    # Create auxiliary variables for subexpression values with computed start values
    v = variable(c, ns...; start = start_values)

    # Create defining constraints: v[k] - expr(itr[k]) = 0 for k = 1:n
    # We pair each element with its linear index for proper variable indexing
    paired_iter = collect(enumerate(iter_collected))
    orig_f = gen.f
    def_f = function (kp)
        k = kp[1]
        p = kp[2]
        # Use direct Var construction with linear index for robustness
        return Var(v.offset + k) - orig_f(p)
    end
    def_gen = Base.Generator(def_f, paired_iter)

    con = constraint(c, def_gen; lcon = zero(T), ucon = zero(T))

    return Subexpr(ns, n, v.offset, con)
end


function jac_structure!(m::ExaModel, rows::AbstractVector, cols::AbstractVector)
    _jac_structure!(m.cons, rows, cols)
    return rows, cols
end

_jac_structure!(cons::ConstraintNull, rows, cols) = nothing
function _jac_structure!(cons, rows, cols)
    _jac_structure!(cons.inner, rows, cols)
    sjacobian!(rows, cols, cons, nothing, nothing, NaN)
end

function hess_structure!(m::ExaModel, rows::AbstractVector, cols::AbstractVector)
    _obj_hess_structure!(m.objs, rows, cols)
    _con_hess_structure!(m.cons, rows, cols)
    return rows, cols
end

_obj_hess_structure!(objs::ObjectiveNull, rows, cols) = nothing
function _obj_hess_structure!(objs, rows, cols)
    _obj_hess_structure!(objs.inner, rows, cols)
    shessian!(rows, cols, objs, nothing, nothing, NaN, NaN)
end

_con_hess_structure!(cons::ConstraintNull, rows, cols) = nothing
function _con_hess_structure!(cons, rows, cols)
    _con_hess_structure!(cons.inner, rows, cols)
    shessian!(rows, cols, cons, nothing, nothing, NaN, NaN)
end

function obj(m::ExaModel, x::AbstractVector)
    return _obj(m.objs, x, m.θ)
end

_obj(objs, x, θ) =
    _obj(objs.inner, x, θ) +
    (isempty(objs.itr) ? zero(eltype(x)) : sum(objs.f(k, x, θ) for k in objs.itr))
_obj(objs::ObjectiveNull, x, θ) = zero(eltype(x))

function cons_nln!(m::ExaModel, x::AbstractVector, g::AbstractVector)
    fill!(g, zero(eltype(g)))
    _cons_nln!(m.cons, x, m.θ, g)
    return g
end

function _cons_nln!(cons, x, θ, g)
    _cons_nln!(cons.inner, x, θ, g)
    @simd for i in eachindex(cons.itr)
        g[offset0(cons, i)] += cons.f(cons.itr[i], x, θ)
    end
end
_cons_nln!(cons::ConstraintNull, x, θ, g) = nothing



function grad!(m::ExaModel, x::AbstractVector, f::AbstractVector)
    fill!(f, zero(eltype(f)))
    _grad!(m.objs, x, m.θ, f)
    return f
end

function _grad!(objs, x, θ, f)
    _grad!(objs.inner, x, θ, f)
    gradient!(f, objs, x, θ, one(eltype(f)))
end
_grad!(objs::ObjectiveNull, x, θ, f) = nothing

function jac_coord!(m::ExaModel, x::AbstractVector, jac::AbstractVector)
    fill!(jac, zero(eltype(jac)))
    _jac_coord!(m.cons, x, m.θ, jac)
    return jac
end

_jac_coord!(cons::ConstraintNull, x, θ, jac) = nothing
function _jac_coord!(cons, x, θ, jac)
    _jac_coord!(cons.inner, x, θ, jac)
    sjacobian!(jac, nothing, cons, x, θ, one(eltype(jac)))
end

function jprod_nln!(m::ExaModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    fill!(Jv, zero(eltype(Jv)))
    _jprod_nln!(m.cons, x, m.θ, v, Jv)
    return Jv
end

_jprod_nln!(cons::ConstraintNull, x, θ, v, Jv) = nothing
function _jprod_nln!(cons, x, θ, v, Jv)
    _jprod_nln!(cons.inner, x, θ, v, Jv)
    sjacobian!((Jv, v), nothing, cons, x, θ, one(eltype(Jv)))
end

function jtprod_nln!(m::ExaModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    fill!(Jtv, zero(eltype(Jtv)))
    _jtprod_nln!(m.cons, x, m.θ, v, Jtv)
    return Jtv
end

_jtprod_nln!(cons::ConstraintNull, x, θ, v, Jtv) = nothing
function _jtprod_nln!(cons, x, θ, v, Jtv)
    _jtprod_nln!(cons.inner, x, θ, v, Jtv)
    sjacobian!(nothing, (Jtv, v), cons, x, θ, one(eltype(Jtv)))
end

function hess_coord!(
    m::ExaModel,
    x::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.objs, x, m.θ, hess, obj_weight)
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
    _obj_hess_coord!(m.objs, x, m.θ, hess, obj_weight)
    _con_hess_coord!(m.cons, x, m.θ, y, hess, obj_weight)
    return hess
end

_obj_hess_coord!(objs::ObjectiveNull, x, θ, hess, obj_weight) = nothing
function _obj_hess_coord!(objs, x, θ, hess, obj_weight)
    _obj_hess_coord!(objs.inner, x, θ, hess, obj_weight)
    shessian!(hess, nothing, objs, x, θ, obj_weight, zero(eltype(hess)))
end

_con_hess_coord!(cons::ConstraintNull, x, θ, y, hess, obj_weight) = nothing
function _con_hess_coord!(cons, x, θ, y, hess, obj_weight)
    _con_hess_coord!(cons.inner, x, θ, y, hess, obj_weight)
    shessian!(hess, nothing, cons, x, θ, y, zero(eltype(hess)))
end

function hprod!(
    m::ExaModel,
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(Hv, zero(eltype(Hv)))
    _obj_hprod!(m.objs, x, m.θ, v, Hv, obj_weight)
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
    _obj_hprod!(m.objs, x, m.θ, v, Hv, obj_weight)
    _con_hprod!(m.cons, x, m.θ, y, v, Hv, obj_weight)
    return Hv
end

_obj_hprod!(objs::ObjectiveNull, x, θ, v, Hv, obj_weight) = nothing
function _obj_hprod!(objs, x, θ, v, Hv, obj_weight)
    _obj_hprod!(objs.inner, x, θ, v, Hv, obj_weight)
    shessian!((Hv, v), nothing, objs, x, θ, obj_weight, zero(eltype(Hv)))
end

_con_hprod!(cons::ConstraintNull, x, θ, y, v, Hv, obj_weight) = nothing
function _con_hprod!(cons, x, θ, y, v, Hv, obj_weight)
    _con_hprod!(cons.inner, x, θ, y, v, Hv, obj_weight)
    shessian!((Hv, v), nothing, cons, x, θ, y, zero(eltype(Hv)))
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
    f.o0 + f.f.first(itr[i], nothing, nothing)
@inbounds @inline offset0(f::F, itr, i) where {I<:Integer,P<:Pair{I},F<:SIMDFunction{P}} =
    f.o0 + f.f.first
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
            return reshape(view(result.$thing, (o+1):(o+len)), s...)
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
    return view(result.multipliers, (o+1):(o+len))
end


_adapt_gen(gen) = Base.Generator(gen.f, collect(gen.iter))
_adapt_gen(gen::Base.Generator{P}) where {P<:Union{AbstractArray,AbstractRange}} = gen
