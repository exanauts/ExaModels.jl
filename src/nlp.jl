abstract type AbstractVariable end
abstract type AbstractParameter end
abstract type AbstractConstraint end
abstract type AbstractObjective end

struct NaNSource{T} end
Base.getindex(::NaNSource{T}, i) where {T} = T(NaN)
Base.eltype(::NaNSource{T}) where {T} = T
Base.eltype(::Type{NaNSource{T}}) where {T} = T

"""
    Variable

A handle to a block of optimization variables added to an [`ExaCore`](@ref) via
[`add_var`](@ref) / [`@add_var`](@ref). Use indexing (e.g. `x[i]`) to reference
individual entries in objective and constraint expressions. Retrieve solution
values with [`solution`](@ref).
"""
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
    Expression

A subexpression created by [`add_expr`](@ref) / [`@add_expr`](@ref).
When indexed (e.g. `s[i]`), the expression is substituted directly into the
enclosing objective or constraint — no auxiliary variables or equality constraints
are introduced.  Use `Expression` to share common sub-expressions across multiple
objectives or constraints without duplicating the expression tree.
"""
struct Expression{S, F, I}
    size::S
    length::Int
    f::F           # The generator function
    iter::I        # The collected iterator (for indexing)
end
Base.show(io::IO, s::Expression) = print(
    io,
    """
Subexpression (reduced)

  s ∈ R^{$(join(size(s.size), " × "))}
""",
)

"""
    Parameter

A handle to a block of model parameters added to an [`ExaCore`](@ref) via
[`add_par`](@ref) / [`@add_par`](@ref). Parameter values can be updated at any
time with [`set_parameter!`](@ref) without rebuilding the model. Use indexing
(e.g. `θ[i]`) to embed parameter values in expressions.
"""
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

"""
    Objective

An objective term group added to an [`ExaCore`](@ref) via [`add_obj`](@ref) /
[`@add_obj`](@ref). All `Objective` objects in a core are summed at evaluation time
to form the total objective value.
"""
struct Objective{F,I} <: AbstractObjective
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


"""
    Constraint

A block of constraints added to an [`ExaCore`](@ref) via [`add_con`](@ref) /
[`@add_con`](@ref). Each element of the iterator corresponds to one constraint row.
Row `k` of this block maps to global constraint index `offset + k`. Dual
solution values can be retrieved with [`multipliers`](@ref).
"""
struct Constraint{F,I,O,S} <: AbstractConstraint
    f::F
    itr::I
    offset::O
    size::S
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


"""
    ConstraintAugmentation

An augmentation layer added to an existing [`Constraint`](@ref) via
[`add_con!`](@ref) / [`@add_con!`](@ref). Each element of the iterator yields an
`idx => expr` pair: `expr` is accumulated into the constraint row identified
by `idx` at evaluation time. Multiple `ConstraintAugmentation` objects can be stacked on
the same base constraint to aggregate contributions from several data sources
(e.g. summing arc flows into nodal balance constraints).
"""
struct ConstraintAugmentation{F,I,D} <: AbstractConstraint
    f::F
    itr::I
    oa::Int
    dims::D  # dimensions of the original constraint (for Pair{Tuple} offset computation)
end

Base.show(io::IO, v::ConstraintAugmentation) = print(
    io,
    """
Constraint Augmentation

  s.t. (...)
       g♭ ≤ (...) + ∑_{p ∈ P} h(x,θ,p) ≤ g♯

  where |P| = $(length(v.itr))
""",
)

"""
    ConstraintSlot{C, I}

A lightweight handle returned by `getindex` on a [`Constraint`](@ref) or
[`ConstraintAugmentation`](@ref).  When added to an expression node via `+`,
it produces a `Pair(idx, expr)` suitable for constraint augmentation.

This enables the `g[idx] += expr for ...` syntactic sugar:
```julia
c, _ = add_con!(c, g[i] += sin(x[i]) for i in 1:N)
```
"""
struct ConstraintSlot{C, I}
    con::C
    idx::I
end

const _AUGMENT_CONSTRAINT_REF = Ref{Any}(nothing)

# For 1D constraints: single index, adjust for range start.
# When start == 1 (the common case), pass idx through unchanged to
# preserve the exact type expected by SIMDFunction/offset0.
Base.getindex(c::Constraint, i) = begin
    _AUGMENT_CONSTRAINT_REF[] = c
    s = _start(c.size[1])
    ConstraintSlot(c, s == 1 ? i : i - (s - 1))
end
# For multi-dim constraints: tuple indices, adjust each for its range start.
# Uses recursive tuple construction (not ntuple) for GPU compatibility.
Base.getindex(c::Constraint, idx::Vararg{Any,N}) where {N} = begin
    _AUGMENT_CONSTRAINT_REF[] = c
    ConstraintSlot(c, _adjust_tuple(idx, c.size))
end
Base.getindex(c::ConstraintAugmentation, idx...) = begin
    _AUGMENT_CONSTRAINT_REF[] = c
    ConstraintSlot(c, length(idx) == 1 ? idx[1] : idx)
end
Base.:+(slot::ConstraintSlot, expr::AbstractNode) = Pair(slot.idx, expr)
Base.:+(expr::AbstractNode, slot::ConstraintSlot) = Pair(slot.idx, expr)
Base.setindex!(::Constraint, val, idx...) = val
Base.setindex!(::ConstraintAugmentation, val, idx...) = val

# Recursive tuple adjustment — avoids ntuple/getindex(::Tuple,::Int) for GPU compat.
# `dims` is the constraint's size tuple (ints or ranges); _start extracts the start.
@inline _adjust_tuple(idx::Tuple{}, dims::Tuple{}) = ()
@inline _adjust_tuple(idx::Tuple, dims::Tuple) = begin
    s = _start(first(dims))
    (_con_adjust(first(idx), s), _adjust_tuple(Base.tail(idx), Base.tail(dims))...)
end

# Adjust a single index for the range start.
# When start == 1, pass through unchanged to preserve the original type.
# Literal integers are wrapped in Constant so they remain callable by `coord`.
@inline _con_adjust(idx, start) = start == 1 ? idx : idx - (start - 1)
@inline _con_adjust(idx::Integer, start) = Constant(idx - start + 1)

abstract type AbstractExaCore{T,VT,B,S} end

"""
    ExaCore([array_eltype::Type; backend = nothing, minimize = true, name = :Generic])

Creates an intermediate data object `ExaCore`, which later can be used for creating an `ExaModel`

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true))
An ExaCore

  Float type: ...................... Float64
  Array type: ...................... Vector{Float64}
  Backend: ......................... Nothing

  number of objective patterns: .... 0
  number of constraint patterns: ... 0

julia> c = ExaCore(Float32; concrete = Val(true))
An ExaCore

  Float type: ...................... Float32
  Array type: ...................... Vector{Float32}
  Backend: ......................... Nothing

  number of objective patterns: .... 0
  number of constraint patterns: ... 0

julia> using CUDA

julia> c = ExaCore(Float32; backend = CUDABackend(), concrete = Val(true))
An ExaCore

  Float type: ...................... Float32
  Array type: ...................... CUDA.CuArray{Float32, 1, CUDA.DeviceMemory}
  Backend: ......................... CUDA.CUDAKernels.CUDABackend

  number of objective patterns: .... 0
  number of constraint patterns: ... 0
```
"""
struct ExaCore{T,VT<:AbstractVector{T}, B, S, V, P, O, C, R} <: AbstractExaCore{T,VT,B,S}
    name::Symbol
    backend::B
    var::V
    par::P
    obj::O
    cons::C
    nvar::Int
    npar::Int
    ncon::Int
    nconaug::Int
    nobj::Int
    nnzc::Int
    nnzg::Int
    nnzj::Int
    nnzh::Int
    x0::VT
    θ::VT
    lvar::VT
    uvar::VT
    y0::VT
    lcon::VT
    ucon::VT
    minimize::Bool
    tags::S  # For storing variable/constraint tags (e.g., scenario tags for two-stage models)
    refs::R
end

@inline function _exa_core(
    ;
    name = :Generic,
    backend = nothing,
    var = (),
    par = (),
    obj = (),
    cons = (),
    nvar = 0,
    npar = 0,
    ncon = 0,
    nconaug = 0,
    nobj = 0,
    nnzc = 0,
    nnzg = 0,
    nnzj = 0,
    nnzh = 0,
    x0 = convert_array(zeros(default_T(backend), 0), backend),
    θ = similar(x0, 0),
    lvar = similar(x0),
    uvar = similar(x0),
    y0 = similar(x0),
    lcon = similar(x0),
    ucon = similar(x0),
    minimize = true,
    tags = nothing,
    refs = (;)
    )

    return ExaCore(
        name,
        backend,
        var,
        par,
        obj,
        cons,
        nvar,
        npar,
        ncon,
        nconaug,
        nobj,
        nnzc,
        nnzg,
        nnzj,
        nnzh,
        x0,
        θ,
        lvar,
        uvar,
        y0,
        lcon,
        ucon,
        minimize,
        tags,
        refs
    )
end

@inline ExaCore(::Type{T}; backend = nothing, concrete = Val(false), kwargs...) where {T<:AbstractFloat} =
    _make_exacore(concrete, T, backend; kwargs...)
@inline ExaCore(; backend = nothing, concrete = Val(false), kwargs...) = ExaCore(default_T(backend); backend, concrete, kwargs...)
@inline _make_exacore(::Val{true}, ::Type{T}, backend; kwargs...) where {T} =
    _exa_core(; x0 = convert_array(zeros(T, 0), backend), backend, kwargs...)
# Val{false} is overridden in deprecated.jl once LegacyExaCore is defined;
# this fallback handles any other Val value by returning a concrete ExaCore.
@inline _make_exacore(::Val, ::Type{T}, backend; kwargs...) where {T} =
    _exa_core(; x0 = convert_array(zeros(T, 0), backend), backend, kwargs...)
@inline ExaCore(c::C; kwargs...) where C <: ExaCore = _exa_core(
    ;
    zip(fieldnames(C), ntuple(i -> getfield(c, i), Val(fieldcount(C))))...,
    kwargs...,
)
@inline default_T(backend) = Float64
@inline append_var_tags(::Nothing, backend, len) = nothing
@inline append_con_tags(::Nothing, backend, len) = nothing


Base.show(io::IO, c::ExaCore{T,VT,B}) where {T,VT,B} = print(
    io,
    """
An ExaCore

  Float type: ...................... $T
  Array type: ...................... $VT
  Backend: ......................... $B

  number of objective patterns: .... $(length(c.obj))
  number of constraint patterns: ... $(length(c.cons))
""",
)

"""
    AbstractExaModel

An abstract type for ExaModel, which is a subtype of `NLPModels.AbstractNLPModel`.
"""
abstract type AbstractExaModel{T,VT,E} <: NLPModels.AbstractNLPModel{T,VT} end

struct ExaModel{T,VT,E,V,P,O,C,S,R} <: AbstractExaModel{T,VT,E}
    name::Symbol
    vars::V
    pars::P
    objs::O
    cons::C
    θ::VT
    meta::NLPModels.NLPModelMeta{T,VT}
    counters::NLPModels.Counters
    ext::E
    tags::S
    refs::R
end

function Base.show(io::IO, c::AbstractExaModel{T,VT}) where {T,VT}
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

julia> c = ExaCore(concrete = Val(true));                           # create an ExaCore object

julia> c, x = add_var(c, 1:10);               # create variables

julia> c, _ = add_obj(c, x[i]^2 for i in 1:10); # set objective function

julia> m = ExaModel(c)                          # create an ExaModel object
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
function ExaModel(c::C; prod = false, kwargs...) where {C<:ExaCore}
    return ExaModel(
        c.name,
        c.var,
        c.par,
        c.obj,
        c.cons,
        c.θ,
        NLPModels.NLPModelMeta(
            c.nvar,
            ncon = c.ncon,
            nnzj = c.nnzj,
            nnzh = c.nnzh,
            x0 = (c.x0),
            lvar = (c.lvar),
            uvar = (c.uvar),
            y0 = (c.y0),
            lcon = (c.lcon),
            ucon = (c.ucon),
            minimize = c.minimize,
        )
        ,
        NLPModels.Counters(),
        build_extension(c; prod),
        c.tags,
        c.refs
    )
end

build_extension(c::ExaCore; kwargs...) = nothing

@inline function Base.getindex(v::V, i) where {D, V<:Variable{D}}
    _bound_check(v.size, i)
    _indexed_var(i, v.offset - _start(v.size[1]) + 1)
end
# For symbolic (AbstractNode) indices: use Node2 directly so the offset (runtime Int)
# is stored as a plain Int64 child — giving concrete type Node2{+, I, Int64}.
# Going through _add_node_real would wrap the runtime offset in Val(d2::Int64),
# which is type-unstable and breaks juliac --trim=safe.
@inline _indexed_var(i::I, o::Int) where {I<:AbstractNode} = Var(Node2(+, i, o))
@inline _indexed_var(i, o) = Var(i + o)
@inline function Base.getindex(v::V, is...) where {D, V<:Variable{D}}
    @assert(length(is) == length(v.size), "Variable index dimension error")
    _bound_check(v.size, is)
    Var(v.offset + idxx(is .- (_start.(v.size) .- 1), _length.(v.size)))
end

# Expression indexing - evaluates the expression directly
# For concrete indices, look up the iterator element and apply f
# For symbolic indices (during expression building), create symbolic iterator elements
@inline function Base.getindex(s::Expression, i::I) where {I <: Integer}
    _bound_check(s.size, i)
    idx = i - _start(s.size[1]) + 1
    return s.f(s.iter[idx])
end
@inline function Base.getindex(s::Expression, i)
    # Symbolic index case - the symbolic index IS the iterator element
    # No adjustment needed; the index is used directly in expression building
    return s.f(i)
end
@inline function Base.getindex(s::Expression, is::Vararg{I, N}) where {I <: Integer, N}
    @assert(length(is) == length(s.size), "Expression index dimension error")
    _bound_check(s.size, is)
    idx = idxx(is .- (_start.(s.size) .- 1), _length.(s.size))
    return s.f(s.iter[idx])
end
@inline function Base.getindex(s::Expression, is...)
    # Symbolic indices case - the symbolic indices ARE the iterator elements
    # No adjustment needed; the indices are used directly in expression building
    @assert(length(is) == length(s.size), "Expression index dimension error")
    return s.f(is)
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
    if lb != 0
        b = _adapt_gen(b)
        la = length(a)
        resize!(a, la + lb)
        map!(b.f, view(a, (la+1):(la+lb)), convert_array(b.iter, backend))
    end
    return a
end

function append!(backend, a, b::Base.Generator{UnitRange{I}}, lb) where {I}
    if lb != 0
        la = length(a)
        resize!(a, la + lb)
        map!(b.f, view(a, (la+1):(la+lb)), b.iter)
    end
    return a
end

function append!(backend, a, b::AbstractArray, lb)
    if lb != 0
        la = length(a)
        resize!(a, la + lb)
        map!(identity, view(a, (la+1):(la+lb)), convert_array(b, backend))
    end
    return a
end

function append!(backend, a, b::Number, lb)
    if lb != 0
        la = length(a)
        resize!(a, la + lb)
        fill!(view(a, (la+1):(la+lb)), eltype(a)(b))
    end
    return a
end

@inline total(ns) = _total(ns...)
@inline _total() = 1
@inline _total(n, ns...) = _length(n) * _total(ns...)
@inline _length(n::Int) = n
@inline _length(n::UnitRange) = length(n)
@inline size(ns) = _size(ns...)
@inline _size() = ()
@inline _size(n, ns...) = (_length(n), _size(ns...)...)
@inline _start(n::Int) = 1
@inline _start(n::UnitRange) = n.start

"""
    add_var(core, dims...; start = 0, lvar = -Inf, uvar = Inf, name = nothing, kwargs...)

Adds variables with dimensions specified by `dims` to `core`. `dims` can be either `Integer` or `UnitRange`. Returns `(core, Variable)`.

## Keyword Arguments
- `start`: The initial guess of the solution. Can either be `Number`, `AbstractArray`, or `Generator`.
- `lvar` : The variable lower bound. Can either be `Number`, `AbstractArray`, or `Generator`.
- `uvar` : The variable upper bound. Can either be `Number`, `AbstractArray`, or `Generator`.
- `name` : When given as `Val(:name)`, registers the variable in `core` for later retrieval as `core.name` or `model.name`. See [`@add_var`](@ref) for the idiomatic named interface.
- `kwargs...`: Additional keyword arguments for variable tags (e.g., `scenario` for two-stage models)

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 10; start = (sin(i) for i=1:10));

julia> x
Variable

  x ∈ R^{10}

julia> c, y = add_var(c, 2:10, 3:5; lvar = zeros(9,3), uvar = ones(9,3));

julia> y
Variable

  x ∈ R^{9 × 3}

```
"""
@inline function add_var(
    c::C,
    ns...;
    name = nothing,
    start = zero(T),
    lvar = T(-Inf),
    uvar = T(Inf),
    kwargs...
) where {T,C<:ExaCore{T}}


    o = c.nvar
    len = total(ns)
    nvar = c.nvar + len
    x0 = append!(c.backend, c.x0, start, total(ns))
    lvar = append!(c.backend, c.lvar, lvar, total(ns))
    uvar = append!(c.backend, c.uvar, uvar, total(ns))

    append_var_tags(c.tags, c.backend, total(ns); kwargs...)
    v = Variable(ns, len, o)

    (ExaCore(c; var = (v, c.var...), nvar=nvar, x0=x0, lvar=lvar, uvar=uvar, refs = add_refs(c.refs, name, v)), v)
end

@inline add_refs(refs, ::Nothing, var) = refs
@inline add_refs(refs, ::Val{N}, var) where {N} = (; refs..., N => var)


"""
    add_par(core, start::AbstractArray; name = nothing)

Adds parameters with initial values specified by `start`, and returns `(core, Parameter)`.

## Keyword Arguments
- `name`: When given as `Val(:name)`, registers the parameter in `core` for later retrieval as `core.name` or `model.name`. See [`@add_par`](@ref) for the idiomatic named interface.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true));

julia> c, θ = add_par(c, ones(10));

julia> θ
Parameter

  θ ∈ R^{10}
```
"""
@inline function add_par(c::C, start::AbstractArray; name = nothing) where {T,C<:ExaCore{T}}

    ns = Base.size(start)
    o = c.npar
    len = total(ns)
    npar = c.npar + len
    θ = append!(c.backend, c.θ, start, len)
    p = Parameter(ns, len, o)
    (ExaCore(c; par = (p, c.par...), θ=θ, npar=npar, refs = add_refs(c.refs, name, p)), p)
end

"""
    set_parameter!(core, param, values)

Updates the values of parameters in the core.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true));

julia> c, p = add_par(c, ones(5));

julia> set_parameter!(c, p, ones(5))
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

    return nothing
end

@inline function add_var(c::C; kwargs...) where {T,C<:ExaCore{T}}
    c, v = add_var(c, 1; kwargs...)
    return (c, v[1])
end

"""
    add_var(core, gen::Base.Generator; kwargs...)

Create variables constrained to equal the expressions produced by `gen`.
Equivalent to creating `length(gen.iter)` variables with equality constraints
tying each to the corresponding generator expression.
Returns `(core, Variable)`.
"""
@inline function add_var(
    c::C,
    gen::Base.Generator;
    name = nothing,
    start = zero(T),
    lvar = T(-Inf),
    uvar = T(Inf),
    kwargs...
) where {T,C<:ExaCore{T}}
    gen = _adapt_gen(gen)
    n = length(gen.iter)
    c, x = add_var(c, n; name = name, start = start, lvar = lvar, uvar = uvar, kwargs...)
    # Pair local 1-based index with original parameter so x[j] uses 1:n
    # while gen.f(orig) sees the original iterator element.
    pars = collect(enumerate(gen.iter))
    c, _ = add_con(c, x[j] - gen.f(orig) for (j, orig) in pars)
    return (c, x)
end


"""
    add_obj(core::ExaCore, generator; name = nothing)

Adds objective terms specified by a `generator` to `core`, and returns `(core, Objective)`. The terms are summed.

## Keyword Arguments
- `name`: When given as `Val(:name)`, registers the objective in `core` for later retrieval as `core.name` or `model.name`. See [`@add_obj`](@ref) for the idiomatic named interface.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 10);

julia> c, obj = add_obj(c, x[i]^2 for i=1:10);

julia> obj
Objective

  min (...) + ∑_{p ∈ P} f(x,θ,p)

  where |P| = 10
```
"""
@inline function add_obj(c::C, gen; name = nothing) where {T, C<:ExaCore{T}}
    gen = _adapt_gen(gen)
    f = SIMDFunction(T, gen, c.nobj, c.nnzg, c.nnzh)
    pars = gen.iter

    _add_obj(c, f, pars, name)
end

"""
    add_obj(core::ExaCore, expr [, pars]; name = nothing)

Low-level form of [`add_obj`](@ref) that accepts a pre-built `AbstractNode`
expression `expr` evaluated over `pars`, and returns `(core, Objective)`.

When `name` is given as `Val(:name)`, the objective is also accessible as
`core.name` or `model.name`.

Prefer the generator form (`add_obj(core, gen)`) for typical use; this form
is intended for code that builds expression trees programmatically.
"""
@inline function add_obj(c::C, expr::N, pars = 1:1; name = nothing) where {T,C<:ExaCore{T},N<:AbstractNode}
    f = _simdfunction(T, expr, c.nobj, c.nnzg, c.nnzh)

      _add_obj(c, f, pars, name)
end

@inline function _add_obj(c, f, pars, name = nothing)
    nitr = length(pars)
    nobj = c.nobj + nitr
    nnzg = c.nnzg + nitr * f.o1step
    nnzh = c.nnzh + nitr * f.o2step

    obj = Objective(f, convert_array(pars, c.backend))
    (ExaCore(c; nobj=nobj, nnzg=nnzg, nnzh=nnzh, obj=(obj, c.obj...), refs = add_refs(c.refs, name, obj)), obj)
end


"""
    add_con(core, generator; start = 0, lcon = 0, ucon = 0, kwargs...)

Adds constraints specified by a `generator` to `core`, and returns `(core, Constraint)`.

## Keyword Arguments
- `start`: The initial guess of the dual solution. Can either be `Number`, `AbstractArray`, or `Generator`.
- `lcon` : The constraint lower bound. Can either be `Number`, `AbstractArray`, or `Generator`.
- `ucon` : The constraint upper bound. Can either be `Number`, `AbstractArray`, or `Generator`.
- `name` : When given as `Val(:name)`, registers the constraint in `core` for later retrieval as `core.name` or `model.name`. See [`@add_con`](@ref) for the idiomatic named interface.
- `kwargs...`: Additional keyword arguments for constraint tags (e.g., `scenario` for two-stage models)

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 10);

julia> c, con = add_con(c, x[i] + x[i+1] for i=1:9; lcon = -1, ucon = (1+i for i=1:9));

julia> con
Constraint

  s.t. (...)
       g♭ ≤ [g(x,θ,p)]_{p ∈ P} ≤ g♯

  where |P| = 9
```
"""
@inline function add_con(
    c::C,
    gen::Base.Generator;
    name = nothing,
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
    kwargs...
) where {T,C<:ExaCore{T}}

    dims = _infer_subexpr_dims(gen.iter)
    gen = _adapt_gen(gen)
    f = SIMDFunction(T, gen, c.ncon, c.nnzj, c.nnzh)
    pars = gen.iter

    _add_con(c, f, pars, dims, start, lcon, ucon, name; kwargs...)
end

"""
    add_con(core, gen, gens...; kwargs...)

Create a constraint from the first generator, then augment it with each
subsequent generator via [`add_con!`](@ref). Returns `(core, Constraint)`.

## Keyword Arguments
- `start`: Initial guess for the dual solution. Can be a `Number`, `AbstractArray`, or `Generator`.
- `lcon` : Constraint lower bound. Can be a `Number`, `AbstractArray`, or `Generator`.
- `ucon` : Constraint upper bound. Can be a `Number`, `AbstractArray`, or `Generator`.
- `name` : When given as `Val(:name)`, registers the constraint for later retrieval as `core.name` or `model.name`.
- `kwargs...`: Additional keyword arguments forwarded to [`add_con`](@ref) (e.g. scenario tags).

## Example
```julia
c, g = add_con(c,
    (x[i] for i in 1:N),
    (i => sin(x[i]) for i in 1:N);
    lcon = 0.0, ucon = 0.0,
)
```
"""
@inline function add_con(
    c::C,
    gen::Base.Generator,
    gens::Base.Generator...;
    kwargs...
) where {T,C<:ExaCore{T}}
    c, con = add_con(c, gen; kwargs...)
    for g in gens
        c, _ = add_con!(c, con, g)
    end
    return (c, con)
end

"""
    add_con(core, expr [, pars]; start = 0, lcon = 0, ucon = 0, name = nothing)

Low-level form of [`add_con`](@ref) that accepts a pre-built `AbstractNode`
expression `expr` evaluated over `pars`, and returns `(core, Constraint)`.

When `name` is given as `Val(:name)`, the constraint is also accessible as
`core.name` or `model.name`.

Prefer the generator form (`add_con(core, gen)`) for typical use; this form
is intended for code that builds expression trees programmatically.
"""
@inline function add_con(
    c::C,
    expr::N,
    pars = 1:1;
    name = nothing,
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
    kwargs...
) where {T,C<:ExaCore{T},N<:AbstractNode}

    f = _simdfunction(T,expr, c.ncon, c.nnzj, c.nnzh)
    dims = _infer_subexpr_dims(pars)

    _add_con(c, f, pars, dims, start, lcon, ucon, name; kwargs...)
end

"""
    add_con(core, dims...; start = 0, lcon = 0, ucon = 0, name = nothing)

Adds empty constraints with dimensions specified by `dims` to `core` and returns
`(core, Constraint)`.  No expression is attached initially — use
[`add_con!`](@ref) / [`@add_con!`](@ref) to accumulate terms into the rows
afterwards.

`dims` can be a single integer (`add_con(c, 9)`), multiple integers
(`add_con(c, 3, 4)` for a 3×4 grid), or `AbstractUnitRange` values
(`add_con(c, 1:3, 2:5)`) — matching the convention used by [`add_var`](@ref).

When `name` is given as `Val(:name)`, the constraint is also accessible as
`core.name` or `model.name`.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 10);

julia> c, g = add_con(c, 9; lcon = -1.0, ucon = 1.0);

julia> c, _ = add_con!(c, g, i => x[i] + x[i+1] for i = 1:9);

julia> g
Constraint

  s.t. (...)
       g♭ ≤ [g(x,θ,p)]_{p ∈ P} ≤ g♯

  where |P| = 9
```
"""
@inline function add_con(
    c::C,
    ns...;
    name = nothing,
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
    kwargs...
) where {T,C<:ExaCore{T}}

    f = _simdfunction(T, Null(nothing), c.ncon, c.nnzj, c.nnzh)
    pars = _empty_con_itr(ns)

    _add_con(c, f, pars, ns, start, lcon, ucon, name; kwargs...)
end

# Build an iterator for empty constraints: 1:n for 1D, collected ProductIterator for multi-dim.
_empty_con_itr(ns::Tuple{Any}) = 1:_length(ns[1])
_empty_con_itr(ns::Tuple) = collect(Iterators.product(map(n -> 1:_length(n), ns)...))


function _add_con(c::C, f, pars, dims, start, lcon, ucon, name = nothing; kwargs...) where {C<:ExaCore}
    nitr = length(pars)
    o = c.ncon
    ncon = c.ncon + nitr
    nnzj = c.nnzj + nitr * f.o1step
    nnzh = c.nnzh + nitr * f.o2step

    y0 = append!(c.backend, c.y0, start, nitr)
    lcon = append!(c.backend, c.lcon, lcon, nitr)
    ucon = append!(c.backend, c.ucon, ucon, nitr)
    append_con_tags(c.tags, c.backend, nitr; kwargs...)

    con = Constraint(f, convert_array(pars, c.backend), o, dims)

    (ExaCore(c; ncon=ncon, nnzj=nnzj, nnzh=nnzh, y0=y0, lcon=lcon, ucon=ucon, cons=(con, c.cons...), refs = add_refs(c.refs, name, con)), con)
end




"""
    add_con!(core, c1, generator)

Augments the existing constraint `c1` by adding extra expression terms to a subset of its
rows, and returns `(core, ConstraintAugmentation)`.

This is the primary mechanism for building constraints that aggregate contributions from
multiple data sources — for example, nodal power-balance constraints that sum flows over
all arcs incident to each bus.  Each call to `add_con!` appends one new "augmentation
layer"; multiple layers for the same base constraint are summed at evaluation time.

The bounds (`lcon`/`ucon`) remain those set on the original `c1` and **cannot** be changed
via `add_con!`.

## Arguments
- `core`: The [`ExaCore`](@ref) to modify.
- `c1`: The base [`Constraint`](@ref) (or a previous [`ConstraintAugmentation`](@ref)) whose rows are being augmented.
- `generator`: A `Base.Generator` yielding `idx => expr` pairs, where
  - `idx` is an index (or tuple of indices) into `c1` identifying which constraint row receives the term, and
  - `expr` is the scalar expression to add to that row.

## Notes
- The index `idx` must be a valid index of `c1`'s iterator (e.g. an integer for a 1-D
  constraint, or a tuple `(i, j)` for a multi-dimensional one).
- One generator element maps to one non-zero Jacobian row; elements with the same `idx`
  are accumulated.
- The iterator of the generator becomes the SIMD work set, so performance is best when it
  is a contiguous collection (array, range, or product iterator).

## Example

Single-index augmentation — add `sin(x[i+1])` to constraint rows 4, 5, 6:

```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 10);

julia> c, c1 = add_con(c, x[i] + x[i+1] for i=1:9; lcon = -1, ucon = (1+i for i=1:9));

julia> c, c2 = add_con!(c, c1, i => sin(x[i+1]) for i=4:6);

julia> c2
Constraint Augmentation

  s.t. (...)
       g♭ ≤ (...) + ∑_{p ∈ P} h(x,θ,p) ≤ g♯

  where |P| = 3
```

Multi-source augmentation (typical power-flow use case) — accumulate arc flows into bus
balance constraints:

```julia
c, bus = add_con(c, pd[b.i] + gs[b.i]*vm[b.i]^2 for b in data.bus)   # one row per bus
add_con!(c, bus, arc.bus => p[arc.i] for arc in data.arc)              # add arc flows
add_con!(c, bus, gen.bus => -pg[gen.i] for gen in data.gen)            # subtract generation
```
"""
@inline function add_con!(c::C, c1, gen::Base.Generator) where {T, C<:ExaCore{T}}

    gen = _adapt_gen(gen)
    f = SIMDFunction(T, gen, offset0(c1, 0), c.nnzj, c.nnzh)
    pars = gen.iter

    _add_con!(c, f, pars, _constraint_dims(c1))
end

"""
    add_con!(core::ExaCore, gen::Base.Generator)

Two-argument form of [`add_con!`](@ref) supporting the `constraint[idx] += expr` sugar.
The generator must use the pattern `g[idx] += expr for ... in itr`, where `g` is an
existing [`Constraint`](@ref) or [`ConstraintAugmentation`](@ref).

Equivalent to `add_con!(core, g, idx => expr for ... in itr)`.

## Example
```julia
c = ExaCore(concrete = Val(true))
c, x = add_var(c, 10)
c, g = add_con(c, 9; lcon = -1.0, ucon = 1.0)
c, _ = add_con!(c, g[i] += x[i] + x[i+1] for i = 1:9)
```
"""
@inline function add_con!(c::ExaCore{T}, gen::Base.Generator) where T
    gen = _adapt_gen(gen)

    # Probe the generator to extract the target constraint.
    # getindex on Constraint/ConstraintAugmentation records the constraint
    # in _AUGMENT_CONSTRAINT_REF as a side effect.
    _AUGMENT_CONSTRAINT_REF[] = nothing
    gen.f(ParSource())
    con = _AUGMENT_CONSTRAINT_REF[]
    _AUGMENT_CONSTRAINT_REF[] = nothing

    con === nothing && error(
        "add_con! two-argument form requires `constraint[idx] += expr` syntax " *
        "in the generator body"
    )

    # The generator already yields Pair(idx, expr) thanks to the
    # ConstraintSlot + AbstractNode → Pair overload, so delegate directly.
    return add_con!(c, con, gen)
end

# Extract the dimensions of the original constraint's iterator
_constraint_dims(c::Constraint) = Tuple(_length(n) for n in c.size)
_constraint_dims(c::ConstraintAugmentation) = c.dims

function _add_con!(c, f, pars, dims)
    oa = c.nconaug
    nitr = length(pars)
    nconaug = c.nconaug + nitr
    nnzj = c.nnzj + nitr * f.o1step
    nnzh = c.nnzh + nitr * f.o2step
    con = ConstraintAugmentation(f, convert_array(pars, c.backend), oa, dims)
    (ExaCore(c; nconaug=nconaug, nnzj=nnzj, nnzh=nnzh, cons=(con, c.cons...)), con)
end

# Helper to infer dimensions from iterator
_infer_subexpr_dims(itr::AbstractRange) = (itr,)
_infer_subexpr_dims(itr::AbstractArray) = Base.size(itr)
_infer_subexpr_dims(itr::Base.Iterators.ProductIterator) = itr.iterators
_infer_subexpr_dims(itr) = (length(collect(itr)),)  # fallback

"""
    add_expr(core, generator; name = nothing)

Creates a subexpression from a `generator` and returns `(core, Expression)`.
The expression is stored for direct substitution (inlining) when indexed — no auxiliary
variables or constraints are added to the problem.

## Keyword Arguments
- `name`: When given as `Val(:name)`, registers the subexpression in `core` for later retrieval as `core.name` or `model.name`. See [`@add_expr`](@ref) for the idiomatic named interface.

## Example
```jldoctest
julia> using ExaModels

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 10);

julia> c, s = add_expr(c, x[i]^2 for i in 1:10);

julia> s
Subexpression (reduced)

  s ∈ R^{10}

julia> c, _ = add_obj(c, s[i] + s[i+1] for i in 1:9);
```

## Multi-dimensional example

```julia
c = ExaCore(concrete = Val(true))
c, x = add_var(c, 1:N, 1:K)
itr = [(i, k) for i in 1:N, k in 1:K]
c, s = add_expr(c, x[i, k]^2 for (i, k) in itr)
# s[i, k] substitutes x[i,k]^2 directly
```
"""
@inline function add_expr(c::C, gen::Base.Generator; name = nothing) where {T, C <: ExaCore{T}}
    ns = _infer_subexpr_dims(gen.iter)

    gen = _adapt_gen(gen)
    n = length(gen.iter)

    ex = Expression(ns, n, gen.f, collect(gen.iter))
    return (ExaCore(c; refs = add_refs(c.refs, name, ex)), ex)
end

function jac_structure!(m::AbstractExaModel{T}, rows::AbstractVector, cols::AbstractVector) where T
    _jac_structure!(T, m.cons, rows, cols)
    return rows, cols
end

_jac_structure!(T, cons::Tuple{}, rows, cols) = nothing
@inline function _jac_structure!(T, cons::Tuple, rows, cols)
    _jac_structure!(T, Base.tail(cons), rows, cols)
    sjacobian!(rows, cols, first(cons), NaNSource{T}(), NaNSource{T}(), T(NaN))
end

function hess_structure!(m::AbstractExaModel{T}, rows::AbstractVector, cols::AbstractVector) where T
    _obj_hess_structure!(T, m.objs, rows, cols)
    _con_hess_structure!(T, m.cons, rows, cols)
    return rows, cols
end

_obj_hess_structure!(T, objs::Tuple{}, rows, cols) = nothing
@inline function _obj_hess_structure!(T, objs::Tuple, rows, cols)
    _obj_hess_structure!(T, Base.tail(objs), rows, cols)
    shessian!(rows, cols, first(objs), NaNSource{T}(), NaNSource{T}(), T(NaN), T(NaN))
end

_con_hess_structure!(T, cons::Tuple{}, rows, cols) = nothing
@inline function _con_hess_structure!(T, cons::Tuple, rows, cols)
    _con_hess_structure!(T, Base.tail(cons), rows, cols)
    shessian!(rows, cols, first(cons), NaNSource{T}(), NaNSource{T}(), T(NaN), T(NaN))
end

function obj(m::AbstractExaModel, x::AbstractVector)
    return _obj(m.objs, x, m.θ)
end

@inline function _obj((obj, objs...), x, θ)
    s = _obj(objs, x, θ)
    for i in obj.itr
        s += obj.f(i, x, θ)
    end
    return s
end

@inline _obj(obj::Tuple{}, x, θ) = zero(eltype(x))

function cons_nln!(m::AbstractExaModel, x::AbstractVector, g::AbstractVector)
    fill!(g, zero(eltype(g)))
    _cons_nln!(m.cons, x, m.θ, g)
    return g
end

@inline function _cons_nln!(cons::Tuple, x, θ, g)
    con = first(cons)
    _cons_nln!(Base.tail(cons), x, θ, g)
    @simd for i in eachindex(con.itr)
        g[offset0(con, i)] += con.f(con.itr[i], x, θ)
    end
end
_cons_nln!(cons::Tuple{}, x, θ, g) = nothing



function grad!(m::AbstractExaModel, x::AbstractVector, f::AbstractVector)
    fill!(f, zero(eltype(f)))
    _grad!(m.objs, x, m.θ, f)
    return f
end

@inline function _grad!(objs::Tuple, x, θ, f)
    _grad!(Base.tail(objs), x, θ, f)
    gradient!(f, first(objs), x, θ, one(eltype(f)))
end
_grad!(objs::Tuple{}, x, θ, f) = nothing

function jac_coord!(m::AbstractExaModel, x::AbstractVector, jac::AbstractVector)
    fill!(jac, zero(eltype(jac)))
    _jac_coord!(m.cons, x, m.θ, jac)
    return jac
end

_jac_coord!(cons::Tuple{}, x, θ, jac) = nothing
@inline function _jac_coord!(cons::Tuple, x, θ, jac)
    _jac_coord!(Base.tail(cons), x, θ, jac)
    sjacobian!(jac, nothing, first(cons), x, θ, one(eltype(jac)))
end

function jprod_nln!(m::AbstractExaModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    fill!(Jv, zero(eltype(Jv)))
    _jprod_nln!(m.cons, x, m.θ, v, Jv)
    return Jv
end

_jprod_nln!(cons::Tuple{}, x, θ, v, Jv) = nothing
@inline function _jprod_nln!(cons::Tuple, x, θ, v, Jv)
    _jprod_nln!(Base.tail(cons), x, θ, v, Jv)
    sjacobian!((Jv, v), nothing, first(cons), x, θ, one(eltype(Jv)))
end

function jtprod_nln!(m::AbstractExaModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    fill!(Jtv, zero(eltype(Jtv)))
    _jtprod_nln!(m.cons, x, m.θ, v, Jtv)
    return Jtv
end

_jtprod_nln!(cons::Tuple{}, x, θ, v, Jtv) = nothing
@inline function _jtprod_nln!(cons::Tuple, x, θ, v, Jtv)
    _jtprod_nln!(Base.tail(cons), x, θ, v, Jtv)
    sjacobian!(nothing, (Jtv, v), first(cons), x, θ, one(eltype(Jtv)))
end

function hess_coord!(
    m::AbstractExaModel,
    x::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.objs, x, m.θ, hess, obj_weight)
    return hess
end

function hess_coord!(
    m::AbstractExaModel,
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

_obj_hess_coord!(objs::Tuple{}, x, θ, hess, obj_weight) = nothing
@inline function _obj_hess_coord!(objs::Tuple, x, θ, hess, obj_weight)
    _obj_hess_coord!(Base.tail(objs), x, θ, hess, obj_weight)
    shessian!(hess, nothing, first(objs), x, θ, obj_weight, zero(eltype(hess)))
end

_con_hess_coord!(cons::Tuple{}, x, θ, y, hess, obj_weight) = nothing
@inline function _con_hess_coord!(cons::Tuple, x, θ, y, hess, obj_weight)
    _con_hess_coord!(Base.tail(cons), x, θ, y, hess, obj_weight)
    shessian!(hess, nothing, first(cons), x, θ, y, zero(eltype(hess)))
end

function hprod!(
    m::AbstractExaModel,
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
    m::AbstractExaModel,
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

_obj_hprod!(objs::Tuple{}, x, θ, v, Hv, obj_weight) = nothing
@inline function _obj_hprod!(objs::Tuple, x, θ, v, Hv, obj_weight)
    _obj_hprod!(Base.tail(objs), x, θ, v, Hv, obj_weight)
    shessian!((Hv, v), nothing, first(objs), x, θ, obj_weight, zero(eltype(Hv)))
end

_con_hprod!(cons::Tuple{}, x, θ, y, v, Hv, obj_weight) = nothing
@inline function _con_hprod!(cons::Tuple, x, θ, y, v, Hv, obj_weight)
    _con_hprod!(Base.tail(cons), x, θ, y, v, Hv, obj_weight)
    shessian!((Hv, v), nothing, first(cons), x, θ, y, zero(eltype(Hv)))
end

@inbounds @inline offset0(a, i) = offset0(a.f, i)
@inbounds @inline offset1(a, i) = offset1(a.f, i)
@inbounds @inline offset2(a, i) = offset2(a.f, i)
@inbounds @inline offset0(f, itr, i) = offset0(f, i)
@inbounds @inline offset0(f::F, i) where {F<:SIMDFunction} = f.o0 + i
@inbounds @inline offset1(f::F, i) where {F<:SIMDFunction} = f.o1 + f.o1step * (i - 1)
@inbounds @inline offset2(f::F, i) where {F<:SIMDFunction} = f.o2 + f.o2step * (i - 1)
@inbounds @inline offset0(a::C, i) where {C<:ConstraintAugmentation} = offset0(a.f, a.itr, i, a.dims)
@inbounds @inline offset0(f::F, itr, i) where {P<:Pair,F<:SIMDFunction{P}} =
    f.o0 + f.f.first(itr[i], nothing, nothing)
@inbounds @inline offset0(f::F, itr, i) where {I<:Integer,P<:Pair{I},F<:SIMDFunction{P}} =
    f.o0 + f.f.first
@inbounds @inline offset0(f::F, itr, i) where {T<:Tuple,P<:Pair{T},F<:SIMDFunction{P}} =
    f.o0 + idxx(coord(itr, i, f.f.first), Base.size(itr))
# 4-arg variant used by ConstraintAugmentation to pass original constraint dims
@inbounds @inline offset0(f, itr, i, dims) = offset0(f, i)
@inbounds @inline offset0(f::F, itr, i, dims) where {P<:Pair,F<:SIMDFunction{P}} =
    f.o0 + f.f.first(itr[i], nothing, nothing)
@inbounds @inline offset0(f::F, itr, i, dims) where {I<:Integer,P<:Pair{I},F<:SIMDFunction{P}} =
    f.o0 + f.f.first
@inbounds @inline offset0(f::F, itr, i, dims) where {T<:Tuple,P<:Pair{T},F<:SIMDFunction{P}} =
    f.o0 + idxx(coord(itr, i, f.f.first), dims)

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
@inline _idxx(::Tuple{}, ::Tuple, a) = 0  # partial indexing: coord exhausted before dims

@inline coord(itr, i, (f, fs...)) = (f(idx(itr, i), nothing, nothing), coord(itr, i, fs)...)
@inline coord(itr, i, ::Tuple{}) = ()

"""
    solution(result, x)

Returns the primal solution values for variable block `x` associated with `result`,
obtained by solving the model. The returned array has the same shape as `x`.

## Example
```jldoctest
julia> using ExaModels, NLPModelsIpopt

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 1:10; lvar = -1, uvar = 1);

julia> c, _ = add_obj(c, (x[i]-2)^2 for i in 1:10);

julia> m = ExaModel(c);

julia> result = ipopt(m; print_level=0);

julia> val = solution(result, x);

julia> isapprox(val, fill(1, 10), atol=sqrt(eps(Float64)), rtol=Inf)
true
```
"""
function solution(result::SolverCore.AbstractExecutionStats, x)
    o = x.offset
    len = total(x.size)
    s = size(x.size)
    return reshape(view(result.solution, (o+1):(o+len)), s...)
end

"""
    multipliers_L(result, x)

Returns the lower-bound dual variables for variable block `x` associated with
`result`, obtained by solving the model.

`multipliers_L[i] ≥ 0` is the dual variable for the bound constraint `x[i] ≥ lvar[i]`.
A nonzero value indicates that the lower bound is active at the solution; the
magnitude measures how much the objective would improve if that bound were relaxed.

## Example
```jldoctest
julia> using ExaModels, NLPModelsIpopt

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 1:10; lvar = -1, uvar = 1);

julia> c, _ = add_obj(c, (x[i]-2)^2 for i in 1:10);

julia> m = ExaModel(c);

julia> result = ipopt(m; print_level=0);

julia> val = multipliers_L(result, x);

julia> isapprox(val, fill(0, 10), atol=sqrt(eps(Float64)), rtol=Inf)
true
```
"""
function multipliers_L(result::SolverCore.AbstractExecutionStats, x)
    o = x.offset
    len = total(x.size)
    s = size(x.size)
    return reshape(view(result.multipliers_L, (o+1):(o+len)), s...)
end

"""
    multipliers_U(result, x)

Returns the upper-bound dual variables for variable block `x` associated with
`result`, obtained by solving the model.

`multipliers_U[i] ≥ 0` is the dual variable for the bound constraint `x[i] ≤ uvar[i]`.
A nonzero value indicates that the upper bound is active at the solution; the
magnitude measures how much the objective would improve if that bound were relaxed.

## Example
```jldoctest
julia> using ExaModels, NLPModelsIpopt

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 1:10; lvar = -1, uvar = 1);

julia> c, _ = add_obj(c, (x[i]-2)^2 for i in 1:10);

julia> m = ExaModel(c);

julia> result = ipopt(m; print_level=0);

julia> val = multipliers_U(result, x);

julia> isapprox(val, fill(2, 10), atol=sqrt(eps(Float64)), rtol=Inf)
true
```
"""
function multipliers_U(result::SolverCore.AbstractExecutionStats, x)
    o = x.offset
    len = total(x.size)
    s = size(x.size)
    return reshape(view(result.multipliers_U, (o+1):(o+len)), s...)
end

solution(result::SolverCore.AbstractExecutionStats, x::Var) = result.solution[x.i]


"""
    multipliers(result, y)

Returns the multipliers for constraints `y` associated with `result`, obtained by solving the model.

## Example
```jldoctest
julia> using ExaModels, NLPModelsIpopt

julia> c = ExaCore(concrete = Val(true));

julia> c, x = add_var(c, 1:10; lvar = -1, uvar = 1);

julia> c, _ = add_obj(c, (x[i]-2)^2 for i in 1:10);

julia> c, y = add_con(c, x[i] + x[i+1] for i=1:9; lcon = -1, ucon = (1+i for i=1:9));

julia> m = ExaModel(c);

julia> result = ipopt(m; print_level=0);

julia> val = multipliers(result, y);


julia> val[1] ≈ 0.81933930
true
```
"""
function multipliers(result::SolverCore.AbstractExecutionStats, y::Constraint)
    o = y.offset
    len = total(y.size)
    s = size(y.size)
    return reshape(view(result.multipliers, (o+1):(o+len)), s...)
end

_adapt_gen(gen) = Base.Generator(gen.f, collect(gen.iter))
_adapt_gen(gen::Base.Generator{P}) where {P<:Union{AbstractArray,AbstractRange}} = gen

function Base.getproperty(core::E, name::Symbol) where {E <: Union{ExaCore, ExaModel}}
    if hasfield(E, name)
        getfield(core,name)
    elseif hasfield(typeof(core.refs), name)
        getfield(core.refs, name)
    else
        getfield(core, name)
    end
end

function _split_macro_args(exs)
    parts = Any[]
    kwargs = Expr(:parameters)
    for ex in exs
        ex isa Expr && ex.head == :parameters ?
            Base.append!(kwargs.args, ex.args) : push!(parts, ex)
    end
    return parts, kwargs
end

"""
    @add_var(core, [name,] dims...; kwargs...)

Macro interface for [`add_var`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_var(core, x, dims...)`): binds `x` to the new `Variable` in the local scope
  and registers it in `core` for later retrieval as `core.x` or `model.x`.
- **Anonymous** (`@add_var(core, dims...)`): equivalent to `c, v = add_var(c, dims...)`.

Accepts the same keyword arguments as [`add_var`](@ref).

## Example
```julia
c = ExaCore(concrete = Val(true))
@add_var(c, x, 10; lvar = -1, uvar = 1)  # x is now in scope; c.x also works
@add_var(c, y, 1:5)                        # y is in scope; c.y also works
```
"""
macro add_var(exs...)
    isempty(exs) && error("@add_var requires at least a core argument")
    parts, kwargs = _split_macro_args(exs)
    core = parts[1]
    xs = [_auto_const_gen(x) for x in parts[2:end]]

    if !isempty(xs) && xs[1] isa Symbol
        name = xs[1]
        args = xs[2:end]
        return quote
            local _var
            $(esc(core)), _var = add_var(
                $(esc(core)),
                $(map(esc, args)...);
                name = $(Val(name)),
                $(map(esc, kwargs.args)...),
            )
            $(esc(name)) = _var
            _var
        end
    else
        var = gensym(:var)

        return quote
            local $var
            $(esc(core)), $var = add_var(
                $(esc(core)),
                $(map(esc, xs)...);
                $(map(esc, kwargs.args)...),
            )
            $var
        end
    end
end

"""
    @add_par(core, [name,] start; kwargs...)

Macro interface for [`add_par`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_par(core, θ, start)`): binds `θ` to the new `Parameter` in the local scope
  and registers it in `core` for later retrieval as `core.θ` or `model.θ`.
- **Anonymous** (`@add_par(core, start)`): equivalent to `c, p = add_par(c, start)`.

## Example
```julia
c = ExaCore(concrete = Val(true))
@add_par(c, θ, ones(10))  # θ is now in scope; c.θ also works
```
"""
macro add_par(exs...)
    isempty(exs) && error("@add_par requires at least a core and start argument")
    parts, kwargs = _split_macro_args(exs)
    core = parts[1]
    xs = parts[2:end]

    if length(xs) >= 2 && xs[1] isa Symbol
        name = xs[1]
        args = xs[2:end]
        return quote
            local _par
            $(esc(core)), _par = add_par(
                $(esc(core)),
                $(map(esc, args)...);
                name = $(Val(name)),
                $(map(esc, kwargs.args)...),
            )
            $(esc(name)) = _par
            _par
        end
    else
        par = gensym(:par)
        return quote
            local $par
            $(esc(core)), $par = add_par(
                $(esc(core)),
                $(map(esc, xs)...);
                $(map(esc, kwargs.args)...),
            )
            $par
        end
    end
end

"""
    @add_obj(core, [name,] generator; kwargs...)

Macro interface for [`add_obj`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_obj(core, f, generator)`): binds `f` to the new `Objective` in the local scope
  and registers it in `core` for later retrieval as `core.f` or `model.f`.
- **Anonymous** (`@add_obj(core, generator)`): equivalent to `c, o = add_obj(c, generator)`.

## Example
```julia
c = ExaCore(concrete = Val(true))
@add_var(c, x, 10)
@add_obj(c, x[i]^2 for i in 1:10)
```
"""
macro add_obj(exs...)
    isempty(exs) && error("@add_obj requires at least a core and generator argument")
    parts, kwargs = _split_macro_args(exs)
    core = parts[1]
    xs = [_auto_const_gen(x) for x in parts[2:end]]

    if length(xs) >= 2 && xs[1] isa Symbol
        name = xs[1]
        args = xs[2:end]
        return quote
            local _obj
            $(esc(core)), _obj = add_obj(
                $(esc(core)),
                $(map(esc, args)...);
                name = $(Val(name)),
                $(map(esc, kwargs.args)...),
            )
            $(esc(name)) = _obj
            _obj
        end
    else
        obj = gensym(:obj)
        return quote
            local $obj
            $(esc(core)), $obj = add_obj(
                $(esc(core)),
                $(map(esc, xs)...);
                $(map(esc, kwargs.args)...),
            )
            $obj
        end
    end
end

"""
    @add_con(core, [name,] generator; kwargs...)

Macro interface for [`add_con`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_con(core, g, generator)`): binds `g` to the new `Constraint` in the local scope
  and registers it in `core` for later retrieval as `core.g` or `model.g`.
- **Anonymous** (`@add_con(core, generator)`): equivalent to `c, g = add_con(c, generator)`.

Accepts the same keyword arguments as [`add_con`](@ref) (`lcon`, `ucon`, `start`, etc.).

## Example
```julia
c = ExaCore(concrete = Val(true))
@add_var(c, x, 10)
@add_con(c, g, x[i] + x[i+1] for i in 1:9; lcon = -1, ucon = 1)  # g in scope; c.g also works
```
"""
macro add_con(exs...)
    isempty(exs) && error("@add_con requires at least a core and generator argument")
    parts, kwargs = _split_macro_args(exs)
    core = parts[1]
    xs = [_auto_const_gen(x) for x in parts[2:end]]

    if length(xs) >= 2 && xs[1] isa Symbol
        name = xs[1]
        args = xs[2:end]
        return quote
            local _con
            $(esc(core)), _con = add_con(
                $(esc(core)),
                $(map(esc, args)...);
                name = $(Val(name)),
                $(map(esc, kwargs.args)...),
            )
            $(esc(name)) = _con
            _con
        end
    else
        con = gensym(:con)
        return quote
            local $con
            $(esc(core)), $con = add_con(
                $(esc(core)),
                $(map(esc, xs)...);
                $(map(esc, kwargs.args)...),
            )
            $con
        end
    end
end

"""
    @add_con!(core, c1, generator; kwargs...)
    @add_con!(core, c1[idx] += expr for ...; kwargs...)

Macro interface for [`add_con!`](@ref). Updates `core` in the calling scope and returns the
new `ConstraintAugmentation`.

Two calling conventions are supported:

- **Three-argument** (`@add_con!(core, c1, idx => expr for ...)`): the constraint `c1` is
  passed explicitly and the generator yields `idx => expr` pairs.
- **Two-argument** (`@add_con!(core, c1[idx] += expr for ...)`): the constraint and index
  are embedded in the generator via `+=` syntax.  This form delegates to the function-level
  [`add_con!(core, gen)`](@ref), which extracts the target constraint automatically.

See [`add_con!`](@ref) for full semantics and usage notes.

## Example
```julia
c = ExaCore(concrete = Val(true))
@add_var(c, x, 10)
@add_con(c, g, x[i] + x[i+1] for i in 1:9; lcon = -1, ucon = 1)

## Three-argument form:
aug = @add_con!(c, g, i => sin(x[i+1]) for i in 4:6)

## Two-argument form (equivalent):
aug = @add_con!(c, g[i] += sin(x[i+1]) for i in 4:6)
```
"""
macro add_con!(exs...)
    isempty(exs) && error("@add_con! requires core and generator arguments")
    parts, kwargs = _split_macro_args(exs)
    core  = parts[1]

    if length(parts) == 2
        # Two-argument form: @add_con!(core, g[idx] += expr for ...)
        # Delegate to function-level add_con!(core, gen) which extracts
        # the constraint automatically via the ConstraintSlot mechanism.
        gen_expr = _auto_const_gen(parts[2])
        con = gensym(:con)
        return quote
            local $con
            $(esc(core)), $con = add_con!(
                $(esc(core)),
                $(esc(gen_expr));
                $(map(esc, kwargs.args)...),
            )
            $con
        end
    end

    # Three-argument form: @add_con!(core, c1, idx => expr for ...)
    c1    = parts[2]
    args  = [_auto_const_gen(a) for a in parts[3:end]]

    con = gensym(:con)
    return quote
        local $con
        $(esc(core)), $con = add_con!(
            $(esc(core)),
            $(esc(c1)),
            $(map(esc, args)...);
            $(map(esc, kwargs.args)...),
        )
        $con
    end
end

"""
    @add_expr(core, [name,] generator; kwargs...)

Macro interface for [`add_expr`](@ref). Updates `core` in the calling scope.

- **Named** (`@add_expr(core, s, generator)`): binds `s` to the new `Expression` in the local scope
  and registers it in `core` for later retrieval as `core.s` or `model.s`.
- **Anonymous** (`@add_expr(core, generator)`): equivalent to `c, s = add_expr(c, generator)`.

## Example
```julia
c = ExaCore(concrete = Val(true))
@add_var(c, x, 10)
@add_expr(c, s, x[i]^2 for i in 1:10)    # s in scope; c.s also works
@add_obj(c, s[i] + s[i+1] for i in 1:9)
```
"""
macro add_expr(exs...)
    isempty(exs) && error("@add_expr requires at least a core and generator argument")
    parts, kwargs = _split_macro_args(exs)
    core = parts[1]
    xs = [_auto_const_gen(x) for x in parts[2:end]]

    if length(xs) >= 2 && xs[1] isa Symbol
        name = xs[1]
        args = xs[2:end]
        return quote
            local _sub
            $(esc(core)), _sub = add_expr(
                $(esc(core)),
                $(map(esc, args)...);
                name = $(Val(name)),
                $(map(esc, kwargs.args)...),
            )
            $(esc(name)) = _sub
            _sub
        end
    else
        sub = gensym(:sub)
        return quote
            local $sub
            $(esc(core)), $sub = add_expr(
                $(esc(core)),
                $(map(esc, xs)...);
                $(map(esc, kwargs.args)...),
            )
            $sub
        end
    end
end
