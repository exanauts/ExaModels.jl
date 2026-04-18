"""
    EachScenario

Marker type used with [`add_var`](@ref), [`add_par`](@ref), and [`add_con`](@ref)
to indicate that the declaration is replicated for each scenario in a two-stage
stochastic program.

Pass `EachScenario()` as the second positional argument (before dimensions) to
create per-scenario (recourse) variables, parameters, or constraints.  Omitting
it creates first-stage (design) components shared across all scenarios.

## Example
```julia
core = TwoStageExaCore(3)                          # 3 scenarios
c, d = add_var(core, 2)                            # 2 design variables (shared)
c, v = add_var(core, EachScenario(), 4)            # 4 recourse variables per scenario
c, g = add_con(core, EachScenario(), v[i] for i in 1:4)  # per-scenario constraints
```
"""
struct EachScenario end
struct FirstStageTag <: AbstractTag end
struct SecondStageTag <: AbstractTag end
struct TwoStageExaModelTag{VI<:AbstractVector{Int}} <: AbstractExaModelTag
    nscen::Int
    var_scen::VI
    con_scen::VI
end

abstract type AbstractTwoStageVariable{V} end

@inline _scen_each_tag(nscen, nitr) = (k for _ in 1:nitr, k in 1:nscen)
@inline _scen_full_tag(nscen, nitr) = (k for _ in 1:div(nitr, nscen), k in 1:nscen)

const FirstStageVariable{S,O} = Variable{S,O,<: FirstStageTag}
const SecondStageVariable{S,O} = Variable{S,O,<: SecondStageTag} 
const FirstStageParameter{S,O} = Parameter{S,O,<: FirstStageTag}
const SecondStageParameter{S,O} = Parameter{S,O,<: SecondStageTag} 
const FirstStageExpression{S,F,I} = Expression{S,F,I,<: FirstStageTag}
const SecondStageExpression{S,F,I} = Expression{S,F,I,<: SecondStageTag}
const FirstStageConstraint{F,I,O,S} = Constraint{F,I,O,S,<: FirstStageTag}
const SecondStageConstraint{F,I,O,S} = Constraint{F,I,O,S,<: SecondStageTag}
"""
    TwoStageExaCore{T,VT,B}

Type alias for an [`ExaCore`](@ref) whose `tag` is a [`TwoStageExaModelTag`].
Create one with [`TwoStageExaCore(nscen)`](@ref).
"""
const TwoStageExaCore{T,VT,B} = ExaCore{T,VT,B,<:TwoStageExaModelTag}

"""
    TwoStageExaModel{T,VT,E,V,P,O,C,R}

Type alias for an [`ExaModel`](@ref) built from a [`TwoStageExaCore`](@ref).
Use [`get_nscen`](@ref), [`get_var_scen`](@ref), and [`get_con_scen`](@ref)
to query the scenario structure after building the model.
"""
const TwoStageExaModel{T,VT,E,V,P,O,C,R} = ExaModel{T,VT,E,V,P,O,C,<:TwoStageExaModelTag,R}

@inline function Base.getindex(v::V, is...) where {V<:SecondStageVariable}
    is = (is..., DataSource()[2])
    @assert(length(is) == length(v.size), "Variable index dimension error")
    _bound_check(v.size, is)
    Var(v.offset + idxx(is .- (_start.(v.size) .- 1), _length.(v.size)))
end

"""
    TwoStageExaCore(nscen; backend = nothing, concrete = Val(false), kwargs...)

Create an [`ExaCore`](@ref) for building two-stage stochastic programs with
`nscen` scenarios.

Use [`add_var`](@ref), [`add_par`](@ref), and [`add_con`](@ref) with
[`EachScenario()`](@ref) to declare per-scenario components, or without it for
first-stage (design) components.

## Example
```julia
core = TwoStageExaCore(5)                    # 5 scenarios
c, d = add_var(core, 3)                      # 3 design variables
c, v = add_var(c, EachScenario(), 2)         # 2 recourse variables per scenario
model = ExaModel(c)
```
"""
function TwoStageExaCore(ns::Integer; backend = nothing, concrete = Val(false), kwargs...)
    return ExaCore(;
        backend,
        concrete,
        tag = TwoStageExaModelTag(
            ns,
            convert_array(zeros(Int, 0), backend),
            convert_array(zeros(Int, 0), backend),
        ),
        kwargs...,
    )
end

"""
    add_var(core::TwoStageExaCore, dims...; start = 0, lvar = -Inf, uvar = Inf, name = nothing)

Add first-stage (design) variables to a two-stage core.  These are shared
across all scenarios and tagged with `FirstStageTag()`.
"""
function add_var(
    c::TwoStageExaCore{T,VT,B},
    ns...;
    name = nothing,
    start = zero(T),
    lvar = T(-Inf),
    uvar = T(Inf),
    ) where {T,VT<:AbstractVector{T},B}
    
    len = total(ns)
    append!(c.backend, c.tag.var_scen, 0, len)
    return _add_var(
        c, FirstStageTag(), name, start, lvar, uvar, ns...
    )    
end
"""
    add_var(core::TwoStageExaCore, ::EachScenario, dims...; start = 0, lvar = -Inf, uvar = Inf, name = nothing)

Add second-stage (recourse) variables to a two-stage core.  Creates
`prod(dims) * nscen` variables total — one copy of the block per scenario —
tagged with `SecondStageTag()`.  The last dimension of the resulting variable
indexes the scenario.
"""
function add_var(
    c::TwoStageExaCore{T,VT,B},
    ::EachScenario,
    ns...;
    name = nothing,
    start = zero(T),
    lvar = T(-Inf),
    uvar = T(Inf),
    ) where {T,VT<:AbstractVector{T},B}
    nscen = c.tag.nscen
    len = total(ns)
    append!(c.backend, c.tag.var_scen, _scen_each_tag(nscen, len), len * nscen)
    return _add_var(
        c, SecondStageTag(), name, start, lvar, uvar, ns..., nscen
    )    
end

"""
    add_par(core::TwoStageExaCore, start; name = nothing)

Add first-stage parameters to a two-stage core.  Tagged with `FirstStageTag()`.
"""
function add_par(
    c::TwoStageExaCore{T,VT,B},
    start;
    name = nothing,
    ) where {T,VT<:AbstractVector{T},B}
    return _add_par(
        c,
        FirstStageTag(),
        name,
        start
    )
end
"""
    add_par(core::TwoStageExaCore, ::EachScenario, start::AbstractVector; name = nothing)

Add second-stage parameters to a two-stage core.  The parameter vector `start`
is replicated for each scenario.  Tagged with `SecondStageTag()`.
"""
function add_par(
    c::TwoStageExaCore{T,VT,B},
    ::EachScenario,
    start::AbstractVector;
    name = nothing,
    ) where {T,VT<:AbstractVector{T},B}
    return _add_par(
        c,
        SecondStageTag(),
        name,
        cat((start for _ in 1:c.tag.nscen)...; dims = ndims(start) + 1)
    )
end

"""
    add_con(core::TwoStageExaCore, dims_or_gen...; start = 0, lcon = 0, ucon = 0, name = nothing)

Add first-stage constraints to a two-stage core.  Accepts the same forms as
the base [`add_con`](@ref) (generator or dims).  Tagged with `FirstStageTag()`.
"""
function add_con(
    c::C,
    ns...;
    name = nothing,
    tag = nothing,
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
    ) where {T,VT<:AbstractVector{T},B,S<:TwoStageExaModelTag,C<:ExaCore{T,VT,B,S}}

    gen = _get_generator(ns)
    dims = _infer_subexpr_dims(gen.iter)
    gen = _adapt_gen(gen)
    f = _simdfunction(T, gen.f(DataSource()), c.ncon, c.nnzj, c.nnzh)
    pars = gen.iter
    
    return _add_con(
        c, gen, start, lcon, ucon, name, FirstStageTag()
    )
end

"""
    add_con(core::TwoStageExaCore, ::EachScenario, dims_or_gen...; start = 0, lcon = 0, ucon = 0, name = nothing)

Add second-stage (per-scenario) constraints to a two-stage core.  The constraint
expression is replicated for each scenario.  The iterator element is wrapped so
that `DataSource()[1]` accesses the original data and `DataSource()[2]` gives the
scenario index.
"""
function add_con(
    c::C,
    ::EachScenario,
    ns...;
    name = nothing,
    tag = nothing,
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
    ) where {T,VT<:AbstractVector{T},B,S<:TwoStageExaModelTag,C<:ExaCore{T,VT,B,S}}
    
    gen = _get_generator(ns)
    nscen = c.tag.nscen
    len = length(gen)
    append!(c.backend, c.tag.var_scen, _scen_each_tag(nscen, len), len * nscen)

    gen = _get_generator(ns)
    dims = _infer_subexpr_dims(gen.iter)
    gen = _adapt_gen(gen)
    pars = collect((i,k) for i in gen.iter for k in 1:nscen)
    f = _simdfunction(T, gen.f(DataSource()[1]), c.ncon, c.nnzj, c.nnzh)
    _add_con(c, f, pars, dims, start, lcon, ucon, name, tag)
end

# --- Accessors ---

"""
    get_nscen(model::TwoStageExaModel)

Return the total number of scenarios.
"""
function get_nscen(model::TwoStageExaModel)
    return model.tag.nscen
end

"""
    get_var_scen(model::TwoStageExaModel)

Return the scenario-index vector for variables.  Entry `k` holds the scenario
number of the `k`-th variable (0 = design variable shared across all scenarios).

```julia
findall(==(0), get_var_scen(model))  # design variable indices
findall(==(s), get_var_scen(model))  # indices for scenario s
```
"""
function get_var_scen(model::TwoStageExaModel)
    return model.tag.var_scen
end

"""
    get_con_scen(model::TwoStageExaModel)

Return the scenario-index vector for constraints.  Entry `k` holds the scenario
number of the `k`-th constraint (0 = shared).

```julia
findall(==(s), get_con_scen(model))  # constraint indices for scenario s
```
"""
function get_con_scen(model::TwoStageExaModel)
    return model.tag.con_scen
end

export EachScenario, SecondStageVariable,
       TwoStageExaCore, TwoStageExaModel,
       get_nscen, get_var_scen, get_con_scen
