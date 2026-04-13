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
const TwoStageExaCore{T,VT,B} = ExaCore{T,VT,B,<:TwoStageExaModelTag}
const TwoStageExaModel{T,VT,E,V,P,O,C,R} = ExaModel{T,VT,E,V,P,O,C,<:TwoStageExaModelTag,R}

@inline function Base.getindex(v::V, is...) where {V<:SecondStageVariable}
    is = (is..., DataSource()[2])
    @assert(length(is) == length(v.size), "Variable index dimension error")
    _bound_check(v.size, is)
    Var(v.offset + idxx(is .- (_start.(v.size) .- 1), _length.(v.size)))
end

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
