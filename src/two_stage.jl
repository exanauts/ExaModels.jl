# ── Two-stage model support ───────────────────────────────────────────────────

# --- Marker type ---

"""
    EachScenario

Marker type for two-stage variable and constraint declarations.
Pass `EachScenario()` as a positional argument to [`add_var`](@ref) or
[`add_con`](@ref) to indicate that the declaration is replicated for each
scenario.

# Examples
```julia
c = TwoStageExaCore(ns; concrete = Val(true))
c, d = add_var(c, nd)                                               # design variables (shared)
c, v = add_var(c, nv, EachScenario())                               # nv recourse vars per scen
c, g = add_con(c, (v[i] + d[1] for i in data), EachScenario())     # constraints per scen
```
"""
struct EachScenario end

# --- Tags struct ---

struct TwoStageTags{VI<:AbstractVector{Int}}
    ns::Int
    var_scen::VI
    con_scen::VI
end

function append_var_tags(tags::TwoStageTags, backend, len; scen = 0)
    append!(backend, tags.var_scen, scen, len)
end
function append_con_tags(tags::TwoStageTags, backend, len; scen = 0)
    append!(backend, tags.con_scen, scen, len)
end

# --- TwoStageExaCore ---

const TwoStageExaCore{T,VT,B} = ExaCore{T,VT,B,S} where {S<:TwoStageTags}

"""
    TwoStageExaCore(ns; backend = nothing, concrete = Val(false), kwargs...)

Create an [`ExaCore`](@ref) for a two-stage stochastic program with `ns` scenarios.

Variables and constraints added with [`EachScenario`](@ref) are replicated for
each scenario and tagged 1 … ns.  Design variables (shared across all scenarios)
are added without `EachScenario` and receive tag 0.
"""
function TwoStageExaCore(ns::Integer; backend = nothing, concrete = Val(false), kwargs...)
    return ExaCore(;
        backend,
        concrete,
        tags = TwoStageTags(
            ns,
            convert_array(zeros(Int, 0), backend),
            convert_array(zeros(Int, 0), backend),
        ),
        kwargs...,
    )
end

# --- add_var with EachScenario ---
#
# One overload per dimensionality to match the explicit overloads in nlp.jl.
# EachScenario is placed after the dimension arguments.

"""
    add_var(core::TwoStageExaCore, n, EachScenario(); kwargs...)
    add_var(core::TwoStageExaCore, n1, n2, EachScenario(); kwargs...)
    add_var(core::TwoStageExaCore, n1, n2, n3, EachScenario(); kwargs...)

Add variables to a two-stage core, replicated for each scenario.
The total variables created is `ns * prod(dims)`, automatically tagged
1 … ns.  All keyword arguments accepted by [`add_var`](@ref) are forwarded.
"""
function add_var(
    c::C,
    n,
    ::EachScenario;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    ns = c.tags.ns
    len = total((n,))
    scen_tags = [k for k in 1:ns for _ in 1:len]
    return add_var(c, ns * len; scen = scen_tags, kwargs...)
end

function add_var(
    c::C,
    n1::N1,
    n2::N2,
    ::EachScenario;
    kwargs...,
) where {
    T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S},
    N1<:Union{Integer,AbstractUnitRange},
    N2<:Union{Integer,AbstractUnitRange},
}
    ns = c.tags.ns
    len = total((n1, n2))
    scen_tags = [k for k in 1:ns for _ in 1:len]
    return add_var(c, ns * len; scen = scen_tags, kwargs...)
end

function add_var(
    c::C,
    n1::N1,
    n2::N2,
    n3::N3,
    ::EachScenario;
    kwargs...,
) where {
    T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S},
    N1<:Union{Integer,AbstractUnitRange},
    N2<:Union{Integer,AbstractUnitRange},
    N3<:Union{Integer,AbstractUnitRange},
}
    ns = c.tags.ns
    len = total((n1, n2, n3))
    scen_tags = [k for k in 1:ns for _ in 1:len]
    return add_var(c, ns * len; scen = scen_tags, kwargs...)
end

# --- add_con with EachScenario ---

# Helper: build scen_tags for nitr constraints split evenly across ns scenarios.
function _scen_tags(ns, nitr)
    @assert nitr % ns == 0 "Number of constraints ($nitr) must be divisible by ns ($ns)"
    nc = nitr ÷ ns
    return [k for k in 1:ns for _ in 1:nc]
end

"""
    add_con(core::TwoStageExaCore, gen, EachScenario(); kwargs...)
    add_con(core::TwoStageExaCore, gen, EachScenario(), gens...; kwargs...)
    add_con(core::TwoStageExaCore, expr::AbstractNode, pars, EachScenario(); kwargs...)
    add_con(core::TwoStageExaCore, n, EachScenario(); kwargs...)

Add constraints to a two-stage core, automatically assigning scenario tags.

- **Generator form**: `gen.iter` must have length divisible by `ns`; constraints
  are tagged 1 … ns in order.  Additional generators `gens...` are appended via
  [`add_con!`](@ref) without scenario tagging.
- **AbstractNode form**: `pars` must have length divisible by `ns`.
- **Integer form**: allocates `ns * n` empty constraint rows tagged 1 … ns; use
  [`add_con!`](@ref) to fill them in.

All keyword arguments accepted by [`add_con`](@ref) are forwarded.
"""
function add_con(
    c::C,
    gen::Base.Generator,
    ::EachScenario;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    scen_tags = _scen_tags(c.tags.ns, length(gen.iter))
    return add_con(c, gen; scen = scen_tags, kwargs...)
end

function add_con(
    c::C,
    gen::Base.Generator,
    ::EachScenario,
    gens::Base.Generator...;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    c, con = add_con(c, gen, EachScenario(); kwargs...)
    for g in gens
        c, _ = add_con!(c, con, g)
    end
    return (c, con)
end

function add_con(
    c::C,
    expr::N,
    pars,
    ::EachScenario;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S},N<:AbstractNode}
    scen_tags = _scen_tags(c.tags.ns, length(pars))
    return add_con(c, expr, pars; scen = scen_tags, kwargs...)
end

function add_con(
    c::C,
    n::Integer,
    ::EachScenario;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    ns = c.tags.ns
    scen_tags = [k for k in 1:ns for _ in 1:n]
    return add_con(c, ns * n; scen = scen_tags, kwargs...)
end

# --- TwoStageExaModel type alias ---

"""
    TwoStageExaModel

Type alias for an [`ExaModel`](@ref) whose `tags` field is a [`TwoStageTags`](@ref).
Two-stage accessor functions [`get_nscen`](@ref), [`get_var_scen`](@ref), and
[`get_con_scen`](@ref) are defined on this type.
"""
const TwoStageExaModel{T,VT,E,V,P,O,C,S<:TwoStageTags,R} =
    ExaModel{T,VT,E,V,P,O,C,S,R}

# --- Accessors ---

"""
    get_nscen(model::TwoStageExaModel)

Return the total number of scenarios.
"""
function get_nscen(model::TwoStageExaModel)
    return model.tags.ns
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
    return model.tags.var_scen
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
    return model.tags.con_scen
end

export EachScenario, TwoStageExaCore, TwoStageExaModel, get_nscen, get_var_scen, get_con_scen
