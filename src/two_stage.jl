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
c, v = add_var(c, EachScenario(), nv)                               # nv recourse vars per scen
c, g = add_con(c, EachScenario(), (v[i] + d[1] for i in data))     # constraints per scen
```
"""
struct EachScenario end

# --- SecondStageVariable ---

"""
    SecondStageVariable{V, Ndim} <: AbstractVariable

Wrapper returned by `add_var(core, EachScenario(), dims...)`.  `Ndim` is the
total number of dimensions of the underlying variable, including the scenario
dimension appended at the end.

**Indexing**:
- `v[idx...]` with `length(idx) == Ndim` — full access, passed directly to the
  underlying variable.
- `v[idx...]` with fewer indices — partial access: the scenario index is
  auto-injected from the current iterator element, so `v[j]` inside
  `(expr for j=1:nv, k=1:ns)` resolves to `v.var[j, k]`.

This makes `x[j]` in an `EachScenario` generator body transparently refer to
the j-th variable of the current scenario without any extra bookkeeping.
"""
struct SecondStageVariable{V<:AbstractVariable, Ndim} <: AbstractVariable
    var::V
end

# Forward property lookups (e.g. .offset, .size) so that solution(),
# multipliers_L(), multipliers_U() work transparently on SecondStageVariable.
@inline function Base.getproperty(h::SecondStageVariable, name::Symbol)
    name === :var && return getfield(h, :var)
    return getproperty(getfield(h, :var), name)
end

# Full indexing (exactly Ndim args): direct passthrough.
@inline Base.getindex(h::SecondStageVariable{V, Ndim}, idx::Vararg{Any, Ndim}) where {V, Ndim} =
    h.var[idx...]

# Partial indexing (fewer args): auto-append the scenario from iterator position Ndim.
@inline Base.getindex(h::SecondStageVariable{V, Ndim}, idx...) where {V, Ndim} =
    h.var[idx..., DataIndexed(DataSource(), Ndim)]

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
    TwoStageExaCore(ns; backend = nothing, concrete = Val(true), kwargs...)

Create an [`ExaCore`](@ref) for a two-stage stochastic program with `ns` scenarios.

Variables and constraints added with [`EachScenario`](@ref) are replicated for
each scenario and tagged 1 … ns.  Design variables (shared across all scenarios)
are added without `EachScenario` and receive tag 0.
"""
function TwoStageExaCore(ns::Integer; backend = nothing, concrete = Val(true), kwargs...)
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
# EachScenario is placed before the dimension arguments, allowing any number
# of dimensions via ns... without conflicting with the base add_var dispatch.

"""
    add_var(core::TwoStageExaCore, EachScenario(), ns...; kwargs...)

Add variables to a two-stage core, replicated for each scenario.
The total variables created is `get_nscen(core) * prod(ns)`, automatically
tagged 1 … ns.  All keyword arguments accepted by [`add_var`](@ref) are
forwarded.

# Examples
```julia
c, v  = add_var(c, EachScenario(), nv)          # 1-D: nv vars per scenario
c, v2 = add_var(c, EachScenario(), n1, n2)      # 2-D block per scenario
```
"""
function add_var(
    c::C,
    ::EachScenario,
    ns...;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    nscen = c.tags.ns
    len = total(ns)
    scen_tags = [k for k in 1:nscen for _ in 1:len]
    c, v = add_var(c, ns..., nscen; scen = scen_tags, kwargs...)
    Ndim = length(ns) + 1   # scenario is the last dimension
    return c, SecondStageVariable{typeof(v), Ndim}(v)
end

# --- add_con with EachScenario ---

# Helper: build scen_tags for nitr constraints split evenly across ns scenarios.
function _scen_tags(ns, nitr)
    @assert nitr % ns == 0 "Number of constraints ($nitr) must be divisible by ns ($ns)"
    nc = nitr ÷ ns
    return [k for k in 1:ns for _ in 1:nc]
end

"""
    add_con(core::TwoStageExaCore, EachScenario(), gen; kwargs...)
    add_con(core::TwoStageExaCore, EachScenario(), gen, gens...; kwargs...)
    add_con(core::TwoStageExaCore, EachScenario(), expr::AbstractNode, pars; kwargs...)
    add_con(core::TwoStageExaCore, EachScenario(), dims...; kwargs...)

Add constraints to a two-stage core, automatically assigning scenario tags.

- **Generator form**: `gen.iter` must have length divisible by `ns`; constraints
  are tagged 1 … ns in order.  Additional generators `gens...` are appended via
  [`add_con!`](@ref) without scenario tagging.
- **AbstractNode form**: `pars` must have length divisible by `ns`.
- **Dims form**: allocates empty constraint rows for a `(dims..., nscen)`-shaped
  block — mirroring the `add_con(core, dims...)` signature in [`add_con`](@ref).
  Use [`add_con!`](@ref) to fill in terms afterwards.

All keyword arguments accepted by [`add_con`](@ref) are forwarded.
"""
function add_con(
    c::C,
    ::EachScenario,
    gen::Base.Generator;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    scen_tags = _scen_tags(c.tags.ns, length(gen.iter))
    return add_con(c, gen; scen = scen_tags, kwargs...)
end

function add_con(
    c::C,
    ::EachScenario,
    gen::Base.Generator,
    gens::Base.Generator...;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    c, con = add_con(c, EachScenario(), gen; kwargs...)
    for g in gens
        c, _ = add_con!(c, con, g)
    end
    return (c, con)
end

function add_con(
    c::C,
    ::EachScenario,
    expr::N,
    pars;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S},N<:AbstractNode}
    scen_tags = _scen_tags(c.tags.ns, length(pars))
    return add_con(c, expr, pars; scen = scen_tags, kwargs...)
end

function add_con(
    c::C,
    ::EachScenario,
    dims...;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    nscen = c.tags.ns
    n = total(dims)
    scen_tags = [k for k in 1:nscen for _ in 1:n]
    return add_con(c, dims..., nscen; scen = scen_tags, kwargs...)
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

export EachScenario, SecondStageVariable,
       TwoStageExaCore, TwoStageExaModel,
       get_nscen, get_var_scen, get_con_scen
