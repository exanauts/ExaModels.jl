# Twostage models

"""
    EachScenario

Marker type for two-stage variable and constraint declarations.
Pass `EachScenario()` as the last positional argument to [`add_var`](@ref) or
[`add_con`](@ref) to indicate that the declaration is replicated for each scenario.

# Examples
```julia
c = TwoStageExaCore(ns; concrete = Val(true))
c, d = add_var(c, nd)                                          # design variables (shared)
c, v = add_var(c, nv, EachScenario())                          # nv recourse vars per scenario
c, g = add_con(c, (v[i] + d[1] for i in data), EachScenario()) # constraints per scenario
```
"""
struct EachScenario end

struct TwoStageTags{VI<:AbstractVector{Int}}
    ns::Int
    var_scenario::VI
    con_scenario::VI
end
function append_var_tags(tags::TwoStageTags, backend, len; scenario = 0)
    append!(backend, tags.var_scenario, scenario, len)
end
function append_con_tags(tags::TwoStageTags, backend, len; scenario = 0)
    append!(backend, tags.con_scenario, scenario, len)
end
const TwoStageExaCore{T,VT,B} = ExaCore{T,VT,B,S} where {S<:TwoStageTags}
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

# --- EachScenario: add_var ---

"""
    add_var(core::TwoStageExaCore, n, EachScenario(); kwargs...)

Add `n` variables per scenario to `core`, returning `(core, Variable)`.
The total number of variables added is `ns * total(n)`, automatically tagged
with their scenario index (1 … ns).  Design variables (shared across scenarios)
are added with the standard `add_var(core, n)` call (tagged with scenario 0).

`n` may be an `Integer` or an `AbstractUnitRange`.
"""
function add_var(
    c::C,
    n,
    ::EachScenario;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    ns = c.tags.ns
    len = total((n,))
    scenario_tags = [k for k in 1:ns for _ in 1:len]
    return add_var(c, ns * len; scenario = scenario_tags, kwargs...)
end

# --- EachScenario: add_con (generator form) ---

"""
    add_con(core::TwoStageExaCore, gen, EachScenario(); kwargs...)

Add constraints from `gen` to `core`, automatically tagging each constraint with
its scenario index.  `gen` must iterate over all scenarios combined; its length
must be divisible by `ns`.  Returns `(core, Constraint)`.

All keyword arguments accepted by [`add_con`](@ref) are forwarded.
"""
function add_con(
    c::C,
    gen::Base.Generator,
    ::EachScenario;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    ns = c.tags.ns
    nitr = length(gen.iter)
    @assert nitr % ns == 0 "Number of constraints ($nitr) must be divisible by ns ($ns)"
    nc_per_s = nitr ÷ ns
    scenario_tags = [k for k in 1:ns for _ in 1:nc_per_s]
    return add_con(c, gen; scenario = scenario_tags, kwargs...)
end

"""
    add_con(core::TwoStageExaCore, gen, gens..., EachScenario(); kwargs...)

Multi-generator form of `add_con` with per-scenario tagging.  The first
generator creates the base constraint; subsequent generators augment it via
[`add_con!`](@ref).  All generators must have length divisible by `ns`.
Returns `(core, Constraint)`.
"""
function add_con(
    c::C,
    gen::Base.Generator,
    gens_and_es::Union{Base.Generator,EachScenario}...;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S}}
    # Split trailing EachScenario() from the generator list
    gens = filter(x -> x isa Base.Generator, gens_and_es)
    has_es = any(x -> x isa EachScenario, gens_and_es)
    if has_es
        c, con = add_con(c, gen, EachScenario(); kwargs...)
        for g in gens
            c, _ = add_con!(c, con, g)
        end
        return (c, con)
    else
        return invoke(add_con, Tuple{typeof(c),Base.Generator,Vararg{Base.Generator}}, c, gen, gens...; kwargs...)
    end
end

"""
    add_con(core::TwoStageExaCore, expr::AbstractNode, pars, EachScenario(); kwargs...)

Low-level form of `add_con` with per-scenario tagging.  `pars` must have length
divisible by `ns`.  Returns `(core, Constraint)`.
"""
function add_con(
    c::C,
    expr::N,
    pars,
    ::EachScenario;
    kwargs...,
) where {T,VT<:AbstractVector{T},B,S<:TwoStageTags,C<:ExaCore{T,VT,B,S},N<:AbstractNode}
    ns = c.tags.ns
    nitr = length(pars)
    @assert nitr % ns == 0 "Number of constraints ($nitr) must be divisible by ns ($ns)"
    nc_per_s = nitr ÷ ns
    scenario_tags = [k for k in 1:ns for _ in 1:nc_per_s]
    return add_con(c, expr, pars; scenario = scenario_tags, kwargs...)
end

export EachScenario, TwoStageExaCore
