# Twostage models

"""
    EachScenario

Marker type for two-stage variable and constraint declarations.
Pass `EachScenario()` as an argument to `variable()` or `constraint()`
to indicate that the declaration is replicated for each scenario.

# Examples
```julia
core = TwoStageExaCore(ns)
d = variable(core, nd)                       # design variables (shared)
v = variable(core, nv, EachScenario())       # nv recourse variables per scenario
constraint(core, (v[i] + d[1] for i in data), EachScenario())
```
"""
struct EachScenario end

struct TwoStageTags{VI <: AbstractVector{Int}}
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
const TwoStageExaCore{T,VT,B} = ExaCore{T,VT,B,S} where S <: TwoStageTags

"""
    TwoStageExaCore(ns; backend=nothing)

Create an `ExaCore` with `TwoStageTags` for building two-stage stochastic models
with `ns` scenarios.
"""
function TwoStageExaCore(ns::Int = 0; backend = nothing, kwargs...)
    return ExaCore(;
        backend,
        tags = TwoStageTags(
            ns,
            convert_array(zeros(Int, 0), backend),
            convert_array(zeros(Int, 0), backend)
        ),
        kwargs...
    )
end

# --- EachScenario variable method ---
"""
    variable(core::TwoStageExaCore, n, EachScenario(); kwargs...)

Create `n` variables per scenario (total `ns * n` variables), automatically
tagged with their scenario index. Design variables (shared) are created
with the standard `variable(core, n)` call (tagged with scenario 0).
"""
function variable(
    c::C, n::Integer, ::EachScenario;
    start = zero(T), lvar = T(-Inf), uvar = T(Inf),
) where {T, VT <: AbstractVector{T}, B, S <: TwoStageTags, C <: ExaCore{T, VT, B, S}}
    ns = c.tags.ns
    @assert ns > 0 "TwoStageExaCore must be created with ns > 0 to use EachScenario()"
    scenario_tags = [k for k in 1:ns for _ in 1:n]
    return variable(c, ns * n; start, lvar, uvar, scenario = scenario_tags)
end

# --- EachScenario constraint methods ---
"""
    constraint(core::TwoStageExaCore, gen::Base.Generator, EachScenario(); kwargs...)

Create constraints from a generator and automatically tag them per scenario.
The total number of constraints must be divisible by `ns`. Constraints are
assigned to scenarios in order: first `nc_per_s` go to scenario 1, etc.
"""
function constraint(
    c::C, gen::Base.Generator, ::EachScenario;
    start = zero(T), lcon = zero(T), ucon = zero(T),
) where {T, VT <: AbstractVector{T}, B, S <: TwoStageTags, C <: ExaCore{T, VT, B, S}}
    ns = c.tags.ns
    @assert ns > 0 "TwoStageExaCore must be created with ns > 0 to use EachScenario()"
    nitr = length(gen.iter)
    @assert nitr % ns == 0 "Number of constraints ($nitr) must be divisible by ns ($ns)"
    nc_per_s = nitr ÷ ns
    scenario_tags = [k for k in 1:ns for _ in 1:nc_per_s]
    return constraint(c, gen; start, lcon, ucon, scenario = scenario_tags)
end

export EachScenario, TwoStageExaCore
