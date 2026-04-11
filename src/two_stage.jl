# --- TwoStageExaModel type alias ---

"""
    TwoStageExaModel

Type alias for an [`ExaModel`](@ref) whose `tags` field is a [`TwoStageTags`](@ref).
All two-stage accessor functions (`num_scenarios`, `scenario_var_tags`,
`scenario_con_tags`) are defined on this type.
"""
const TwoStageExaModel{T,VT,E,V,P,O,C,S<:TwoStageTags,R} =
    ExaModel{T,VT,E,V,P,O,C,S,R}

# --- Accessors ---

"""
    num_scenarios(model::TwoStageExaModel)

Return the total number of scenarios.
"""
function num_scenarios(model::TwoStageExaModel)
    return model.tags.ns
end

"""
    scenario_var_tags(model::TwoStageExaModel)

Return the scenario-index vector for variables.  Entry `k` holds the scenario
number of the `k`-th variable (0 = design variable shared across all scenarios).

Users can derive any partition from this vector, e.g.:

```julia
# indices of design variables
findall(==(0), scenario_var_tags(model))

# indices belonging to scenario s
findall(==(s), scenario_var_tags(model))
```
"""
function scenario_var_tags(model::TwoStageExaModel)
    return model.tags.var_scenario
end

"""
    scenario_con_tags(model::TwoStageExaModel)

Return the scenario-index vector for constraints.  Entry `k` holds the scenario
number of the `k`-th constraint (0 = shared constraint).

Users can derive any partition from this vector, e.g.:

```julia
findall(==(s), scenario_con_tags(model))
```
"""
function scenario_con_tags(model::TwoStageExaModel)
    return model.tags.con_scenario
end
