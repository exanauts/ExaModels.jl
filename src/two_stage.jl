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
    get_nscenarios(model::TwoStageExaModel)

Return the total number of scenarios.
"""
function get_nscenarios(model::TwoStageExaModel)
    return model.tags.ns
end

"""
    get_var_scenario(model::TwoStageExaModel)

Return the scenario-index vector for variables.  Entry `k` holds the scenario
number of the `k`-th variable (0 = design variable shared across all scenarios).

Users can derive any partition from this vector, e.g.:

```julia
# indices of design variables
findall(==(0), get_var_scenario(model))

# indices belonging to scenario s
findall(==(s), get_var_scenario(model))
```
"""
function get_var_scenario(model::TwoStageExaModel)
    return model.tags.var_scenario
end

"""
    get_con_scenario(model::TwoStageExaModel)

Return the scenario-index vector for constraints.  Entry `k` holds the scenario
number of the `k`-th constraint (0 = shared constraint).

Users can derive any partition from this vector, e.g.:

```julia
findall(==(s), get_con_scenario(model))
```
"""
function get_con_scenario(model::TwoStageExaModel)
    return model.tags.con_scenario
end
