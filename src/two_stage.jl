# --- TwoStageExaModel type alias ---

"""
    TwoStageExaModel

Type alias for an [`ExaModel`](@ref) whose `tags` field is a [`TwoStageTags`](@ref).
All two-stage accessor functions (`get_nscen`, `get_var_scen`,
`get_con_scen`) are defined on this type.
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

Return the scenario-index vector for variables.  Entry `k` holds the scen
number of the `k`-th variable (0 = design variable shared across all scenarios).

Users can derive any partition from this vector, e.g.:

```julia
# indices of design variables
findall(==(0), get_var_scen(model))

# indices belonging to scenario s
findall(==(s), get_var_scen(model))
```
"""
function get_var_scen(model::TwoStageExaModel)
    return model.tags.var_scen
end

"""
    get_con_scen(model::TwoStageExaModel)

Return the scenario-index vector for constraints.  Entry `k` holds the scen
number of the `k`-th constraint (0 = shared constraint).

Users can derive any partition from this vector, e.g.:

```julia
findall(==(s), get_con_scen(model))
```
"""
function get_con_scen(model::TwoStageExaModel)
    return model.tags.con_scen
end
