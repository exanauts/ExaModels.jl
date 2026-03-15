# --- Index accessors (ExaModel with TwoStageTags) ---
"""
    scenario_var_indices(model::ExaModel, s::Int)

Return indices of variables belonging to scenario `s` (from `TwoStageTags`).
"""
function scenario_var_indices(model::ExaModel, s::Int)
    tags = model.tags::TwoStageTags
    return findall(==(s), Array(tags.var_scenario))
end

"""
    design_var_indices(model::ExaModel)

Return indices of design variables (scenario tag 0) from `TwoStageTags`.
"""
function design_var_indices(model::ExaModel)
    tags = model.tags::TwoStageTags
    return findall(==(0), Array(tags.var_scenario))
end

"""
    scenario_con_indices(model::ExaModel, s::Int)

Return indices of constraints belonging to scenario `s` (from `TwoStageTags`).
"""
function scenario_con_indices(model::ExaModel, s::Int)
    tags = model.tags::TwoStageTags
    return findall(==(s), Array(tags.con_scenario))
end

"""
    num_scenarios(model::ExaModel)

Return the number of scenarios from `TwoStageTags`.
"""
function num_scenarios(model::ExaModel)
    tags = model.tags::TwoStageTags
    return tags.ns
end

"""
    num_design_vars(model::ExaModel)

Return the number of design variables (tagged with scenario 0).
"""
function num_design_vars(model::ExaModel)
    tags = model.tags::TwoStageTags
    return count(==(0), Array(tags.var_scenario))
end

"""
    num_recourse_vars_per_scenario(model::ExaModel)

Return the number of recourse variables per scenario (counted from scenario 1).
"""
function num_recourse_vars_per_scenario(model::ExaModel)
    tags = model.tags::TwoStageTags
    return count(==(1), Array(tags.var_scenario))
end

"""
    num_cons_per_scenario(model::ExaModel)

Return the number of constraints per scenario (counted from scenario 1).
"""
function num_cons_per_scenario(model::ExaModel)
    tags = model.tags::TwoStageTags
    return count(==(1), Array(tags.con_scenario))
end
