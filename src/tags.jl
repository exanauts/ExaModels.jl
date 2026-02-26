# Twostage models
struct TwoStageTags{VI <: AbstractVector{Bool}}
    ns::Int
    var_scenario::VI
    con_scenario::VI
end
function append_var_tags(tags::TwoStageTags, backend, len; scenario = false)
    if scenario isa AbstractArray
        length(scenario) == len || throw(
            DimensionMismatch("scenario tag length ($(length(scenario))) must match count ($len)"),
        )
    end
    append!(backend, tags.var_scenario, Bool.(scenario), len)
end
function append_con_tags(tags::TwoStageTags, backend, len; scenario = false)
    if scenario isa AbstractArray
        length(scenario) == len || throw(
            DimensionMismatch("scenario tag length ($(length(scenario))) must match count ($len)"),
        )
    end
    append!(backend, tags.con_scenario, Bool.(scenario), len)
end
const TwoStageExaCore{T, VT, B} = ExaCore{T, VT, B, S} where {S <: TwoStageTags}
function TwoStageExaCore(args...; ns::Int, backend = nothing, kwargs...)
    return ExaCore(
        args...;
        backend,
        tags = TwoStageTags(
            ns,
            convert_array(Bool[], backend),
            convert_array(Bool[], backend),
        ),
        kwargs...,
    )
end

export TwoStageExaCore
