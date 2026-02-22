# Twostage models
struct TwoStageTags{VI <: AbstractVector{Int}}
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
function TwoStageExaCore(args...; backend = nothing, kwargs...)
    return ExaCore(
        args...;
        backend, 
        tags = TwoStageTags(
            convert_array(zeros(Int, 0), backend),
            convert_array(zeros(Int, 0), backend)
        ),
        kwargs...
    )
end

export TwoStageExaCore
