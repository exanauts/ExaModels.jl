module ExaModelsMadNLP

import ExaModels
import MadNLP
import MathOptInterface

const MOI = MathOptInterface

ExaModels.MadNLPOptimizer(backend = nothing; kwargs...) = ExaModels.Optimizer(MadNLP.madnlp, nothing; kwargs...)

function ExaModels.result_status_translator(::typeof(MadNLP.madnlp), status)
    Base.get(_RESULT_STATUS_CODES, status, MOI.UNKNOWN_RESULT_STATUS)
end

function ExaModels.termination_status_translator(::typeof(MadNLP.madnlp), status)
    Base.get(_TERMINATION_STATUS_CODES, status, MOI.OTHER_ERROR)
end

const _RESULT_STATUS_CODES = Dict{Symbol,MathOptInterface.ResultStatusCode}(
    :first_order => MOI.FEASIBLE_POINT,
    :acceptable => MOI.NEARLY_FEASIBLE_POINT,
    :infeasible => MOI.INFEASIBLE_POINT,
)
const _TERMINATION_STATUS_CODES = Dict{Symbol,MOI.TerminationStatusCode}(
    :first_order => MOI.LOCALLY_SOLVED,
    :acceptable => MOI.ALMOST_LOCALLY_SOLVED,
    :small_step => MOI.SLOW_PROGRESS,
    :infeasible => MOI.INFEASIBLE_OR_UNBOUNDED,
    :max_iter => MOI.ITERATION_LIMIT,
    :max_time => MOI.TIME_LIMIT,
    :user => MOI.INTERRUPTED,
    :exception=> MOI.OTHER_ERROR,
)

end
