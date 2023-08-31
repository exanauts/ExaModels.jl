module ExaModelsBenchmark

import ExaModels: ExaModels, NLPModels
import ExaModelsExamples
import JuMP
import PyCall: @py_str
import MadNLP
import AmplNLReader
import NLPModelsIpopt: ipopt
import Downloads
import Printf: @printf
using LibGit2, Downloads, CpuId, Printf
import UUIDs

include("opf.jl")
include("luksanvlcek.jl")
include("distillation.jl")
include("quadrotor.jl")
include("benchmark.jl")

function MathOptNLPModel(jm)
    JuMP.set_optimizer(jm, MadNLP.Optimizer)
    JuMP.set_optimizer_attribute(jm, "max_iter", 0)
    JuMP.set_optimizer_attribute(jm, "print_level", MadNLP.ERROR)
    JuMP.optimize!(jm)
    return jm.moi_backend.optimizer.model.nlp
end


function __init__()
    if haskey(ENV, "EXA_MODELS_DEPOT")
        global TMPDIR = ENV["EXA_MODELS_DEPOT"]
    else
        global TMPDIR = tempname()
        mkdir(TMPDIR)
    end
end

export runbenchmark

for name in filter(names(ExaModelsExamples; all=true)) do x
    endswith(string(x), "model")
end
    @eval export $name
end 

end # module ExaModelsBenchmark
