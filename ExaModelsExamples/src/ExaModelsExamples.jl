module ExaModelsExamples

import ExaModels: ExaModels, NLPModels
import JuMP, NLPModelsJuMP
import PowerModels: PowerModels, silence
import PyCall: @py_str
import MadNLP
import AmplNLReader
import CPUTime: @CPUtime
import SnoopPrecompile
import NLPModelsIpopt: ipopt
import Downloads
import Printf: @printf
import CUDA: CUDA, CUDABackend
import KernelAbstractions: CPU


include("opf.jl")
include("luksanvlcek.jl")
include("distillation.jl")
include("quadrotor.jl")
include("extras.jl")

const NAMES = filter(names(ExaModelsExamples; all = true)) do x
    str = string(x)
    endswith(str, "model") && !startswith(str, "#")
end

export ipopt # rexport

for name in NAMES
    @eval export $name
end

function __init__()
    silence()
end

function __compile__()
    for name in NAMES
        @eval begin
            m = $name()
            ipopt(m; print_level = 0)
        end
    end
end


for name in filter(names(ExaModelsExamples; all=true)) do x
    endswith(string(x), "model")
end
    @eval export $name
end 

end # module ExaModelsExamples
