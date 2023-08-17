module ExaModelsExamples

import Downloads
import ExaModels: ExaModels, NLPModels
import PowerModels: PowerModels, silence

include("opf.jl")
include("luksanvlcek.jl")
include("distillation.jl")
include("quadrotor.jl")

const NAMES = filter(names(ExaModelsExamples; all = true)) do x
    str = string(x)
    endswith(str, "model") && !startswith(str, "#")
end

for name in filter(names(ExaModelsExamples; all=true)) do x
    endswith(string(x), "model")
end
    @eval export $name
end

function __init__()
    global TMPDIR = tempname()
    mkdir(TMPDIR)
    PowerModels.silence()
end

end # module ExaModelsExamples
