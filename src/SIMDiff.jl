module SIMDiff

import NLPModels

include("graph.jl")
include("dual.jl")
include("triple.jl")
include("register.jl")
include("functionlist.jl")
include("function.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")
include("helper.jl")
include("nlp.jl")

export data, variable, objective, constraint, constraint!, WrapperModel

end # module SIMDiffes
