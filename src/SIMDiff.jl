module SIMDiff

import NLPModels

include("graph.jl")
include("register.jl")
include("functionlist.jl")
include("simdfunction.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")
include("nlp.jl")
include("templates.jl")

export data, variable, objective, constraint, constraint!

end # module SIMDiffes
