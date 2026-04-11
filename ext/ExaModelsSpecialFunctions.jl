module ExaModelsSpecialFunctions

using ExaModels, SpecialFunctions

# Type-generic constant helpers (avoid Float64 literals for Float32 compatibility)
@inline _cinvsqrtpi(x) = oftype(x, 1 / sqrt(π))
@inline _csqrtpihalf(x) = oftype(x, sqrt(π) / 2)
@inline _cpi(x) = oftype(x, π)

include("functionlist.jl")

end # module ExaModelsSpecialFunctions
