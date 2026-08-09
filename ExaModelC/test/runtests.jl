using Test, ExaModels, ExaModelC
import JuliaC

include("aot_libraries.jl")

@testset "ExaModelC" begin
    aot_library_tests()
end
