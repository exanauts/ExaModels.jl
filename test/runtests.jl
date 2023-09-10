using Test, ExaModels
using Random
Random.seed!(0)

include("NLPTest/NLPTest.jl")
include("ADTest/ADTest.jl")

@testset "ExaModels test" begin
    ADTest.runtests()
    NLPTest.runtests()
end
