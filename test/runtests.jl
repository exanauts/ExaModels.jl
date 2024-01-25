using Test, ExaModels
using Random
using KernelAbstractions, CUDA, AMDGPU, oneAPI

Random.seed!(0)

include("backends.jl")
include("NLPTest/NLPTest.jl")
include("ADTest/ADTest.jl")
include("JuMPTest/JuMPTest.jl")

@testset "ExaModels test" begin
    # ADTest.runtests()
    # NLPTest.runtests()
    JuMPTest.runtests()
end
