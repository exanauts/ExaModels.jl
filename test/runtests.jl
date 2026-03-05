using Test, ExaModels
using Random
using KernelAbstractions

Random.seed!(0)

include("backends.jl")
include("NLPTest/NLPTest.jl")
include("ADTest/ADTest.jl")
include("JuMPTest/JuMPTest.jl")
include("UtilsTest/UtilsTest.jl")
include("TwoStageTest/TwoStageTest.jl")
include("LinAlgTest/LinAlgTest.jl")

@testset verbose = true "ExaModels test" begin
    @info "Running AD Test"
    ADTest.runtests()

    @info "Running NLP Test"
    NLPTest.runtests()

    @info "Running JuMP Test"
    JuMPTest.runtests()

    @info "Running Utils Test"
    UtilsTest.runtests()

    @info "Running TwoStage Test"
    TwoStageTest.runtests()

    @info "Running LinAlg Test"
    LinAlgTest.runtests()
end
