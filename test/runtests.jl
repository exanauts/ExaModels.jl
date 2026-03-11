using Test, ExaModels
using Random
using KernelAbstractions

Random.seed!(0)

ad_tolerance(m1,m2) = max(ad_tolerance(m1), ad_tolerance(m2))
sol_tolerance(m1,m2) = max(sol_tolerance(m1), sol_tolerance(m2))
ad_tolerance(::Type{T}) where T = 10^(log(eps(T))/4)
sol_tolerance(::Type{T}) where T = 10^(log(eps(T))/8)
solver_tolerance(::Type{T}) where T = Float64(sqrt(eps(T))/100)

include("backends.jl")
include("NLPTest/NLPTest.jl")
include("ADTest/ADTest.jl")
include("JuMPTest/JuMPTest.jl")
include("UtilsTest/UtilsTest.jl")
include("TwoStageTest/TwoStageTest.jl")
include("LinAlgTest/LinAlgTest.jl")

@testset "ExaModels test" begin
    @info "Running AD Test"
    ADTest.runtests()

    @info "Running NLP Test"
    NLPTest.runtests()

    @info "Running JuMP Test"
    JuMPTest.runtests()

    @info "Running Utils Test"
    UtilsTest.runtests()

    # @info "Running TwoStage Test"
    # TwoStageTest.runtests()

    @info "Running LinAlg Test"
    LinAlgTest.runtests()
end
