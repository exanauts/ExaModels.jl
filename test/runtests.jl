using Test, ExaModels

include("NLPTest/NLPTest.jl")

@testset "ExaModels test" begin
    @testset "NLP test" begin
        NLPTest.runtests()
    end
end
