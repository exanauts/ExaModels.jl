module NLPTest

using SIMDiff, Test, ADNLPModels, NLPModels, NLPModelsIpopt, KernelAbstractions, CUDA

const NLP_TEST_ARGUMENTS = [
    (
        "luksan_vlcek",
        3
    ),
    (
        "luksan_vlcek",
        20
    ),
]

const BACKENDS = Any[
    nothing,
    CPU()
]

if CUDA.has_cuda()
    push!(BACKENDS, CUDABackend())
end

include("utils.jl")
include("luksan.jl")

function test_nlp(simdiff_model, adnlp_model, backend, args)
    
    m1 = WrapperNLPModel(simdiff_model(backend,args...))
    m2 = WrapperNLPModel(adnlp_model(backend,args...))
    
    result1 = ipopt(m1; print_level = 0)
    result2 = ipopt(m2; print_level = 0)

    @test result1.status == result2.status
    
    for field in [
        :solution,
        :multipliers,
        :multipliers_L,
        :multipliers_U
        ]
        @test getfield(result1, field) ≈ getfield(result2, field) atol=1e-6
    end
end

function runtests()
    @testset "NLP tests" begin
        for (name, args) in NLP_TEST_ARGUMENTS
            for backend in BACKENDS
                simdiff_model = getfield(
                    @__MODULE__,
                    Symbol(name * "_simdiff_model")
                )
                adnlp_model = getfield(
                    @__MODULE__,
                    Symbol(name * "_adnlp_model")
                )

                @testset "$name $args $backend" begin
                    test_nlp(simdiff_model, adnlp_model, backend, args)
                end
            end
        end
    end
end

end # NLPTest

