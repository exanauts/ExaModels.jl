module NLPTest

using ExaModels, Test, ADNLPModels, NLPModels, KernelAbstractions, CUDA, AMDGPU, oneAPI
using NLPModelsIpopt, MadNLP, Percival


const BACKENDS = Any[nothing, CPU()]

const NLP_TEST_ARGUMENTS = [("luksan_vlcek", 3), ("luksan_vlcek", 20)]

const SOLVERS = [
    ("ipopt", nlp -> ipopt(nlp; print_level = 0)),
    ("madnlp", nlp -> madnlp(nlp; print_level = MadNLP.ERROR)),
    ("percival", nlp -> percival(nlp)),
]

const EXCLUDE = []

if CUDA.has_cuda()
    push!(BACKENDS, CUDABackend())
end

if AMDGPU.has_rocm_gpu()
    push!(BACKENDS, ROCBackend())
end

try
    oneAPI.oneL0.zeInit(0)
    push!(BACKENDS, oneAPIBackend())
    push!(EXCLUDE, ("percival", oneAPIBackend()))
catch e
end

include("luksan.jl")

function test_nlp(simdiff_model, adnlp_model, solver, backend, args)

    m1 = WrapperNLPModel(simdiff_model(backend, args...))
    m2 = WrapperNLPModel(adnlp_model(backend, args...))

    result1 = solver(m1)
    result2 = solver(m2)

    @test result1.status == result2.status

    for field in [:solution, :multipliers, :multipliers_L, :multipliers_U]
        @test getfield(result1, field) ≈ getfield(result2, field) atol = 1e-6
    end
end

function runtests()
    @testset "NLP tests" begin
        for (sname, solver) in SOLVERS
            for (name, args) in NLP_TEST_ARGUMENTS
                for backend in BACKENDS
                    if (sname, backend) in EXCLUDE
                        continue
                    end
                    simdiff_model = getfield(@__MODULE__, Symbol(name * "_simdiff_model"))
                    adnlp_model = getfield(@__MODULE__, Symbol(name * "_adnlp_model"))

                    @testset "$sname $name $args $backend" begin
                        test_nlp(simdiff_model, adnlp_model, solver, backend, args)
                    end
                end
            end
        end
    end
end

end # NLPTest
