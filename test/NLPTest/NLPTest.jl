module NLPTest

using ExaModels, Test, NLPModels, JuMP, NLPModelsJuMP, PowerModels, Downloads
using KernelAbstractions, CUDA, AMDGPU, oneAPI
using NLPModelsIpopt, MadNLP, Percival

const BACKENDS = Any[nothing, CPU()]

const NLP_TEST_ARGUMENTS = [
    ("luksan_vlcek", 3), ("luksan_vlcek", 20),
    ("ac_power", "pglib_opf_case3_lmbd.m"), ("ac_power", "pglib_opf_case14_ieee.m")
]

const SOLVERS = [
    ("ipopt", nlp -> ipopt(nlp; print_level = 0)),
    ("madnlp", nlp -> madnlp(nlp; print_level = MadNLP.ERROR)),
    ("percival", nlp -> percival(nlp)),
]

const EXCLUDE1 = [
    ("ac_power", "percival") # does not converge
]
const EXCLUDE2 = []

if CUDA.has_cuda()
    push!(BACKENDS, CUDABackend())
    @info "testing CUDA"
end

if AMDGPU.has_rocm_gpu()
    push!(BACKENDS, ROCBackend())
    @info "testing AMDGPU"
end

try
    oneAPI.oneL0.zeInit(0)
    push!(BACKENDS, oneAPIBackend())
    push!(EXCLUDE2, ("percival", oneAPIBackend()))
    @info "testing oneAPI"
catch e
end

include("luksan.jl")
include("power.jl")

function test_nlp(exa_model, jump_model, solver, backend, args)

    m1 = WrapperNLPModel(exa_model(backend, args))
    m2 = WrapperNLPModel(jump_model(backend, args))

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
                if (name, sname) in EXCLUDE1
                    continue
                end
                for backend in BACKENDS
                    if (sname, backend) in EXCLUDE2
                        continue
                    end
                    exa_model = getfield(@__MODULE__, Symbol("exa_$(name)_model"))
                    jump_model = getfield(@__MODULE__, Symbol("jump_$(name)_model"))

                    @testset "$sname $name $args $backend" begin
                        test_nlp(exa_model, jump_model, solver, backend, args)
                    end
                end
            end
        end
    end
end

function __init__()
    if haskey(ENV, "EXA_MODELS_DEPOT")
        global TMPDIR = ENV["EXA_MODELS_DEPOT"]
    else
        global TMPDIR = tempname()
        mkdir(TMPDIR)
    end
    PowerModels.silence()
end


end # NLPTest
