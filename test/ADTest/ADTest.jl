module ADTest

using ExaModels
using Random, Test, ForwardDiff
using KernelAbstractions, CUDA, AMDGPU, oneAPI

Random.seed!(0)

const BACKENDS = Any[nothing, CPU()]

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


const FUNCTIONS = [
    ("function-test-1-1",x->beta(erf(x[1]/x[2]/3.0)+3.0*x[2],erf(x[9])^2)), 
    ("function-test-1-2",x->0*x[1]), 
    ("function-test-1-3",x->beta(cos(log(inv(inv(x[1])))),erfc(tanh(0*x[1])))), 
    ("function-test-1-4",x->(0*x[1]^x[3]^1.0+x[1])/x[9]/x[10]), 
    ("function-test-1-5",x->(x[1]+1.)^x[2]*log(x[3])/tanh(x[2])), 
    ("function-test-1-6",x->beta(2*logbeta(x[1],x[5]),beta(x[2],x[3]))), 
    ("function-test-1-7",x->besselj0(exp(erf(-x[1])))), 
    ("function-test-1-8",x->erfc((x[1]^2/x[2])^x[9]/x[10])), 
    ("function-test-1-9",x->erfc(x[1])^erf(2.5x[2])), 
    ("function-test-1-10",x->sin(1/x[1])), 
    ("function-test-1-13",x->exp(x[2])/cos(x[1])^2+sin(x[1]^2)), 
    ("function-test-1-14",x->airyai(exp(x[1]+x[2]*2.0^8))), 
    ("function-test-1-16",x->sin(x[9]inv(x[1])-x[8]inv(x[2]))), 
    ("function-test-1-19",x->x[1]/log(x[2]^2+9.)), 
    ("function-test-1-21",x->beta(beta(tan(beta(x[1],1)+2.0),cos(sin(x[2]))),x[3])), 
    ("function-test-1-24",x->beta(cos(beta(beta(x[1]^9,x[2]),x[2]*x[3])),sin(x[2]*x[3]/2.0)/1.0)),
]

function gradient(f, x)
    y = similar(y)
end

function sgradient(f, x)
end

function sjacobian(f, x)
end

function shessian(f, x)
end

function runtests()
    @testset "NLP test" begin
        for (name, f) in FUNCTIONS
            x0 = randn(10)
            @testset "$name" begin
                @test gradient(f, x0) ≈ ForwardDiff.gradient(f, x0) atol=1e-6
            end
        end
    end
end

end #module
