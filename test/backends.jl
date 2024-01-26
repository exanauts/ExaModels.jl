const BACKENDS = Any[nothing, CPU()]

if CUDA.has_cuda()
    push!(BACKENDS, CUDABackend())
    @info "including CUDA"
end

if AMDGPU.has_rocm_gpu()
    push!(BACKENDS, ROCBackend())
    @info "including AMDGPU"
end

try
    oneAPI.oneL0.zeInit(0)
    push!(BACKENDS, oneAPIBackend())
    push!(EXCLUDE2, ("percival", oneAPIBackend()))
    @info "including oneAPI"
catch e
end

