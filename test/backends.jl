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

