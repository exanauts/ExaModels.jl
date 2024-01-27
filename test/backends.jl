const BACKENDS = Any[nothing, CPU()]

if CUDA.has_cuda()
    push!(BACKENDS, CUDABackend())
    @info "including CUDA"
else
    @info "excluding CUDA"
end

if AMDGPU.has_rocm_gpu()
    push!(BACKENDS, ROCBackend())
    @info "including AMDGPU"
else
    @info "excluding AMDGPU"
end

try
    oneAPI.oneL0.zeInit(0)
    push!(BACKENDS, oneAPIBackend())
    @info "including oneAPI"
catch e
    @info "excluding oneAPI"
end

