const BACKENDS = Any[nothing, CPU()]

try
    CUDA.zeros(1)
    push!(BACKENDS, CUDABackend())
    @info "including CUDA"
catch e
    @info "excluding CUDA"
end

try
    AMDGPU.zeros(1)
    push!(BACKENDS, ROCBackend())
    @info "including AMDGPU"
catch e
    @info "excluding AMDGPU"
end

try
    oneAPI.zeros(1)
    push!(BACKENDS, oneAPIBackend())
    @info "including oneAPI"
catch e
    @info "excluding oneAPI"
end
