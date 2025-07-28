const BACKENDS = Any[nothing, CPU()]

if haskey(ENV, "EXAMODELS_TEST_CUDA")
    using CUDA
    push!(BACKENDS, CUDABackend())
    @info "including CUDA"
else
    @info "excluding CUDA"
end

if haskey(ENV, "EXAMODELS_TEST_AMDGPU")
    using AMDGPU
    push!(BACKENDS, ROCBackend())
    @info "including AMDGPU"
else
    @info "excluding AMDGPU"
end

if haskey(ENV, "EXAMODELS_TEST_ONEAPI")
    using oneAPI
    push!(BACKENDS, oneAPIBackend())
    @info "including oneAPI"
else
    @info "excluding oneAPI"
end

if haskey(ENV, "EXAMODELS_TEST_POCL")
    using OpenCL, pocl_jll
    push!(BACKENDS, OpenCLBackend())
    @info "including PoCL"
    OpenCL.versioninfo()
else
    @info "excluding PoCL"
end