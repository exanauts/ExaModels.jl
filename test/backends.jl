const BACKENDS = Any[nothing, CPU(), POCLBackend()]

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
    if !(Sys.iswindows() && OpenCL.cl.is_high_integrity_level())
        push!(BACKENDS, OpenCLBackend())
        @info "including PoCL"
        OpenCL.versioninfo()
        @info "OpenCL Device:" OpenCL.cl.device()
    else
        @info "excluding PoCL (cannot use pocl_jll when running on Windows as administrator)"
    end
else
    @info "excluding PoCL"
end
