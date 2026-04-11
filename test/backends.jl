const BACKENDS = []

if "EXAMODELS_NO_TEST_CPU" in ARGS
    @info "excluding CPU"
else
    push!(BACKENDS, nothing)
end

if "EXAMODELS_TEST_KA" in ARGS
    @eval using KernelAbstractions
    @eval push!(BACKENDS, CPU())
    @info "including KernelAbstractions"
else
    @info "excluding KernelAbstractions"
end

if "EXAMODELS_TEST_CUDA" in ARGS
    @eval using CUDA
    @eval push!(BACKENDS, CUDABackend())
    @info "including CUDA"
else
    @info "excluding CUDA"
end

if "EXAMODELS_TEST_AMDGPU" in ARGS
    @eval using AMDGPU
    @eval push!(BACKENDS, ROCBackend())
    @info "including AMDGPU"
else
    @info "excluding AMDGPU"
end

if "EXAMODELS_TEST_ONEAPI" in ARGS
    @eval using oneAPI
    @eval push!(BACKENDS, oneAPIBackend())
    @info "including oneAPI"
else
    @info "excluding oneAPI"
end

if "EXAMODELS_TEST_METAL" in ARGS
    @eval using Metal
    @eval push!(BACKENDS, MetalBackend())
    @info "including Metal"
else
    @info "excluding Metal"
end

if "EXAMODELS_TEST_POCL" in ARGS
    @eval begin
        using OpenCL, pocl_jll
        if !(Sys.iswindows() && OpenCL.cl.is_high_integrity_level())
            push!(BACKENDS, OpenCLBackend())
            @info "including PoCL"
            OpenCL.versioninfo()
            @info "OpenCL Device:" OpenCL.cl.device()
        else
            @info "excluding PoCL (cannot use pocl_jll when running on Windows as administrator)"
        end
    end
else
    @info "excluding PoCL"
end
