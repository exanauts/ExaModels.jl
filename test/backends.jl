import Pkg
const BACKENDS = []

if haskey(ARGS, "EXAMODELS_NO_TEST_CPU")
    @info "excluding CPU"
else
    @eval push!(BACKENDS, nothing)
end

if haskey(ARGS, "EXAMODELS_TEST_KA")
    Pkg.activate(joinpath(@__DIR__, "test-ka"))
    @eval using KernelAbstractions
    @eval push!(BACKENDS, CPU())
    @info "including CPU"
else
    @info "excluding CPU"
end

if haskey(ARGS, "EXAMODELS_TEST_CUDA")
    Pkg.activate(joinpath(@__DIR__, "test-cuda"))
    @eval using CUDA
    @eval push!(BACKENDS, CUDABackend())
    @info "including CUDA"
else
    @info "excluding CUDA"
end

if haskey(ARGS, "EXAMODELS_TEST_AMDGPU")
    Pkg.activate(joinpath(@__DIR__, "test-amdgpu"))
    @eval using AMDGPU
    @eval push!(BACKENDS, ROCBackend())
    @info "including AMDGPU"
else
    @info "excluding AMDGPU"
end

if haskey(ARGS, "EXAMODELS_TEST_ONEAPI")
    Pkg.activate(joinpath(@__DIR__, "test-oneapi"))
    @eval using oneAPI
    @eval push!(BACKENDS, oneAPIBackend())
    @info "including oneAPI"
else
    @info "excluding oneAPI"
end

if haskey(ARGS, "EXAMODELS_TEST_METAL") 
    Pkg.activate(joinpath(@__DIR__, "test-metal"))
    @eval using Metal
    @eval push!(BACKENDS, MetalBackend())
    @info "including Metal"
else
    @info "excluding Metal"
end

if haskey(ARGS, "EXAMODELS_TEST_POCL")
    Pkg.activate(joinpath(@__DIR__, "test-opencl"))
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
