import Pkg
const BACKENDS = []

# Instantiate a sub-env and add it to LOAD_PATH so its packages
# remain resolvable after we switch back to the test/ project.
function _setup_subenv(name)
    dir = joinpath(@__DIR__, name)
    Pkg.activate(dir)
    Pkg.instantiate()
    push!(LOAD_PATH, dir)
end

if "EXAMODELS_NO_TEST_CPU" in ARGS
    @info "excluding CPU"
else
    push!(BACKENDS, nothing)
end

# Phase 1: instantiate sub-envs and add to LOAD_PATH
"EXAMODELS_TEST_KA" in ARGS && _setup_subenv("test-ka")
"EXAMODELS_TEST_CUDA" in ARGS && _setup_subenv("test-cuda")
"EXAMODELS_TEST_AMDGPU" in ARGS && _setup_subenv("test-amdgpu")
"EXAMODELS_TEST_ONEAPI" in ARGS && _setup_subenv("test-oneapi")
"EXAMODELS_TEST_METAL" in ARGS && _setup_subenv("test-metal")
"EXAMODELS_TEST_POCL" in ARGS && _setup_subenv("test-opencl")

# Switch back to the test/ project
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Phase 2: load backend packages (resolvable via LOAD_PATH)
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
