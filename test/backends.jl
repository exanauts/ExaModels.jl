const BACKENDS = Any[nothing, CPU()]

is_package_installed(name::String) = !isnothing(Base.find_package(name))
const EXAMODELS_TEST_CUDA = is_package_installed("CUDA")
if EXAMODELS_TEST_CUDA
    @eval using CUDA
    @eval push!(BACKENDS, CUDABackend())
    @info "including CUDA"
else
    @info "excluding CUDA"
end

const EXAMODELS_TEST_AMDGPU = is_package_installed("AMDGPU")
if EXAMODELS_TEST_AMDGPU
    @eval using AMDGPU
    @eval push!(BACKENDS, ROCBackend())
    @info "including AMDGPU"
else
    @info "excluding AMDGPU"
end

const EXAMODELS_TEST_ONEAPI = is_package_installed("oneAPI")
if EXAMODELS_TEST_ONEAPI
    @eval using oneAPI
    @eval push!(BACKENDS, oneAPIBackend())
    @info "including oneAPI"
else
    @info "excluding oneAPI"
end

const EXAMODELS_TEST_OPENCL = is_package_installed("OpenCL")
if EXAMODELS_TEST_OPENCL
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
