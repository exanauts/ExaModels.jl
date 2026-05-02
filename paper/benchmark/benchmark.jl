# ============================================================================
# AD Performance Benchmark: ExaModels vs JuMP vs AMPL
# ============================================================================
#
# Usage:
#   julia --project=. benchmark.jl reference           # JuMP + AMPL (CPU)
#   julia --project=. benchmark.jl                     # ExaModels CPU (nothing)
#   julia --project=. benchmark.jl CPU                 # ExaModels KA CPU
#   julia --project=. benchmark.jl CUDA                # ExaModels CUDA
#   julia --project=. benchmark.jl AMDGPU              # ExaModels AMD GPU
#   julia --project=. benchmark.jl oneAPI              # ExaModels Intel GPU
#
# Outputs: results/<hostname>_<tag>.csv

using BenchmarkTools
using CSV
using DataFrames
using NLPModels
using ExaModels
using NLPModelsJuMP
using JuMP
using KernelAbstractions

using AmplNLWriter
using AmplNLReader
import MathOptInterface as MOI

if "minimal" in ARGS
    include("cases_minimal.jl")
elseif "quick" in ARGS
    include("cases_quick.jl")
else
    include("cases.jl")
end

# ============================================================================
# Hardware detection
# ============================================================================

function hardware_info(; device_name = "CPU")
    info = Dict{String,String}()
    info["hostname"]     = gethostname()
    info["julia"]        = string(VERSION)
    info["examodels"]    = string(pkgversion(ExaModels))
    info["os"]           = string(Sys.KERNEL, " ", Sys.MACHINE)
    info["cpu"]          = Sys.cpu_info()[1].model
    info["cpu_cores"]    = string(length(Sys.cpu_info()))
    info["total_memory"] = string(round(Sys.total_memory() / 2^30; digits = 1), " GiB")
    info["threads"]      = string(Threads.nthreads())
    info["device"]       = device_name
    return info
end

# ============================================================================
# Device setup
# ============================================================================

function setup_device(arg)
    if arg == "CPU"
        backend = KernelAbstractions.CPU()
        device_name = "CPU-$(Threads.nthreads())T"
        return backend, device_name
    elseif arg == "CUDA"
        @eval using CUDA
        backend = @eval CUDA.CUDABackend()
        device_name = @eval "CUDA-$(CUDA.name(CUDA.device()))"
        return backend, device_name
    elseif arg == "AMDGPU"
        @eval using AMDGPU
        backend = @eval AMDGPU.ROCBackend()
        device_name = @eval "AMDGPU-$(AMDGPU.HIP.name(AMDGPU.device()))"
        return backend, device_name
    elseif arg == "oneAPI"
        @eval using oneAPI
        backend = @eval oneAPI.oneAPIBackend()
        device_name = @eval "oneAPI-$(oneAPI.properties(oneAPI.device()).name)"
        return backend, device_name
    elseif arg == "Metal"
        @eval using Metal
        backend = @eval Metal.MetalBackend()
        device_name = @eval "Metal-$(Metal.current_device().name)"
        return backend, device_name
    else
        return nothing, "CPU"
    end
end

# ============================================================================
# Model builders — ExaModels
# ============================================================================

function build_examodels_lv(model_func, args...; backend = nothing, T = Float64)
    if backend === nothing
        return model_func(LuksanVlcekBenchmark.ExaModelsBackend(), args...; T = T, prod = true)
    else
        return model_func(LuksanVlcekBenchmark.ExaModelsBackend(), args...; T = T, backend = backend, prod = true)
    end
end

function build_examodels_cops(model_func, args...; backend = nothing, T = Float64)
    if backend === nothing
        return model_func(COPSBenchmark.ExaModelsBackend(), args...; T = T, prod = true)
    else
        return model_func(COPSBenchmark.ExaModelsBackend(), args...; T = T, backend = backend, prod = true)
    end
end

function build_examodels_opf(filename; backend = nothing, form = :polar, T = Float64)
    m, _, _ = ExaModelsPower.ac_opf_model(filename; backend = backend, form = form, T = T)
    return m
end

# ============================================================================
# Model builders — JuMP
# ============================================================================

function build_jump_lv(model_func, args...)
    jm = model_func(LuksanVlcekBenchmark.JuMPBackend(), args...)
    return NLPModelsJuMP.MathOptNLPModel(jm)
end

function build_jump_cops(model_func, args...)
    jm = model_func(COPSBenchmark.JuMPBackend(), args...)
    return NLPModelsJuMP.MathOptNLPModel(jm)
end

# ============================================================================
# Model builders — AMPL (via JuMP → AmplNLWriter → .nl → AmplNLReader)
# ============================================================================

const AMPL_TMPDIR = mktempdir()

function build_ampl_lv(model_func, args...)
    nlfile = joinpath(AMPL_TMPDIR, "lv_$(hash(args)).nl")
    jm = model_func(LuksanVlcekBenchmark.JuMPBackend(), args...)
    MOI.write_to_file(JuMP.backend(jm), nlfile)
    return AmplNLReader.AmplModel(nlfile)
end

function build_ampl_cops(model_func, args...)
    nlfile = joinpath(AMPL_TMPDIR, "cops_$(hash(args)).nl")
    jm = model_func(COPSBenchmark.JuMPBackend(), args...)
    MOI.write_to_file(JuMP.backend(jm), nlfile)
    return AmplNLReader.AmplModel(nlfile)
end

# ============================================================================
# Project x into [l + ε, u − ε]
# ============================================================================

import NLPModels: get_nvar, get_ncon, get_nnzj, get_nnzh, get_x0, get_lvar, get_uvar,
                  obj, cons!, grad!, jac_coord!, hess_coord!,
                  hprod!, jprod!, jtprod!

function project!(x, l, u; margin = 1e-4)
    map!(x, l, x, u) do li, xi, ui
        max(li + margin, min(ui - margin, xi))
    end
end

# ============================================================================
# Benchmark a single NLPModel (AD callbacks)
# ============================================================================

function btime(f; seconds = 0.5)
    f()  # warmup
    GC.gc()
    # Calibrate N from a single timed call
    t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e9
    N = max(3, min(10_000, round(Int, seconds / max(dt, 1e-9))))
    return minimum(begin
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end for _ = 1:N)
end

function benchmark_model(m; seconds = 0.5)
    nvar = get_nvar(m)
    ncon = get_ncon(m)
    nnzj = get_nnzj(m)
    nnzh = get_nnzh(m)

    x  = copy(get_x0(m))
    y  = similar(x, ncon); fill!(y, one(eltype(x)))
    c  = similar(x, ncon)
    g  = similar(x, nvar)
    jv = similar(x, nnzj)
    hv = similar(x, nnzh)
    v  = similar(x, nvar); fill!(v, one(eltype(x)))
    Hv = similar(x, nvar)
    Jv = similar(x, ncon)
    Jtv = similar(x, nvar)

    project!(x, get_lvar(m), get_uvar(m))

    # Benchmark each callback independently (calibrates N per-callback)
    tobj   = btime(() -> obj(m, x); seconds = seconds)
    tcon   = ncon > 0 ? btime(() -> cons!(m, x, c); seconds = seconds) : 0.0
    tgrad  = btime(() -> grad!(m, x, g); seconds = seconds)
    tjac   = ncon > 0 ? btime(() -> jac_coord!(m, x, jv); seconds = seconds) : 0.0
    thess  = btime(() -> hess_coord!(m, x, y, hv); seconds = seconds)
    thprod = btime(() -> hprod!(m, x, v, Hv); seconds = seconds)
    tjprod  = ncon > 0 ? btime(() -> jprod!(m, x, v, Jv); seconds = seconds) : 0.0
    tjtprod = ncon > 0 ? btime(() -> jtprod!(m, x, y, Jtv); seconds = seconds) : 0.0

    return (
        nvar = nvar, ncon = ncon, nnzj = nnzj, nnzh = nnzh,
        tobj = tobj, tcon = tcon, tgrad = tgrad, tjac = tjac, thess = thess,
        thprod = thprod, tjprod = tjprod, tjtprod = tjtprod,
    )
end

# ============================================================================
# Benchmark model creation
# ============================================================================

function benchmark_creation(builder, args...; seconds = 0.5, kwargs...)
    return btime(() -> builder(args...; kwargs...); seconds = seconds)
end

# ============================================================================
# Result row helper
# ============================================================================

function make_result_df()
    return DataFrame(
        suite   = String[],
        problem = String[],
        size    = String[],
        ams     = String[],
        nvar    = Int[],
        ncon    = Int[],
        nnzj    = Int[],
        nnzh    = Int[],
        tobj    = Float64[],
        tcon    = Float64[],
        tgrad   = Float64[],
        tjac    = Float64[],
        thess   = Float64[],
        thprod  = Float64[],
        tjprod  = Float64[],
        tjtprod = Float64[],
        tcreate = Float64[],
    )
end

function push_result!(rows, suite, problem, sz, ams, r, tc)
    push!(rows, (suite, problem, string(sz), ams,
                 r.nvar, r.ncon, r.nnzj, r.nnzh,
                 r.tobj, r.tcon, r.tgrad, r.tjac, r.thess,
                 r.thprod, r.tjprod, r.tjtprod, tc))
end

# ============================================================================
# Run ExaModels benchmarks (any backend)
# ============================================================================

function run_examodels(; backend = nothing, seconds = 0.5, suites = nothing, T = Float64)
    rows = make_result_df()
    run_suite(s) = suites === nothing || s in suites

    # LV
    if run_suite("LV")
        for case in LV_CASES
            for sz in case.sizes
                args = sz isa Tuple ? sz : (sz,)
                label = "$(case.name)/$(sz)"
                @info "LV ExaModels: $label"; flush(stderr)
                try
                    m  = build_examodels_lv(case.model, args...; backend = backend, T = T)
                    tc = benchmark_creation(build_examodels_lv, case.model, args...; backend = backend, seconds = seconds)
                    r  = benchmark_model(m; seconds = seconds)
                    push_result!(rows, "LV", case.name, sz, "ExaModels", r, tc)
                catch e
                    @warn "Failed: LV ExaModels $label" exception=(e, catch_backtrace())
                end
                GC.gc()
            end
        end
    end

    # COPS
    if run_suite("COPS")
        for case in COPS_CASES
            for sz in case.sizes
                args = sz isa Tuple ? sz : (sz,)
                label = "$(case.name)/$(sz)"
                @info "COPS ExaModels: $label"; flush(stderr)
                try
                    m  = build_examodels_cops(case.model, args...; backend = backend, T = T)
                    tc = benchmark_creation(build_examodels_cops, case.model, args...; backend = backend, seconds = seconds)
                    r  = benchmark_model(m; seconds = seconds)
                    push_result!(rows, "COPS", case.name, sz, "ExaModels", r, tc)
                catch e
                    @warn "Failed: COPS ExaModels $label" exception=(e, catch_backtrace())
                end
                GC.gc()
            end
        end
    end

    # OPF (ACP + ACR)
    if run_suite("OPF")
        for form in OPF_FORMS
            for filename in OPF_CASES
                @info "OPF ExaModels ($form): $filename"; flush(stderr)
                try
                    m  = build_examodels_opf(filename; backend = backend, form = form, T = T)
                    tc = benchmark_creation(build_examodels_opf, filename; backend = backend, form = form, seconds = seconds)
                    r  = benchmark_model(m; seconds = seconds)
                    push_result!(rows, "OPF-$form", filename, filename, "ExaModels", r, tc)
                catch e
                    @warn "Failed: OPF ExaModels ($form) $filename" exception=(e, catch_backtrace())
                end
                GC.gc()
            end
        end
    end

    return rows
end

# ============================================================================
# Run reference benchmarks (JuMP + AMPL, CPU only)
# ============================================================================

function run_reference(; seconds = 0.5)
    rows = make_result_df()

    # LV — JuMP
    for case in LV_CASES
        for sz in case.sizes
            args = sz isa Tuple ? sz : (sz,)
            label = "$(case.name)/$(sz)"

            @info "LV JuMP: $label"; flush(stderr)
            try
                m  = build_jump_lv(case.model, args...)
                tc = benchmark_creation(build_jump_lv, case.model, args...; seconds = seconds)
                r  = benchmark_model(m; seconds = seconds)
                push_result!(rows, "LV", case.name, sz, "JuMP", r, tc)
            catch e
                @warn "Failed: LV JuMP $label" exception=(e, catch_backtrace())
            end
            GC.gc()
        end
    end

    # LV — AMPL (via AmplNLWriter)
    for case in LV_CASES
        for sz in case.sizes
            args = sz isa Tuple ? sz : (sz,)
            label = "$(case.name)/$(sz)"

            @info "LV AMPL: $label"; flush(stderr)
            try
                m  = build_ampl_lv(case.model, args...)
                tc = benchmark_creation(build_ampl_lv, case.model, args...; seconds = seconds)
                r  = benchmark_model(m; seconds = seconds)
                push_result!(rows, "LV", case.name, sz, "AMPL", r, tc)
                finalize(m)
            catch e
                @warn "Failed: LV AMPL $label" exception=(e, catch_backtrace())
            end
            GC.gc()
        end
    end

    # COPS — JuMP
    for case in COPS_CASES
        for sz in case.sizes
            args = sz isa Tuple ? sz : (sz,)
            label = "$(case.name)/$(sz)"

            @info "COPS JuMP: $label"; flush(stderr)
            try
                m  = build_jump_cops(case.model, args...)
                tc = benchmark_creation(build_jump_cops, case.model, args...; seconds = seconds)
                r  = benchmark_model(m; seconds = seconds)
                push_result!(rows, "COPS", case.name, sz, "JuMP", r, tc)
            catch e
                @warn "Failed: COPS JuMP $label" exception=(e, catch_backtrace())
            end
            GC.gc()
        end
    end

    # COPS — AMPL (via AmplNLWriter)
    for case in COPS_CASES
        for sz in case.sizes
            args = sz isa Tuple ? sz : (sz,)
            label = "$(case.name)/$(sz)"

            @info "COPS AMPL: $label"; flush(stderr)
            try
                m  = build_ampl_cops(case.model, args...)
                tc = benchmark_creation(build_ampl_cops, case.model, args...; seconds = seconds)
                r  = benchmark_model(m; seconds = seconds)
                push_result!(rows, "COPS", case.name, sz, "AMPL", r, tc)
                finalize(m)
            catch e
                @warn "Failed: COPS AMPL $label" exception=(e, catch_backtrace())
            end
            GC.gc()
        end
    end

    return rows
end

# ============================================================================
# Save results with hardware info as CSV header comments
# ============================================================================

function save_results(rows, hw_info, tag)
    mkpath("results")
    host = hw_info["hostname"]
    fname = joinpath("results", "$(host)_$(tag).csv")

    # Write hardware info as separate file
    open(joinpath("results", "$(host)_$(tag)_hw.txt"), "w") do io
        for (k, v) in sort(collect(hw_info))
            println(io, "$k: $v")
        end
    end

    # Write CSV
    CSV.write(fname, rows)

    @info "Results saved to $fname ($(nrow(rows)) rows)"
    return fname
end

# ============================================================================
# Entry point
# ============================================================================

function main()
    args = filter(a -> a ∉ ("quick", "minimal", "fp32"), ARGS)
    T = "fp32" in ARGS ? Float32 : Float64
    mode = length(args) >= 1 ? args[1] : "nothing"
    seconds = length(args) >= 2 ? parse(Float64, args[2]) : 2.0

    # Suite filter: pass suite names after seconds, e.g. `benchmark.jl nothing 0.5 OPF`
    suite_args = length(args) >= 3 ? args[3:end] : nothing

    precision_tag = T == Float32 ? "fp32" : "fp64"

    if mode == "reference"
        hw = hardware_info(; device_name = "CPU-reference")
        @info "Running reference benchmarks (JuMP + AMPL)" seconds=seconds; flush(stderr)
        rows = run_reference(; seconds = seconds)
        save_results(rows, hw, "reference")
    else
        backend, device_name = setup_device(mode)
        hw = hardware_info(; device_name = device_name)
        @info "Running ExaModels benchmarks" device=device_name T=T seconds=seconds suites=suite_args; flush(stderr)
        rows = run_examodels(; backend = backend, seconds = seconds, suites = suite_args, T = T)
        tag = replace(device_name, " " => "_", "/" => "_") * "_" * precision_tag
        if suite_args !== nothing
            tag *= "_" * join(suite_args, "_")
        end
        save_results(rows, hw, tag)
    end
end

main()
