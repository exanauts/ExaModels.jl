# Generate GPU speedup table from benchmark results
using CSV, DataFrames, Printf

cpu = CSV.read("results/shin-compute-000_CPU_fp64_COPS_LV.csv", DataFrame)
cuda = CSV.read("results/shin-compute-000_CUDA-Quadro_GV100_fp64_COPS_LV.csv", DataFrame)
amdgpu = CSV.read("results/shin-compute-002_AMDGPU-AMD_Radeon_VII_fp64_COPS_LV.csv", DataFrame)

# Join on suite, problem, size
cpu.key = cpu.suite .* "/" .* cpu.problem .* "/" .* string.(cpu.size)
cuda.key = cuda.suite .* "/" .* cuda.problem .* "/" .* string.(cuda.size)
amdgpu.key = amdgpu.suite .* "/" .* amdgpu.problem .* "/" .* string.(amdgpu.size)

callbacks = [:tgrad, :tjac, :thess]
cb_names = ["grad!", "jac_coord!", "hess_coord!"]

# SGM function
function sgm(vals; shift=1e-5)
    n = length(vals)
    n == 0 && return NaN
    return exp(sum(log.(vals .+ shift)) / n) - shift
end

function classify_size(nnzj, nnzh)
    nnz = max(nnzj, nnzh)
    nnz < 1_000 ? "Small" : (nnz < 100_000 ? "Medium" : "Large")
end

cpu.sizeclass = [classify_size(r.nnzj, r.nnzh) for r in eachrow(cpu)]

println("=" ^ 80)
println("GPU Speedup Summary (CPU / GPU)")
println("=" ^ 80)

for (suite_name, suite_label) in [("LV", "Lukšan-Vlček"), ("COPS", "COPS")]
    println("\n--- $suite_label ---")
    cpu_s = filter(r -> r.suite == suite_name, cpu)
    cuda_s = filter(r -> r.suite == suite_name, cuda)
    amdgpu_s = filter(r -> r.suite == suite_name, amdgpu)

    for sc in ["Small", "Medium", "Large"]
        cpu_sc = filter(r -> r.sizeclass == sc, cpu_s)
        cuda_sc = filter(r -> r.key in cpu_sc.key, cuda_s)
        amdgpu_sc = filter(r -> r.key in cpu_sc.key, amdgpu_s)

        print(@sprintf("  %-8s", sc))
        for (ci, cb) in enumerate(callbacks)
            cpu_val = sgm(cpu_sc[!, cb])
            cuda_val = sgm(cuda_sc[!, cb])
            amdgpu_val = sgm(amdgpu_sc[!, cb])
            cuda_sp = cpu_val / cuda_val
            amdgpu_sp = cpu_val / amdgpu_val
            print(@sprintf("  %s: CUDA %.1fx  AMD %.1fx", cb_names[ci], cuda_sp, amdgpu_sp))
        end
        println()
    end
end

# Generate LaTeX table for GPU section
println("\n\n" * "=" ^ 80)
println("LaTeX table")
println("=" ^ 80)

println(raw"""
\begin{table}[t]
\centering
\caption{GPU speedup over single-threaded CPU for AD callbacks (shifted geometric mean). All runs use Float64.}
\label{tab:gpu}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll rrr rrr@{}}
  \toprule
  & & \multicolumn{3}{c}{\textbf{CUDA} (Quadro GV100)} & \multicolumn{3}{c@{}}{\textbf{AMDGPU} (Radeon VII)} \\
  \cmidrule(lr){3-5} \cmidrule(l){6-8}
  suite & size & grad & jac & hess & grad & jac & hess \\
  \midrule""")

for suite_name in ["LV", "COPS"]
    cpu_s = filter(r -> r.suite == suite_name, cpu)
    cuda_s = filter(r -> r.suite == suite_name, cuda)
    amdgpu_s = filter(r -> r.suite == suite_name, amdgpu)

    first_row = true
    for sc in ["Small", "Medium", "Large"]
        cpu_sc = filter(r -> r.sizeclass == sc, cpu_s)
        cuda_sc = filter(r -> r.key in cpu_sc.key, cuda_s)
        amdgpu_sc = filter(r -> r.key in cpu_sc.key, amdgpu_s)

        label = first_row ? suite_name : ""
        first_row = false

        vals = String[]
        for cb in callbacks
            cpu_val = sgm(cpu_sc[!, cb])
            cuda_val = sgm(cuda_sc[!, cb])
            push!(vals, @sprintf("\$%.1f\\times\$", cpu_val / cuda_val))
        end
        for cb in callbacks
            cpu_val = sgm(cpu_sc[!, cb])
            amdgpu_val = sgm(amdgpu_sc[!, cb])
            push!(vals, @sprintf("\$%.1f\\times\$", cpu_val / amdgpu_val))
        end

        println("  $label & $sc & $(join(vals, " & ")) \\\\")
    end
    suite_name == "LV" && println("  \\midrule")
end

println(raw"""  \bottomrule
\end{tabular*}
\end{table}""")
