using ExaModelsExamples, KernelAbstractions, CUDA
using Downloads

neval = 3

powers = (
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case1354_pegase.m",
    "pglib_opf_case9241_pegase.m",
)

cases = [
    (
        "LV",
        jump_luksan_vlcek_model,
        ampl_luksan_vlcek_model,
        luksan_vlcek_model,
        (100, 1000, 10000)
    ),
    (
        "QR",
        jump_quadrotor_model,
        ampl_quadrotor_model,
        quadrotor_model,
        (50, 500, 5000)
    ),
    (
        "DC",
        jump_distillation_column_model,
        ampl_distillation_column_model,
        distillation_column_model,
        (5, 50, 500)
    ),
    (
        "PF",
        jump_ac_power_model,
        ampl_ac_power_model,
        ac_power_model,
        powers
    ),
]

# Obtain power system data
for power in powers
    if !isfile(power)
        Downloads.download(
            "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3/$power",
            power
        )
    end
end


save = []


try 
    GC.enable(false)
    for (name, jump_model, ampl_model, exa_model, args) in cases
        for (cnt, arg) in enumerate(args)
            
            println("$name$cnt")
            
            m = jump_model(arg)
            tj = ExaModelsExamples.benchmark_callbacks(m; N = neval)
            
            m = ampl_model(arg)
            ta = ExaModelsExamples.benchmark_callbacks(m; N = neval)

            m = exa_model(arg)
            te = ExaModelsExamples.benchmark_callbacks(m; N = neval)

            m = exa_model(arg, CPU())
            tec = ExaModelsExamples.benchmark_callbacks(m; N = neval)

            m = exa_model(arg, CUDABackend())
            teg = ExaModelsExamples.benchmark_callbacks(m; N = neval)

            push!(
                save, (
                    name = "$name$cnt",
                    nvar = m.meta.nvar,
                    ncon = m.meta.ncon,
                    tj = tj,
                    ta = ta,
                    te = te,
                    tec = tec,
                    teg = teg
                )
            )
        end
    end
catch e
    throw(e)
finally
    GC.enable(true)
end
