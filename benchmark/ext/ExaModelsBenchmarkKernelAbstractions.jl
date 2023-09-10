module ExaModelsBenchmarkKernelAbstractions

import ExaModelsBenchmark
import KernelAbstractions
import CpuId

function __init__()
    push!(
        ExaModelsBenchmark.DEVICES,
        (
            name = "$(Threads.nthreads())T",
            hardware = "$(CpuId.cpubrand())",
            backend = KernelAbstractions.CPU(),
        ),
    )
end

end
