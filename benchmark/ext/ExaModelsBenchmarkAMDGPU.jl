module ExaModelsBenchmarkAMDGPU

import ExaModelsBenchmark
import AMDGPU

function __init__()
    push!(
        ExaModelsBenchmark.DEVICES,
        (
            name = "AMDGPU",
            hardware = "$(AMDGPU.HIP.name(AMDGPU.device()))",
            backend = AMDGPU.ROCBackend()
        )
    )
end

end
