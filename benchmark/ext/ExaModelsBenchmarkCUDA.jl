module ExaModelsBenchmarkCUDA

import ExaModelsBenchmark
import CUDA

function __init__()
    push!(
        ExaModelsBenchmark.DEVICES,
        (
            name = "CUDA",
            hardware = "$(CUDA.name(CUDA.device()))",
            backend = CUDA.CUDABackend(),
        ),
    )
end

end
