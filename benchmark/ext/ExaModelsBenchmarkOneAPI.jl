module ExaModelsBenchmarkOneAPI

import ExaModelsBenchmark
import oneAPI

function __init__()
    push!(
        ExaModelsBenchmark.DEVICES,
        (
            name = "oneAPI",
            hardware = "$(oneAPI.properties(oneAPI.device()).name)",
            backend = oneAPI.oneAPIBackend()
        )
    )
end

end
