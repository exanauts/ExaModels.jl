module ExaModelsAMDGPU

import ExaModels, AMDGPU

ExaModels.convert_array(v, backend::AMDGPU.ROCBackend) = AMDGPU.ROCArray(v)

end
