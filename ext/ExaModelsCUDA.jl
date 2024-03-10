module ExaModelsCUDA

import ExaModels: ExaModels, NLPModels
import CUDA: CUDA, CUDABackend, CuArray

ExaModels.ExaCore(backend::CUDABackend) = ExaModels.ExaCore(Float64, backend)
ExaModels.convert_array(v, backend::CUDABackend) = CuArray(v)

end
