module ExaModelsCUDA

import ExaModels: ExaModels, NLPModels
import CUDA: CUDA, CUDABackend, CuArray

ExaModels.ExaCore(backend::CUDABackend) = ExaModels.ExaCore(Float64, backend)
ExaModels.ExaCore(T, backend::CUDABackend) = ExaModels.ExaCore(x0 = CUDA.zeros(T,0), backend = backend)
ExaModels.convert_array(v, backend::CUDABackend) = CuArray(v)

end
