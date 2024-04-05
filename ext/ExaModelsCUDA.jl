module ExaModelsCUDA

import ExaModels: ExaModels, NLPModels
import CUDA: CUDA, CUDABackend, CuArray

ExaModels.convert_array(v, backend::CUDABackend) = CuArray(v)

end
