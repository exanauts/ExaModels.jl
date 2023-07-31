module ExaModelsCUDA

using ExaModels: ExaModels, NLPModels
using CUDA

ExaModels.Core(backend::CUDABackend) = ExaModels.Core(Float64, backend)
ExaModels.Core(T, backend::CUDABackend) = ExaModels.Core(x0 = CUDA.zeros(T,0), backend = backend)
ExaModels.convert_array(v, backend::CUDABackend) = CuArray(v)

end
