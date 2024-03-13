module ExaModelsCUDA

import ExaModels: ExaModels, NLPModels
import CUDA: CUDA, CUDABackend, CuArray

ExaModels.ExaCore(backend::CUDABackend) = ExaModels.ExaCore(Float64, backend)
ExaModels.convert_array(v::Base.Iterators.ProductIterator, backend::CUDABackend) =
    Base.product((ExaModels.convert_array(i, backend) for i in v.iterators)...)
ExaModels.convert_array(v::UnitRange, backend::CUDABackend) = v
ExaModels.convert_array(v, backend::CUDABackend) = CuArray(v)

end
