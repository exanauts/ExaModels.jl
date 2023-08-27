module ExaModelsAMDGPU

import ExaModels, AMDGPU

ExaModels.ExaCore(backend::AMDGPU.ROCBackend) = ExaModels.ExaCore(Float64, backend)
ExaModels.ExaCore(T, backend::AMDGPU.ROCBackend) =
    ExaModels.ExaCore(x0 = AMDGPU.zeros(T, 0), backend = backend)

ExaModels.convert_array(v, backend::AMDGPU.ROCBackend) = AMDGPU.ROCArray(v)

# Below are type piracy
function ExaModels.findall(f::F, bitarray::A) where {F<:Function,A<:AMDGPU.ROCVector}
    a = Array(bitarray)
    b = findall(f, a)
    c = similar(bitarray, eltype(b), length(b))
    return copyto!(c, b)
end
ExaModels.findall(bitarray::A) where {A<:AMDGPU.ROCVector} = ExaModels.findall(identity, bitarray)
ExaModels.sort!(array::A; lt = isless) where {A<:AMDGPU.ROCVector} =
    copyto!(array, sort!(Array(array)))
end
