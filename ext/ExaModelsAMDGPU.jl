module ExaModelsAMDGPU

import ExaModels, AMDGPU

ExaModels.convert_array(v, backend::AMDGPU.ROCBackend) = AMDGPU.ROCArray(v)

# Below are type piracy
function Base.findall(f::F, bitarray::A) where {F<:Function,A<:AMDGPU.ROCVector}
    a = Array(bitarray)
    b = findall(f, a)
    c = similar(bitarray, eltype(b), length(b))

    return copyto!(c, b)
end
Base.findall(bitarray::A) where {A<:AMDGPU.ROCVector} = Base.findall(identity, bitarray)
end
