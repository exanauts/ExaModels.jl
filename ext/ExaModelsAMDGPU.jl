module ExaModelsAMDGPU

import ExaModels, AMDGPU

ExaModels.ExaCore(backend::AMDGPU.ROCBackend) = ExaModels.ExaCore(Float64, backend)
ExaModels.ExaCore(T, backend::AMDGPU.ROCBackend) =
    ExaModels.ExaCore(x0 = AMDGPU.zeros(T, 0), backend = backend)

# function ExaModels.myappend!(a::A, b::Base.Generator, lb) where {A<:AMDGPU.ROCVector}
#     la = length(a)
#     a = similar(a, la + lb)
#     map!(b.f, view(a, (la+1):(la+lb)), b.iter)
#     return a
# end

# function ExaModels.myappend!(a::A, b::A, lb) where {A<:AMDGPU.ROCVector}
#     la = length(a)
#     a = similar(a, la + lb)
#     copyto!(view(a, (la+1):(la+lb)), b)
#     return a
# end


# function ExaModels.myappend!(a::A, b::Number, lb) where {A<:AMDGPU.ROCVector}
#     la = length(a)
#     a = similar(a, la + lb)
#     fill!(view(a, (la+1):(la+lb)), b)
#     return a
# end

ExaModels.convert_array(v, backend::AMDGPU.ROCBackend) = AMDGPU.ROCArray(v)

# Below are type piracy
function Base.findall(f::F, bitarray::A) where {F<:Function,A<:AMDGPU.ROCVector}
    a = Array(bitarray)
    b = findall(f, a)
    c = similar(bitarray, eltype(b), length(b))
    return copyto!(c, b)
end
Base.findall(bitarray::A) where {A<:AMDGPU.ROCVector} = Base.findall(identity, bitarray)
Base.sort!(array::A; lt = isless) where {A<:AMDGPU.ROCVector} =
    copyto!(array, sort!(Array(array)))
end
