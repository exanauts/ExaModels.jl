module ExaModelsMetal

import ExaModels
import Metal: Metal, MtlVector, MetalBackend, mtl

ExaModels.ExaCore(backend::MetalBackend) = ExaModels.ExaCore(Float32, backend)
ExaModels.ExaCore(T, backend::MetalBackend) = ExaModels.ExaCore(x0 = Metal.zeros(T, 0), backend = backend)
ExaModels.convert_array(v, backend::MetalBackend) = mtl(v)

function ExaModels.myappend!(a::A, b::Base.Generator, lb) where {A<:MtlVector}
    la = length(a)
    a = similar(a, la + lb)
    map!(b.f, view(a, (la+1):(la+lb)), b.iter)
    return a
end
function ExaModels.myappend!(a::A, b::A, lb) where {A<:MtlVector}
    la = length(a)
    a = similar(a, la + lb)
    copyto!(view(a, (la+1):(la+lb)), b)
    return a
end
function ExaModels.myappend!(a::A, b::Number, lb) where {A<:MtlVector}
    la = length(a)
    a = similar(a, la + lb)
    fill!(view(a, (la+1):(la+lb)), b)
    return a
end

# Below are type piracy
function Base.findall(f::F, bitarray::A) where {F<:Function,A<:MtlVector}
    a = Array(bitarray)
    b = findall(f, a)
    c = similar(bitarray, eltype(b), length(b))
    return copyto!(c, b)
end
Base.findall(bitarray::A) where {A<:MtlVector} = Base.findall(identity, bitarray)
Base.sort!(array::A; lt = isless) where {A<:MtlVector} =
    copyto!(array, sort!(Array(array)))

end


end    
