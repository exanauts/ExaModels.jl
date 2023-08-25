module ExaModelsOneAPI

import ExaModels, oneAPI

ExaModels.ExaCore(backend::oneAPI.oneAPIBackend) = ExaModels.ExaCore(Float64, backend)
ExaModels.ExaCore(T, backend::oneAPI.oneAPIBackend) =
    ExaModels.ExaCore(x0 = oneAPI.zeros(T, 0), backend = backend)

function ExaModels.myappend!(a::A, b::Base.Generator, lb) where {A<:oneAPI.oneVector}
    la = length(a)
    a = similar(a, la + lb)
    map!(b.f, view(a, (la+1):(la+lb)), b.iter)
    return a
end

function ExaModels.myappend!(a::A, b::A, lb) where {A<:oneAPI.oneVector}
    la = length(a)
    a = similar(a, la + lb)
    copyto!(view(a, (la+1):(la+lb)), b)
    return a
end


function ExaModels.myappend!(a::A, b::Number, lb) where {A<:oneAPI.oneVector}
    la = length(a)
    a = similar(a, la + lb)
    fill!(view(a, (la+1):(la+lb)), b)
    return a
end

ExaModels.convert_array(v, backend::oneAPI.oneAPIBackend) = oneAPI.oneArray(v)

# Below are type piracy
function Base.sum(a::A) where {A<:oneAPI.oneVector{Float64}}
    b = zeros(length(a))
    copyto!(b, a)
    return sum(b)
end
function Base.findall(f::F, bitarray::A) where {F<:Function,A<:oneAPI.oneVector}
    a = Array(bitarray)
    b = findall(f, a)
    c = similar(bitarray, eltype(b), length(b))
    return copyto!(c, b)
end
Base.findall(bitarray::A) where {A<:oneAPI.oneVector} = Base.findall(identity, bitarray)
Base.sort!(array::A; lt = isless) where {A<:oneAPI.oneVector} =
    copyto!(array, sort!(Array(array)))
end
