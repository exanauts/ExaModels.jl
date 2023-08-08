module ExaModelsOneAPI

import ExaModels, oneAPI

ExaModels.ExaCore(backend::oneAPI.oneAPIBackend) = ExaModels.ExaCore(x0 = oneAPI.zeros(Float64,0), backend = backend)

function ExaModels.myappend!(a::A,b) where A <: oneAPI.oneVector
    la = length(a);
    lb = length(b);
    a = similar(a, la+lb);
    map!(b.f, view(a,(la+1):(la+lb)) , b.iter)
    return a
end

function Base.sum(a::A) where A <: oneAPI.oneVector{Float64}
    b = similar(a,Float32,length(a))
    copyto!(b,a)
    Float64(sum(b))
end

# oneAPI doesn't supportt findall
function Base.findall(f::F,bitarray::A) where {F <: Function, A <: oneAPI.oneVector}
    a = Array(bitarray)
    b = findall(f,a)
    c = similar(bitarray, eltype(b), length(b))
    return copyto!(c,b)
end
Base.findall(bitarray::A) where A <: oneAPI.oneVector = Base.findall(identity,bitarray)

Base.sort!(array::A; lt = isless) where A <: oneAPI.oneVector = copyto!(array,sort!(Array(array)))

end
