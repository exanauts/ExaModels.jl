module ExaModelsOneAPI

import ExaModels, oneAPI

function ExaModels.append!(
    backend,
    a::A,
    b::Base.Generator{UnitRange{I}},
    lb,
) where {I,A<:oneAPI.oneVector}
    la = length(a)
    aa = similar(a, la + lb)
    copyto!(view(aa, 1:la), a)
    map!(b.f, view(aa, (la+1):(la+lb)), b.iter)
    return aa
end

function ExaModels.append!(backend, a::A, b::Base.Generator, lb) where {A<:oneAPI.oneVector}
    la = length(a)
    aa = similar(a, la + lb)
    copyto!(view(aa, 1:la), a)
    copyto!(view(aa, (la+1):(la+lb)), b)
    map!(b.f, view(aa, (la+1):(la+lb)), view(aa, (la+1):(la+lb)))
    return aa
end

function ExaModels.append!(
    backend,
    a::A,
    b::V,
    lb,
) where {A<:oneAPI.oneVector,V<:AbstractVector}
    la = length(a)
    aa = similar(a, la + lb)
    copyto!(view(aa, 1:la), a)
    copyto!(view(aa, (la+1):(la+lb)), b)
    return aa
end


function ExaModels.append!(backend, a::A, b::Number, lb) where {A<:oneAPI.oneVector}
    la = length(a)
    aa = similar(a, la + lb)
    copyto!(view(aa, 1:la), a)
    fill!(view(aa, (la+1):(la+lb)), b)
    return aa
end

ExaModels.convert_array(v, backend::oneAPI.oneAPIBackend) = oneAPI.oneArray(v)

ExaModels.sort!(array::A; lt = isless) where {A<:oneAPI.oneVector} =
    copyto!(array, sort!(Array(array); lt = lt))

# below is type piracy
function Base.findall(f::F, bitarray::A) where {F<:Function,A<:oneAPI.oneVector}
    a = Array(bitarray)
    b = findall(f, a)
    c = similar(bitarray, eltype(b), length(b))
    return copyto!(c, b)
end
Base.findall(bitarray::A) where {A<:oneAPI.oneVector} = Base.findall(identity, bitarray)

end # module
