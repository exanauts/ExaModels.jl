module ExaModelsOneAPI

import ExaModels, oneAPI

ExaModels.convert_array(v, backend::oneAPI.oneAPIBackend) = oneAPI.oneArray(v)

ExaModels.sort!(array::A; lt = isless) where {A<:oneAPI.oneArray} =
    copyto!(array, sort!(Array(array); lt = lt))

function ExaModels.findall(f::F, bitarray::A) where {F<:Function,A<:oneAPI.oneArray}
     a = Array(bitarray)
     b = findall(f, a)
     c = similar(bitarray, eltype(b), length(b))
     return copyto!(c, b)
 end
ExaModels.findall(bitarray::A) where {A<:oneAPI.oneArray} = findall(identity, bitarray)

end # module
