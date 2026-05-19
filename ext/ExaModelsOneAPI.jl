module ExaModelsOneAPI

import ExaModels, oneAPI

ExaModels.convert_array(v::oneAPI.oneArray, ::oneAPI.oneAPIBackend) = v
ExaModels.convert_array(v, ::oneAPI.oneAPIBackend) =
    oneAPI.oneArray(ExaModels.replace_float_64.(v))

ExaModels.default_T(::oneAPI.oneAPIBackend) = Float32

if pkgversion(oneAPI) < v"2.6"

    ExaModels.sort!(array::A; lt = isless) where {A<:oneAPI.oneArray} =
        copyto!(array, sort!(Array(array); lt = lt))

    # below is type piracy
    function Base.findall(f::F, bitarray::A) where {F<:Function,A<:oneAPI.oneArray}
        a = Array(bitarray)
        b = findall(f, a)
        c = similar(bitarray, eltype(b), length(b))
        return copyto!(c, b)
    end
    Base.findall(bitarray::A) where {A<:oneAPI.oneArray} = Base.findall(identity, bitarray)
end

end # module
