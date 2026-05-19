module ExaModelsMetal

import ExaModels, Metal

ExaModels.sort!(array::A; lt = isless) where {A<:Metal.MtlArray} =
    copyto!(array, sort!(Array(array); lt = lt))

ExaModels.convert_array(v::Metal.MtlArray, ::Metal.MetalBackend) = v
ExaModels.convert_array(v, ::Metal.MetalBackend) =
    Metal.MtlArray(ExaModels.replace_float_64.(v))

ExaModels.default_T(::Metal.MetalBackend) = Float32

end # module
