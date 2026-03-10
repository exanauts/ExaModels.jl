module ExaModelsMetal

import ExaModels, Metal

ExaModels.sort!(array::A; lt = isless) where {A<:Metal.MtlArray} =
    copyto!(array, sort!(Array(array); lt = lt))

replace_float_64(a::Tuple) = replace_float_64.(a)
replace_float_64(x::Float64) = Float32(x)
replace_float_64(x) = x

ExaModels.convert_array(v, backend::Metal.MetalBackend) =  ExaModels.adapt(backend, replace_float_64.(v))

end # module

