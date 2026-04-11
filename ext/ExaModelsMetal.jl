module ExaModelsMetal

import ExaModels, Metal

ExaModels.sort!(array::A; lt = isless) where {A<:Metal.MtlArray} =
    copyto!(array, sort!(Array(array); lt = lt))

replace_float_64(a::NamedTuple{names,T}) where {names, T} = NamedTuple{names}(replace_float_64.(Tuple(a)))
replace_float_64(a::Tuple) = replace_float_64.(a)
replace_float_64(x::Float64) = Float32(x)
replace_float_64(x) = x

ExaModels.convert_array(v, ::Metal.MetalBackend) = ExaModels.adapt(Metal.MetalBackend(), replace_float_64.(v))

ExaModels.default_T(::Metal.MetalBackend) = Float32

end # module

