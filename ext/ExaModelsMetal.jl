module ExaModelsMetal

import ExaModels, Metal

ExaModels.sort!(array::A; lt = isless) where {A<:Metal.MtlArray} =
    copyto!(array, sort!(Array(array); lt = lt))

replace_float_64(a::NamedTuple{names,T}) where {names, T} = NamedTuple{names}(replace_float_64.(Tuple(a)))
replace_float_64(a::Tuple) = replace_float_64.(a)
replace_float_64(x::Float64) = Float32(x)
replace_float_64(x::T) where {T} = _replace_struct_float64(Val(isstructtype(T) && !isprimitivetype(T) && fieldcount(T) > 0), x)
_replace_struct_float64(::Val{false}, x) = x
function _replace_struct_float64(::Val{true}, x::T) where T
    vals = ntuple(i -> replace_float_64(getfield(x, i)), fieldcount(T))
    W = T.name.wrapper
    try
        return W{Float32}(vals...)
    catch
        try
            return W(vals...)
        catch
            return x
        end
    end
end

ExaModels.convert_array(v::Metal.MtlArray, ::Metal.MetalBackend) = v
ExaModels.convert_array(v, ::Metal.MetalBackend) = Metal.MtlArray(replace_float_64.(v))

ExaModels.default_T(::Metal.MetalBackend) = Float32

end # module

