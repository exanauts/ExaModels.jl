# Convert array to the appropriate device using Adapt.jl.
convert_array(v, ::Nothing) = v
convert_array(v, backend) = adapt(backend, v)

# Recursively replace Float64 with Float32 in scalars, containers, and the
# fields of arbitrary structs. Used by backend extensions (Metal, oneAPI) that
# either reject Float64 outright or perform poorly on it. Type-stable via
# multiple dispatch on leaf types and Val(fieldcount(T)) on the generic struct
# path.
replace_float_64(x::Float64) = Float32(x)
replace_float_64(x::Tuple) = map(replace_float_64, x)
replace_float_64(x::NamedTuple) = map(replace_float_64, x)
replace_float_64(x::AbstractArray{Float64}) = Float32.(x)
replace_float_64(x::AbstractArray) = replace_float_64.(x)
@inline replace_float_64(x::T) where {T} = _rebuild_float_32(x, Val(fieldcount(T)))
@inline _rebuild_float_32(x, ::Val{0}) = x
@inline function _rebuild_float_32(x::T, ::Val{N}) where {T, N}
    ConstructionBase.constructorof(T)(
        ntuple(i -> replace_float_64(getfield(x, i)), Val(N))...
    )
end

# to avoid type privacy
sort!(array; kwargs...) = Base.sort!(array; kwargs...)

# MOI
function Optimizer end
