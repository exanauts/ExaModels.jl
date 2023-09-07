convert_array(v, ::Nothing) = v

# to avoid type privacy
sort!(array; kwargs...) = Base.sort!(array; kwargs...)
