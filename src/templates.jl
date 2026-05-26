# Convert array to the appropriate device using Adapt.jl.
convert_array(v, ::Nothing) = v
convert_array(v, backend) = adapt(backend, v)

# to avoid type privacy
sort!(array; kwargs...) = Base.sort!(array; kwargs...)

# MOI
function Optimizer end
