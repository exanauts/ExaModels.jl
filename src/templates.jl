# A template for convert_array. This is extended in extension packages for each device architecture.
convert_array(v, ::Nothing) = v

# to avoid type privacy
sort!(array; kwargs...) = Base.sort!(array; kwargs...)

# MOI
function Optimizer end
function result_status_translator end
function termination_status_translator end
