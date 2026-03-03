# Convert array to the appropriate device using Adapt.jl.
convert_array(v, ::Nothing) = v
convert_array(v, backend) = adapt(backend, v)

# to avoid type privacy
sort!(array; kwargs...) = Base.sort!(array; kwargs...)

# MOI
function Optimizer end
function IpoptOptimizer end
function MadNLPOptimizer end
function result_status_translator end
function termination_status_translator end
