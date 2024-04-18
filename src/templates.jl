# A template for convert_array. This is extended in extension packages for each device architecture.
convert_array(v, ::Nothing) = v

# template to avoid oneAPI sum issue
sum(a) = Base.sum(a)

# to avoid type privacy
sort!(array; kwargs...) = Base.sort!(array; kwargs...)

# MOI
function Optimizer end
function IpoptOptimizer end
function MadNLPOptimizer end
function result_status_translator end
function termination_status_translator end
