# A template for convert_array. This is extended in extension packages for each device architecture.
convert_array(v, ::Nothing) = v

# to avoid type privacy
sort!(array; kwargs...) = Base.sort!(array; kwargs...)



"""
    ExaModelsBackend([backend]) 

A examodel automatic differentiation backend for JuMP/MOI models.
"""
function ExaModelsBackend end


function Optimizer end
