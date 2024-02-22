module ExaModelsJuMP

import ExaModels
import JuMP

function ExaModels.ExaModel(jm::JuMP.GenericModel{T}; backend = nothing) where {T}
    return ExaModels.ExaModel(jm.moi_backend.model_cache; backend = backend)
end

end # module ExaModelsJuMP
