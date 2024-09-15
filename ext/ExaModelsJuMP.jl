module ExaModelsJuMP

import ExaModels
import JuMP

function ExaModels.ExaModel(jm::JuMP.GenericModel{T}; options...) where {T}
    return ExaModels.ExaModel(jm.moi_backend.model_cache; options...)
end

end # module ExaModelsJuMP
