module ExaModelsJuMP

import ExaModels
import JuMP

function ExaModels.ExaModel(jm::JuMP.GenericModel{T}; options...) where {T}
    return ExaModels.ExaModel(JuMP.backend(jm); options...)
end

end # module ExaModelsJuMP
