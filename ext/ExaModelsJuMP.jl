module ExaModelsJuMP

import ExaModels
import JuMP

function ExaModels.ExaModel(jm::JuMP.GenericModel{T}; options...) where {T}
    # FIXME: what is user passes `T` under `options`?
    return ExaModels.ExaModel(JuMP.backend(jm); T=T, options...)
end

end # module ExaModelsJuMP
