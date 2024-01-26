module UtilsTest

import ExaModels, NLPModelsIpopt
import ..NLPTest: _exa_luksan_vlcek_model

UTIL_MODELS = [
    ExaModels.TimedNLPModel,
    ExaModels.CompressedNLPModel
]

FIELDS = [
    :solution,
    :multipliers,
    :multipliers_L,
    :multipliers_U,
    :objective
]

m = _exa_luksan_vlcek_model(backend,3)

result = ipopt(m)

for util_model in UTIL_MODELS
    util_result = ipopt(util_model(m))

    @testset "$util_model"
    for field in FIELDS
        @testset "$field" begin
            @test getfield(util_result, field) ≈ getfield(result, field) atol = 1e-6
        end
    end
end

end
