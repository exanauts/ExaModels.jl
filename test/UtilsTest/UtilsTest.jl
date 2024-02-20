module UtilsTest

using Test
import ExaModels, NLPModelsIpopt
import ..NLPTest: _exa_luksan_vlcek_model

UTIL_MODELS = [ExaModels.TimedNLPModel, ExaModels.CompressedNLPModel]

FIELDS = [:solution, :multipliers, :multipliers_L, :multipliers_U, :objective]

function runtests()
    @testset "Utils tests" begin
        m, ~ = _exa_luksan_vlcek_model(nothing, 3)

        result = NLPModelsIpopt.ipopt(m; print_level = 0)

        for util_model in UTIL_MODELS
            util_result = NLPModelsIpopt.ipopt(util_model(m); print_level = 0)

            @testset "$util_model" begin
                for field in FIELDS
                    @testset "$field" begin
                        @test getfield(util_result, field) ≈ getfield(result, field) atol =
                            1e-6
                    end
                end
            end
        end
    end
end

end
