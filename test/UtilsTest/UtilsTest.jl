module UtilsTest

using Test
import ExaModels, NLPModelsIpopt
import ..NLPTest: _exa_luksan_vlcek_model

UTIL_MODELS = [ExaModels.TimedNLPModel, ExaModels.CompressedNLPModel]

FIELDS = [:solution, :multipliers, :multipliers_L, :multipliers_U, :objective]

# Test struct for replace_float_64 coverage
struct _RFTestBus{T}
    id::Int
    vmax::T
    vmin::T
    name::Symbol
end

struct _RFTestNetwork{T}
    baseMVA::T
    bus::Vector{_RFTestBus{T}}
    coefs::NTuple{3, T}
end

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

        @testset "replace_float_64" begin
            # Scalar leaf
            @test ExaModels.replace_float_64(1.5) === 1.5f0
            @test ExaModels.replace_float_64(1.5f0) === 1.5f0
            @test ExaModels.replace_float_64(3) === 3
            @test ExaModels.replace_float_64(:foo) === :foo

            # Tuple / NamedTuple recursion
            @test ExaModels.replace_float_64((1.0, 2, "x")) === (1.0f0, 2, "x")
            @test ExaModels.replace_float_64((a = 1.0, b = 2)) === (a = 1.0f0, b = 2)

            # Array of Float64
            @test ExaModels.replace_float_64([1.0, 2.0]) == Float32[1.0, 2.0]
            @test eltype(ExaModels.replace_float_64([1.0, 2.0])) === Float32

            # Mixed-eltype array routed elementwise
            arr = Any[1.0, 2]
            out = ExaModels.replace_float_64(arr)
            @test out[1] === 1.0f0 && out[2] === 2

            # Parametric struct rebuild
            b = _RFTestBus{Float64}(1, 1.1, 0.9, :b1)
            b32 = ExaModels.replace_float_64(b)
            @test b32 isa _RFTestBus{Float32}
            @test b32.id === 1
            @test b32.vmax === 1.1f0
            @test b32.name === :b1

            # Nested struct + array field
            net = _RFTestNetwork(100.0,
                [_RFTestBus{Float64}(1, 1.1, 0.9, :b1),
                 _RFTestBus{Float64}(2, 1.2, 0.8, :b2)],
                (0.1, 0.2, 0.3))
            net32 = ExaModels.replace_float_64(net)
            @test net32 isa _RFTestNetwork{Float32}
            @test net32.baseMVA === 100.0f0
            @test eltype(net32.bus) === _RFTestBus{Float32}
            @test net32.coefs === (0.1f0, 0.2f0, 0.3f0)

            # Identity on type with no Float64 anywhere
            @test ExaModels.replace_float_64((1, 2, 3)) === (1, 2, 3)
        end
    end
end

end
