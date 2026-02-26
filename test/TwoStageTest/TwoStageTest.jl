module TwoStageTest

using Test
using ExaModels
import NLPModels
import NLPModels: obj, grad!

function runtests()
    @testset "TwoStageTags" begin

        @testset "Basic construction and tag tracking" begin
            c = TwoStageExaCore(; ns = 3)

            # Design variable: scenario = false
            d = variable(c; start = 0.0, scenario = false)
            @test c.tags.var_scenario == Bool[false]

            # Recourse variables: scenario = true
            v = variable(c, 6; start = 0.0, scenario = true)
            @test c.tags.var_scenario == Bool[false, true, true, true, true, true, true]

            @test c.tags.ns == 3
        end

        @testset "Scalar scenario broadcasting" begin
            c = TwoStageExaCore(; ns = 2)

            # Scalar false broadcasts to all elements
            variable(c, 3; scenario = false)
            @test c.tags.var_scenario == Bool[false, false, false]

            # Scalar true broadcasts to all elements
            variable(c, 4; scenario = true)
            @test c.tags.var_scenario == Bool[false, false, false, true, true, true, true]
        end

        @testset "Tag/element count validation" begin
            c = TwoStageExaCore(; ns = 2)

            # Correct length should work
            variable(c, 3; scenario = [true, false, true])
            @test c.tags.var_scenario == Bool[true, false, true]

            # Wrong length should throw
            @test_throws DimensionMismatch variable(c, 3; scenario = [true, false])

            # Same for constraints
            c2 = TwoStageExaCore(; ns = 2)
            x = variable(c2, 2)
            constraint(c2, x[1] + x[2] for _ in 1:3; scenario = [true, false, true])
            @test c2.tags.con_scenario == Bool[true, false, true]

            @test_throws DimensionMismatch constraint(
                c2, x[1] + x[2] for _ in 1:2; scenario = [true]
            )
        end

        @testset "Constraint tags" begin
            c = TwoStageExaCore(; ns = 2)
            x = variable(c, 4)

            # Design constraint
            constraint(c, x[1] + x[2] for _ in 1:2; scenario = false)
            @test c.tags.con_scenario == Bool[false, false]

            # Recourse constraint
            constraint(c, x[3] + x[4] for _ in 1:3; scenario = true)
            @test c.tags.con_scenario == Bool[false, false, true, true, true]
        end

        @testset "subexpr forwards scenario tags" begin
            c = TwoStageExaCore(; ns = 2)
            x = variable(c, 4; scenario = true)

            # Lifted subexpr should forward scenario tag to auxiliary vars and constraints
            s = subexpr(c, x[i]^2 for i in 1:4; scenario = true)

            # 4 original vars + 4 auxiliary vars from subexpr, all with scenario = true
            @test length(c.tags.var_scenario) == 8
            @test all(c.tags.var_scenario)

            # 4 defining constraints from subexpr, all with scenario = true
            @test length(c.tags.con_scenario) == 4
            @test all(c.tags.con_scenario)
        end

        @testset "set_parameter! on ExaModel" begin
            c = ExaCore()
            p = parameter(c, [1.0, 2.0])
            x = variable(c, 2)
            objective(c, p[i] * x[i]^2 for i in 1:2)
            m = ExaModel(c)

            # Initial: obj at x=[1,1] = 1*1 + 2*1 = 3
            @test obj(m, [1.0, 1.0]) ≈ 3.0

            # Update parameters
            set_parameter!(m, p, [10.0, 20.0])

            # After: obj at x=[1,1] = 10*1 + 20*1 = 30
            @test obj(m, [1.0, 1.0]) ≈ 30.0
        end

        @testset "set_parameter! size mismatch on ExaModel" begin
            c = ExaCore()
            p = parameter(c, [1.0, 2.0])
            x = variable(c, 2)
            objective(c, p[i] * x[i]^2 for i in 1:2)
            m = ExaModel(c)

            @test_throws DimensionMismatch set_parameter!(m, p, [1.0, 2.0, 3.0])
        end

        @testset "Full two-stage model construction" begin
            ns = 3
            nv = 2

            c = TwoStageExaCore(; ns = ns)

            # Design variable
            d = variable(c; start = 1.0, scenario = false)

            # Recourse variables
            v = variable(c, ns, nv; start = 1.0, scenario = true)

            # Parameters for each scenario
            p = parameter(c, Float64[1, 2, 3])

            # Recourse constraints
            constraint(c, v[s, 1] + v[s, 2] - d for s in 1:ns; lcon = 0.0, scenario = true)

            # Objective
            objective(c, d^2)
            objective(c, (v[s, j] - p[s])^2 for s in 1:ns, j in 1:nv)

            m = ExaModel(c)

            # Verify tag structure
            @test m.tags.ns == 3
            # 1 design + 6 recourse
            @test length(m.tags.var_scenario) == 7
            @test m.tags.var_scenario[1] == false  # design
            @test all(m.tags.var_scenario[2:7])     # recourse

            # 3 constraints, all recourse
            @test length(m.tags.con_scenario) == 3
            @test all(m.tags.con_scenario)

            # Verify model dimensions
            @test NLPModels.get_nvar(m) == 7
            @test NLPModels.get_ncon(m) == 3
        end

        @testset "Integer to Bool conversion" begin
            c = TwoStageExaCore(; ns = 2)

            # 0 and 1 should convert to false/true
            variable(c, 2; scenario = [0, 1])
            @test c.tags.var_scenario == Bool[false, true]

            # Scalar 0/1 also works
            variable(c; scenario = 0)
            @test c.tags.var_scenario[end] == false
            variable(c; scenario = 1)
            @test c.tags.var_scenario[end] == true
        end

    end
end

end # module
