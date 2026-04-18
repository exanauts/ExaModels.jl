module TwoStageTest

using Test
using ExaModels
import NLPModels
import NLPModelsIpopt: ipopt

function runtests()
    @testset "TwoStageExaCore" begin

        @testset "Construction and type" begin
            core = TwoStageExaCore(concrete = Val(true))
            @test core isa TwoStageExaCore
            @test core isa ExaCore
        end

        @testset "Variable and constraint scenario tags" begin
            ns, nv, nd = 3, 2, 1

            core = TwoStageExaCore(concrete = Val(true))
            @add_var(core, d; start = 0.0, scenario = 0)
            @add_var(core, v, ns, nv; start = 1.0, scenario = [s for s in 1:ns, j in 1:nv])

            # design var (1 var) + recourse vars (ns * nv vars)
            @test length(core.tags.var_scenario) == 1 + ns * nv
            @test core.tags.var_scenario[1] == 0  # design var tagged as scenario 0
            # recourse vars tagged with their scenario index
            for s in 1:ns, j in 1:nv
                idx = 1 + 1 + (s - 1) + ns * (j - 1)
                @test core.tags.var_scenario[idx] == s
            end

            @add_con(core, v[s, 1] - v[s, 2] for s in 1:ns; scenario = 1:ns)
            @test length(core.tags.con_scenario) == ns
            for s in 1:ns
                @test core.tags.con_scenario[s] == s
            end
        end

        @testset "Build and solve simple two-stage model" begin
            ns = 3
            nv = 2
            weight = 1.0 / ns

            core = TwoStageExaCore(concrete = Val(true))
            @add_var(core, d; start = 1.0, lvar = 0.0, uvar = Inf, scenario = 0)
            @add_var(core, v, ns, nv; start = 1.0, lvar = 0.0, uvar = Inf, scenario = [s for s in 1:ns, j in 1:nv])

            @add_con(core, v[s, 1] - v[s, 2]^2 for s in 1:ns; lcon = 0.0, scenario = 1:ns)

            @add_obj(core, d^2)
            @add_obj(core, weight * (v[s, i] - d)^2 for s in 1:ns, i in 1:nv)

            m = ExaModel(core)

            @test m.meta.nvar == 1 + ns * nv
            @test m.meta.ncon == ns

            result = ipopt(m; print_level = 0)
            @test result.status == :first_order
        end

        @testset "Design-only model (no recourse)" begin
            core = TwoStageExaCore(concrete = Val(true))
            @add_var(core, d, 3; start = 1.0, scenario = 0)
            @add_obj(core, (d[i] - 2)^2 for i in 1:3)

            m = ExaModel(core)
            @test m.meta.nvar == 3
            @test m.meta.ncon == 0

            result = ipopt(m; print_level = 0)
            @test result.status == :first_order
            @test solution(result, d) ≈ [2.0, 2.0, 2.0] atol = 1e-4
        end

    end
end

end # module TwoStageTest
