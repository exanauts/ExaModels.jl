module TwoStageTest

using Test
using ExaModels
import NLPModels
import NLPModelsIpopt: ipopt

function runtests()
    @testset "TwoStageExaCore + EachScenario" begin

        @testset "Construction and dimensions" begin
            ns, nv, nd = 3, 2, 2
            nθ = 2
            θ_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)
            θ = @add_par(core, θ_vals)

            obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
            objective(
                core, θ[θ_off + 1] * v[v_idx]^2 + θ[θ_off + 2] * d[1] * v[v_idx]
                    for (i, j, v_idx, θ_off) in obj_data
            )

            con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_con(core, (v[v_idx] + d[1] for (i, j, v_idx) in con_data), EachScenario())

            model = ExaModel(core)

            @test NLPModels.get_nvar(model) == ns * nv + nd
            @test NLPModels.get_ncon(model) == ns * nv
            @test get_nscen(model) == ns
            @test count(==(0), get_var_scen(model)) == nd
            @test count(==(1), get_var_scen(model)) == nv
            @test count(==(1), get_con_scen(model)) == nv
        end

        @testset "Variable and constraint tagging" begin
            ns, nv, nd = 2, 3, 2

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)

            model = ExaModel(core)
            vtags = get_var_scen(model)

            @test findall(==(1), vtags) == [1, 2, 3]
            @test findall(==(2), vtags) == [4, 5, 6]
            @test findall(==(0), vtags) == [7, 8]

            x_global = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

            @test x_global[findall(==(1), vtags)] == [1.0, 2.0, 3.0]
            @test x_global[findall(==(2), vtags)] == [4.0, 5.0, 6.0]
            @test x_global[findall(==(0), vtags)] == [7.0, 8.0]
        end

        @testset "Objective evaluation" begin
            ns, nv, nd = 2, 2, 1
            θ_vals = [2.0, 3.0]

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)
            θ = @add_par(core, θ_vals)

            obj_data = [(i, j, (i - 1) * nv + j, i) for i in 1:ns for j in 1:nv]
            @add_obj(core, θ[θi] * v[v_idx]^2 for (i, j, v_idx, θi) in obj_data)

            con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_con(core, (v[v_idx] for (i, j, v_idx) in con_data), EachScenario())

            model = ExaModel(core)

            x_global = [1.0, 2.0, 3.0, 4.0, 0.5]
            total_obj = obj(model, x_global)
            @test total_obj ≈ 85.0
        end

        @testset "Constraint evaluation" begin
            ns, nv, nd = 2, 2, 1
            θ_vals = [1.0, 2.0]

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)
            θ = @add_par(core, θ_vals)

            obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_obj(core, v[v_idx]^2 for (i, j, v_idx) in obj_data)

            con_data = [(i, j, (i - 1) * nv + j, i) for i in 1:ns for j in 1:nv]
            @add_con(core, (v[v_idx] + d[1] - θ[θi] for (i, j, v_idx, θi) in con_data), EachScenario())

            model = ExaModel(core)

            x_global = [1.0, 2.0, 3.0, 4.0, 0.5]
            c_global = zeros(ns * nv)
            cons_nln!(model, x_global, c_global)

            @test c_global ≈ [0.5, 1.5, 1.5, 2.5]

            ctags = get_con_scen(model)
            @test c_global[findall(==(1), ctags)] ≈ [0.5, 1.5]
            @test c_global[findall(==(2), ctags)] ≈ [1.5, 2.5]
        end

        @testset "Gradient evaluation" begin
            ns, nv, nd = 2, 2, 1
            θ_vals = [2.0, 3.0]

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)
            θ = @add_par(core, θ_vals)

            obj_data = [(i, j, (i - 1) * nv + j, i) for i in 1:ns for j in 1:nv]
            @add_obj(core, θ[θi] * v[v_idx]^2 for (i, j, v_idx, θi) in obj_data)

            con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_con(core, (v[v_idx] for (i, j, v_idx) in con_data), EachScenario())

            model = ExaModel(core)

            x_global = [1.0, 2.0, 3.0, 4.0, 0.5]
            g_global = zeros(NLPModels.get_nvar(model))
            grad!(model, x_global, g_global)

            @test g_global ≈ [4.0, 8.0, 18.0, 24.0, 0.0]

            vtags = get_var_scen(model)
            @test g_global[findall(==(1), vtags)] ≈ [4.0, 8.0]
            @test g_global[findall(==(0), vtags)] ≈ [0.0]
        end

        @testset "Jacobian structure and evaluation" begin
            ns, nv, nd = 2, 2, 1

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)

            obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_obj(core, v[v_idx]^2 for (i, j, v_idx) in obj_data)

            con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_con(core, (v[v_idx] + d[1] for (i, j, v_idx) in con_data), EachScenario())

            model = ExaModel(core)

            nnzj = NLPModels.get_nnzj(model)
            @test nnzj > 0

            rows = zeros(Int, nnzj)
            cols = zeros(Int, nnzj)
            jac_structure!(model, rows, cols)

            @test all(r -> 1 <= r <= NLPModels.get_ncon(model), rows)
            @test all(c -> 1 <= c <= NLPModels.get_nvar(model), cols)

            x_global = ones(NLPModels.get_nvar(model))
            jac_vals = zeros(nnzj)
            jac_coord!(model, x_global, jac_vals)

            @test all(v -> v ≈ 1.0, jac_vals)
        end

        @testset "Hessian structure and evaluation" begin
            ns, nv, nd = 2, 2, 1
            θ_vals = [2.0, 3.0]

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)
            θ = @add_par(core, θ_vals)

            obj_data = [(i, j, (i - 1) * nv + j, i) for i in 1:ns for j in 1:nv]
            @add_obj(core, θ[θi] * v[v_idx]^2 for (i, j, v_idx, θi) in obj_data)

            con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_con(core, (v[v_idx] + d[1] for (i, j, v_idx) in con_data), EachScenario())

            model = ExaModel(core)

            nnzh = NLPModels.get_nnzh(model)
            @test nnzh > 0

            rows = zeros(Int, nnzh)
            cols = zeros(Int, nnzh)
            hess_structure!(model, rows, cols)

            nvar = NLPModels.get_nvar(model)
            @test all(r -> 1 <= r <= nvar, rows)
            @test all(c -> 1 <= c <= nvar, cols)

            x_global = ones(nvar)
            y_global = zeros(NLPModels.get_ncon(model))
            hess_vals = zeros(nnzh)
            hess_coord!(model, x_global, y_global, hess_vals)

            @test any(v -> v ≈ 4.0, hess_vals)
            @test any(v -> v ≈ 6.0, hess_vals)
        end

        @testset "Ipopt solver — inequality constraints" begin
            # min  d^2 + (1/ns) * Σ (v_i - θ_i)^2   s.t. v_i ≥ d
            # Solution: d* = θ̄/2, v* = d*
            ns, nv, nd = 3, 1, 1
            θ_vals = [2.0, 4.0, 6.0]
            weight = 1.0 / ns

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)
            θ = @add_par(core, θ_vals)

            @add_obj(core, d[1]^2)
            @add_obj(core, weight * (v[i] - θ[i])^2 for i in 1:ns)

            @add_con(core,
                (v[i] - d[1] for i in 1:ns),
                EachScenario();
                lcon = 0.0,
            )

            model = ExaModel(core)
            result = ipopt(model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution
            vtags = get_var_scen(model)
            θ_bar = sum(θ_vals) / ns
            d_expected = θ_bar / 2

            @test x_sol[findall(==(0), vtags)][1] ≈ d_expected atol = 1e-5
            for i in 1:ns
                @test x_sol[findall(==(i), vtags)][1] ≈ d_expected atol = 1e-5
            end
        end

        @testset "Ipopt solver — multiple recourse and design vars" begin
            # min d₁²+d₂² + Σ (v_ij - θ_ij)²   s.t. v_i1+v_i2 = d₁+d₂
            ns, nv, nd = 2, 2, 2
            nθ = 2
            θ_vals = [1.0, 3.0, 2.0, 2.0]

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario(); lvar = 0.0)
            d = @add_var(core, nd; lvar = 0.0)
            θ = @add_par(core, θ_vals)

            @add_obj(core, d[1]^2 + d[2]^2)
            obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ + j) for i in 1:ns for j in 1:nv]
            @add_obj(core, (v[v_idx] - θ[θ_idx])^2 for (i, j, v_idx, θ_idx) in obj_data)

            con_data = [(i, (i - 1) * nv + 1, (i - 1) * nv + 2) for i in 1:ns]
            @add_con(core,
                (v[v1] + v[v2] - d[1] - d[2] for (i, v1, v2) in con_data),
                EachScenario();
                lcon = 0.0, ucon = 0.0,
            )

            model = ExaModel(core)
            result = ipopt(model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution
            vtags = get_var_scen(model)
            d_sol  = x_sol[findall(==(0), vtags)]
            v1_sol = x_sol[findall(==(1), vtags)]
            v2_sol = x_sol[findall(==(2), vtags)]

            @test v1_sol[1] + v1_sol[2] ≈ d_sol[1] + d_sol[2] atol = 1e-5
            @test v2_sol[1] + v2_sol[2] ≈ d_sol[1] + d_sol[2] atol = 1e-5
        end

        @testset "Ipopt solver — equality d-v coupling" begin
            # min d² + Σ v_i²   s.t. 2v_i + d = θ_i
            # d* = Σθ / (4+ns)
            ns, nv, nd = 3, 1, 1
            θ_vals = [4.0, 6.0, 8.0]

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)
            θ = @add_par(core, θ_vals)

            @add_obj(core, d[1]^2)
            @add_obj(core, v[i]^2 for i in 1:ns)

            @add_con(core,
                (2 * v[i] + d[1] - θ[i] for i in 1:ns),
                EachScenario();
                lcon = 0.0, ucon = 0.0,
            )

            model = ExaModel(core)
            result = ipopt(model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution
            vtags = get_var_scen(model)
            d_expected = sum(θ_vals) / (4 + ns)

            @test x_sol[findall(==(0), vtags)][1] ≈ d_expected atol = 1e-5

            for (i, θ) in enumerate(θ_vals)
                v_expected = (θ - d_expected) / 2
                @test x_sol[findall(==(i), vtags)][1] ≈ v_expected atol = 1e-5
            end
        end

        @testset "Variable bounds" begin
            ns, nv, nd = 2, 2, 2

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario();
                start = [0.1, 0.2, 0.3, 0.4], lvar = 0.0, uvar = 10.0)
            d = @add_var(core, nd;
                start = [0.5, 0.8], lvar = 0.0, uvar = 1.0)

            obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_obj(core, v[v_idx]^2 + d[1]^2 + d[2]^2 for (i, j, v_idx) in obj_data)

            con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_con(core,
                (v[v_idx] + d[1] for (i, j, v_idx) in con_data),
                EachScenario();
                lcon = 0.0, ucon = 100.0,
            )

            model = ExaModel(core)
            vtags = get_var_scen(model)

            x0 = model.meta.x0
            @test x0[findall(==(1), vtags)] ≈ [0.1, 0.2]
            @test x0[findall(==(2), vtags)] ≈ [0.3, 0.4]
            @test x0[findall(==(0), vtags)] ≈ [0.5, 0.8]

            lvar = model.meta.lvar
            uvar = model.meta.uvar

            @test all(lvar[1:ns*nv] .== 0.0)
            @test all(uvar[1:ns*nv] .== 10.0)
            @test all(lvar[ns*nv+1:end] .== 0.0)
            @test all(uvar[ns*nv+1:end] .== 1.0)
        end

    end

    @testset "EachScenario API" begin

        @testset "Variable creation with EachScenario" begin
            ns, nv, nd = 3, 2, 2
            core = TwoStageExaCore(ns)

            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)

            @test core.nvar == ns * nv + nd

            tags = core.tags
            @test length(tags.var_scen) == ns * nv + nd
            @test tags.var_scen[1:ns*nv] == [k for k in 1:ns for _ in 1:nv]
            @test tags.var_scen[ns*nv+1:end] == zeros(Int, nd)
        end

        @testset "Constraint creation with EachScenario" begin
            ns, nv, nd = 3, 2, 1
            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)

            con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
            @add_con(core, (v[v_idx] + d[1] for (i, j, v_idx) in con_data), EachScenario())

            tags = core.tags
            @test length(tags.con_scen) == ns * nv
            @test tags.con_scen == [k for k in 1:ns for _ in 1:nv]
        end

        @testset "Full model structure with EachScenario" begin
            ns, nv, nd = 3, 1, 1
            θ_vals = [4.0, 6.0, 8.0]

            core = TwoStageExaCore(ns)
            v = @add_var(core, nv, EachScenario())
            d = @add_var(core, nd)
            θ = @add_par(core, θ_vals)

            @add_obj(core, d[1]^2)
            @add_obj(core, v[i]^2 for i in 1:ns*nv)

            @add_con(core,
                (2 * v[i] + d[1] - θ[i] for i in 1:ns),
                EachScenario();
                lcon = 0.0, ucon = 0.0,
            )

            model = ExaModel(core)
            vtags = get_var_scen(model)
            ctags = get_con_scen(model)

            @test get_nscen(model) == 3
            @test findall(==(1), vtags) == [1]
            @test findall(==(2), vtags) == [2]
            @test findall(==(3), vtags) == [3]
            @test findall(==(0), vtags) == [4]
            @test findall(==(1), ctags) == [1]
            @test findall(==(2), ctags) == [2]
            @test findall(==(3), ctags) == [3]
        end

        @testset "Multiple variable blocks with EachScenario" begin
            ns = 2
            core = TwoStageExaCore(ns)

            y = @add_var(core, 3, EachScenario())
            z = @add_var(core, 2, EachScenario())
            d = @add_var(core, 1)

            @test core.nvar == ns * 3 + ns * 2 + 1

            tags = core.tags
            @test tags.var_scen == [1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 0]

            model = ExaModel(core)
            vtags = get_var_scen(model)
            @test findall(==(1), vtags) == [1, 2, 3, 7, 8]
            @test findall(==(2), vtags) == [4, 5, 6, 9, 10]
            @test findall(==(0), vtags) == [11]
        end

    end
end

end # module TwoStageTest
