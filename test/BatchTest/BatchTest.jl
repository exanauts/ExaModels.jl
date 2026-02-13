module BatchTest

using Test
using ExaModels
import NLPModels
import NLPModels: obj, cons!, cons_nln!, grad!, jac_coord!, hess_coord!, jac_structure!, hess_structure!
import ExaModels: num_scenarios, num_vars_per_scenario, num_constraints_per_scenario, total_vars,
                  total_cons, set_scenario_parameters!, set_all_scenario_parameters!, var_indices,
                  cons_block_indices, grad_indices,
                  extract_vars!, extract_cons_block!,
                  global_var_index, global_con_index, get_model

import NLPModelsIpopt: ipopt

function runtests()
    @testset "BatchExaModel" begin

        @testset "Construction and dimensions" begin
            ns, nv = 3, 2
            nθ = 2
            θ_sets = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(
                    c, θ[θ_off + 1] * v[v_idx]^2
                        for (i, j, v_idx, θ_off) in obj_data
                )

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            @test num_scenarios(model) == 3
            @test num_vars_per_scenario(model) == 2
            @test num_constraints_per_scenario(model) == 2
            @test total_vars(model) == ns * nv  # 3*2 = 6
            @test total_cons(model) == ns * nv  # 3*2 = 6
        end

        @testset "Variable extraction" begin
            ns, nv = 2, 3
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            # Global: [v1_1, v1_2, v1_3, v2_1, v2_2, v2_3]
            x_global = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

            # Test index range functions
            @test var_indices(model, 1) == 1:3
            @test var_indices(model, 2) == 4:6
            @test cons_block_indices(model, 1) == 1:3
            @test cons_block_indices(model, 2) == 4:6

            # Test using index ranges directly
            @test x_global[var_indices(model, 1)] == [1.0, 2.0, 3.0]
            @test x_global[var_indices(model, 2)] == [4.0, 5.0, 6.0]

            # Test in-place extraction functions
            v1 = zeros(3)
            extract_vars!(v1, model, 1, x_global)
            @test v1 == [1.0, 2.0, 3.0]

            v2 = zeros(3)
            extract_vars!(v2, model, 2, x_global)
            @test v2 == [4.0, 5.0, 6.0]
        end

        @testset "Index mapping" begin
            ns, nv = 3, 2
            nθ = 1
            θ_sets = [[1.0], [2.0], [3.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            # Test global_var_index
            @test global_var_index(model, 1, 1) == 1
            @test global_var_index(model, 1, 2) == 2
            @test global_var_index(model, 2, 1) == 3
            @test global_var_index(model, 2, 2) == 4
            @test global_var_index(model, 3, 1) == 5
            @test global_var_index(model, 3, 2) == 6

            # Test global_con_index
            @test global_con_index(model, 1, 1) == 1
            @test global_con_index(model, 1, 2) == 2
            @test global_con_index(model, 2, 1) == 3
            @test global_con_index(model, 2, 2) == 4
            @test global_con_index(model, 3, 1) == 5
            @test global_con_index(model, 3, 2) == 6
        end

        @testset "Objective evaluation" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[2.0], [3.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            # Global: [v1_1, v1_2, v2_1, v2_2]
            x_global = [1.0, 2.0, 3.0, 4.0]

            # Total obj = scenario1 + scenario2
            # scenario1: θ=2, v=[1,2], obj = 2*(1^2 + 2^2) = 2*5 = 10
            # scenario2: θ=3, v=[3,4], obj = 3*(3^2 + 4^2) = 3*25 = 75
            # Total: 85
            total_obj = obj(model.model, x_global)
            @test total_obj ≈ 85.0
        end

        @testset "Constraint evaluation" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] - θ[θ_off + 1] for (i, j, v_idx, θ_off) in con_data)
            end

            # Global: [v1_1, v1_2, v2_1, v2_2]
            x_global = [1.0, 2.0, 3.0, 4.0]

            # Global constraint vector
            c_global = zeros(ns * nv)
            cons_nln!(model.model, x_global, c_global)

            # Scenario 1: v=[1,2], θ=1 -> [1-1, 2-1] = [0, 1]
            # Scenario 2: v=[3,4], θ=2 -> [3-2, 4-2] = [1, 2]
            @test c_global ≈ [0.0, 1.0, 1.0, 2.0]

            # Test block extraction using index ranges
            @test c_global[cons_block_indices(model, 1)] ≈ [0.0, 1.0]
            @test c_global[cons_block_indices(model, 2)] ≈ [1.0, 2.0]
        end

        @testset "Gradient evaluation" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[2.0], [3.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            # Global: [v1_1, v1_2, v2_1, v2_2]
            x_global = [1.0, 2.0, 3.0, 4.0]

            g_global = zeros(total_vars(model))
            grad!(model.model, x_global, g_global)

            # ∂obj/∂v1_1 = 2*θ1*v1_1 = 2*2*1 = 4
            # ∂obj/∂v1_2 = 2*θ1*v1_2 = 2*2*2 = 8
            # ∂obj/∂v2_1 = 2*θ2*v2_1 = 2*3*3 = 18
            # ∂obj/∂v2_2 = 2*θ2*v2_2 = 2*3*4 = 24
            @test g_global ≈ [4.0, 8.0, 18.0, 24.0]

            # Test gradient block extraction using index ranges
            @test g_global[grad_indices(model, 1)] ≈ [4.0, 8.0]
            @test g_global[grad_indices(model, 2)] ≈ [18.0, 24.0]
        end

        @testset "Jacobian structure and evaluation" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            nnzj = NLPModels.get_nnzj(model)
            @test nnzj > 0

            rows = zeros(Int, nnzj)
            cols = zeros(Int, nnzj)
            jac_structure!(model.model, rows, cols)

            # Verify indices are valid
            @test all(r -> 1 <= r <= total_cons(model), rows)
            @test all(c -> 1 <= c <= total_vars(model), cols)

            # Evaluate Jacobian
            x_global = ones(total_vars(model))
            jac_vals = zeros(nnzj)
            jac_coord!(model.model, x_global, jac_vals)

            # All entries should be 1 (linear constraints)
            @test all(v -> v ≈ 1.0, jac_vals)
        end

        @testset "Hessian structure and evaluation" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[2.0], [3.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            nnzh = NLPModels.get_nnzh(model)
            @test nnzh > 0

            rows = zeros(Int, nnzh)
            cols = zeros(Int, nnzh)
            hess_structure!(model.model, rows, cols)

            @test all(r -> 1 <= r <= total_vars(model), rows)
            @test all(c -> 1 <= c <= total_vars(model), cols)

            x_global = ones(total_vars(model))
            y_global = zeros(total_cons(model))
            hess_vals = zeros(nnzh)
            hess_coord!(model.model, x_global, y_global, hess_vals)

            # Hessian of θ*v[j]^2 is 2*θ on diagonal
            # Scenario 1: 2*2 = 4 for v1_1, v1_2
            # Scenario 2: 2*3 = 6 for v2_1, v2_2
            @test any(v -> v ≈ 4.0, hess_vals)
            @test any(v -> v ≈ 6.0, hess_vals)
        end

        @testset "Parameter updates" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            x_global = ones(total_vars(model))

            # Initial: θ1=1, θ2=2, so obj = 1*2 + 2*2 = 6
            obj_before = obj(model.model, x_global)
            @test obj_before ≈ 6.0

            # Update scenario 1: θ1 = 5
            set_scenario_parameters!(model, 1, [5.0])

            # Now: θ1=5, θ2=2, so obj = 5*2 + 2*2 = 14
            obj_after = obj(model.model, x_global)
            @test obj_after ≈ 14.0
        end

        @testset "Get underlying model" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            inner = get_model(model)
            @test inner isa ExaModels.ExaModel
            @test NLPModels.get_nvar(inner) == total_vars(model)
            @test NLPModels.get_ncon(inner) == total_cons(model)
        end

        @testset "NLPModels interface" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            @test NLPModels.get_nvar(model) == total_vars(model)
            @test NLPModels.get_ncon(model) == total_cons(model)
            @test NLPModels.get_nnzj(model) > 0
            @test NLPModels.get_nnzh(model) > 0
        end

        @testset "set_all_scenario_parameters!" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            x_global = ones(total_vars(model))

            # Initial: θ1=1, θ2=2, so obj = 1*2 + 2*2 = 6
            @test obj(model.model, x_global) ≈ 6.0

            # Update all scenarios
            new_θ_sets = [[10.0], [20.0]]
            set_all_scenario_parameters!(model, new_θ_sets)

            # Now: θ1=10, θ2=20, so obj = 10*2 + 20*2 = 60
            @test obj(model.model, x_global) ≈ 60.0
        end

        @testset "Ipopt solver with known solution" begin
            # Problem: Independent quadratic problems
            #
            # min  Σᵢ (vᵢ - θᵢ)^2
            # s.t. vᵢ ≥ 0  for all scenarios i
            #
            # Each scenario is independent:
            # If θᵢ ≥ 0, optimal vᵢ* = θᵢ, obj = 0
            # If θᵢ < 0, optimal vᵢ* = 0, obj = θᵢ^2

            ns, nv = 3, 1
            nθ = 1
            θ_vals = [2.0, 4.0, 6.0]
            θ_sets = [[θ] for θ in θ_vals]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
                objective(c, (v[v_idx] - θ[θ_off + 1])^2 for (i, v_idx, θ_off) in obj_data)

                # Dummy constraints (bounds only)
                con_data = [(i, i) for i in 1:ns]
                constraint(c, v[v_idx] for (i, v_idx) in con_data; lcon = 0.0, ucon = Inf)
            end

            result = ipopt(model.model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution

            # Each scenario's variable should equal its parameter
            for (i, θ) in enumerate(θ_vals)
                v_sol = x_sol[var_indices(model, i)]
                @test v_sol[1] ≈ θ atol = 1e-5
            end

            # Optimal objective should be 0
            @test result.objective ≈ 0.0 atol = 1e-8
        end

        @testset "Ipopt solver - multiple variables per scenario" begin
            # Problem with multiple variables per scenario
            #
            # min  Σᵢ [(vᵢ₁ - θᵢ₁)^2 + (vᵢ₂ - θᵢ₂)^2]
            # s.t. vᵢ₁ + vᵢ₂ ≤ 10  for all i
            #      vᵢⱼ ≥ 0 for all i,j
            #
            # With θ = [[1,3], [2,2]]:
            # Optimal: vᵢⱼ = θᵢⱼ (unconstrained min, constraints are slack)

            ns, nv = 2, 2
            nθ = 2
            θ_sets = [[1.0, 3.0], [2.0, 2.0]]

            model = BatchExaModel(nv, ns, θ_sets) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ + j) for i in 1:ns for j in 1:nv]
                objective(c, (v[v_idx] - θ[θ_idx])^2 for (i, j, v_idx, θ_idx) in obj_data)

                # Sum constraint per scenario: v₁ + v₂ ≤ 10
                con_data = [(i, (i - 1) * nv + 1, (i - 1) * nv + 2) for i in 1:ns]
                constraint(c, v[v1] + v[v2] for (i, v1, v2) in con_data; ucon = 10.0)
            end

            result = ipopt(model.model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution

            # Scenario 1: v = [1, 3]
            @test x_sol[var_indices(model, 1)] ≈ [1.0, 3.0] atol = 1e-5

            # Scenario 2: v = [2, 2]
            @test x_sol[var_indices(model, 2)] ≈ [2.0, 2.0] atol = 1e-5

            @test result.objective ≈ 0.0 atol = 1e-8
        end

        @testset "Variable bounds with start values" begin
            ns, nv = 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            v_start_vals = [0.1, 0.2, 0.3, 0.4]  # ns * nv = 4 values

            model = BatchExaModel(nv, ns, θ_sets;
                v_start = v_start_vals,
                v_lvar = 0.0,
                v_uvar = 10.0
            ) do c, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data; lcon = 0.0, ucon = 100.0)
            end

            # Check that initial values are set correctly
            x0 = model.model.meta.x0
            @test x0[var_indices(model, 1)] ≈ [0.1, 0.2]
            @test x0[var_indices(model, 2)] ≈ [0.3, 0.4]

            # Check bounds
            lvar = model.model.meta.lvar
            uvar = model.model.meta.uvar

            @test all(lvar .== 0.0)
            @test all(uvar .== 10.0)
        end

    end
end

end # module
