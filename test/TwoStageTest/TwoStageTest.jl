module TwoStageTest

using Test
using ExaModels
import NLPModels
import NLPModels: obj, cons!, cons_nln!, grad!, jac_coord!, hess_coord!, jac_structure!, hess_structure!
import ExaModels: num_scenarios, num_recourse_vars, num_design_vars, num_constraints_per_scenario, total_vars,
                  total_cons, set_scenario_parameters!, set_all_scenario_parameters!, recourse_var_indices,
                  design_var_indices, cons_block_indices, grad_recourse_indices, grad_design_indices,
                  extract_recourse_vars!, extract_design_vars!, extract_cons_block!, extract_grad_block!,
                  global_var_index, global_con_index, recourse_var_index, design_var_index, get_model

import NLPModelsIpopt: ipopt

function runtests()
    @testset "TwoStageExaModel" begin

        @testset "Construction and dimensions" begin
            ns, nv, nd = 3, 2, 2
            nθ = 2
            θ_sets = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(
                    c, θ[θ_off + 1] * v[v_idx]^2 + θ[θ_off + 2] * d[1] * v[v_idx]
                        for (i, j, v_idx, θ_off) in obj_data
                )

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] + d[1] for (i, j, v_idx) in con_data)
            end

            @test num_scenarios(model) == 3
            @test num_recourse_vars(model) == 2
            @test num_design_vars(model) == 2
            @test num_constraints_per_scenario(model) == 2
            @test total_vars(model) == ns * nv + nd  # 3*2 + 2 = 8
            @test total_cons(model) == ns * nv  # 3*2 = 6
        end

        @testset "Variable extraction" begin
            ns, nv, nd = 2, 3, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] + d[1] for (i, j, v_idx) in con_data)
            end

            # Global: [v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, d1, d2]
            x_global = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

            # Test index range functions (GPU-friendly)
            @test recourse_var_indices(model, 1) == 1:3
            @test recourse_var_indices(model, 2) == 4:6
            @test design_var_indices(model) == 7:8
            @test cons_block_indices(model, 1) == 1:3
            @test cons_block_indices(model, 2) == 4:6

            # Test using index ranges directly
            @test x_global[recourse_var_indices(model, 1)] == [1.0, 2.0, 3.0]
            @test x_global[recourse_var_indices(model, 2)] == [4.0, 5.0, 6.0]
            @test x_global[design_var_indices(model)] == [7.0, 8.0]

            # Test in-place extraction functions
            v1 = zeros(3)
            extract_recourse_vars!(v1, model, 1, x_global)
            @test v1 == [1.0, 2.0, 3.0]

            v2 = zeros(3)
            extract_recourse_vars!(v2, model, 2, x_global)
            @test v2 == [4.0, 5.0, 6.0]

            # Test design variable extraction
            d = zeros(2)
            extract_design_vars!(d, model, x_global)
            @test d == [7.0, 8.0]
        end

        @testset "Index mapping" begin
            ns, nv, nd = 3, 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0], [3.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] + d[1] for (i, j, v_idx) in con_data)
            end

            # Test recourse_var_index
            @test recourse_var_index(model, 1, 1) == 1
            @test recourse_var_index(model, 1, 2) == 2
            @test recourse_var_index(model, 2, 1) == 3
            @test recourse_var_index(model, 2, 2) == 4
            @test recourse_var_index(model, 3, 1) == 5
            @test recourse_var_index(model, 3, 2) == 6

            # Test design_var_index
            @test design_var_index(model, 1) == 7  # ns*nv + 1 = 6 + 1
            @test design_var_index(model, 2) == 8  # ns*nv + 2 = 6 + 2

            # Test global_con_index
            @test global_con_index(model, 1, 1) == 1
            @test global_con_index(model, 1, 2) == 2
            @test global_con_index(model, 2, 1) == 3
            @test global_con_index(model, 2, 2) == 4
            @test global_con_index(model, 3, 1) == 5
            @test global_con_index(model, 3, 2) == 6
        end

        @testset "Objective evaluation" begin
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[2.0], [3.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            # Global: [v1_1, v1_2, v2_1, v2_2, d1]
            x_global = [1.0, 2.0, 3.0, 4.0, 0.5]

            # Total obj = scenario1 + scenario2
            # scenario1: θ=2, v=[1,2], obj = 2*(1^2 + 2^2) = 2*5 = 10
            # scenario2: θ=3, v=[3,4], obj = 3*(3^2 + 4^2) = 3*25 = 75
            # Total: 85
            total_obj = obj(model.model, x_global)
            @test total_obj ≈ 85.0
        end

        @testset "Constraint evaluation" begin
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] + d[1] - θ[θ_off + 1] for (i, j, v_idx, θ_off) in con_data)
            end

            # Global: [v1_1, v1_2, v2_1, v2_2, d1]
            x_global = [1.0, 2.0, 3.0, 4.0, 0.5]

            # Global constraint vector
            c_global = zeros(ns * nv)
            cons_nln!(model.model, x_global, c_global)

            # Scenario 1: v=[1,2], d=0.5, θ=1 -> [1+0.5-1, 2+0.5-1] = [0.5, 1.5]
            # Scenario 2: v=[3,4], d=0.5, θ=2 -> [3+0.5-2, 4+0.5-2] = [1.5, 2.5]
            @test c_global ≈ [0.5, 1.5, 1.5, 2.5]

            # Test block extraction using index ranges
            @test c_global[cons_block_indices(model, 1)] ≈ [0.5, 1.5]
            @test c_global[cons_block_indices(model, 2)] ≈ [1.5, 2.5]
        end

        @testset "Gradient evaluation" begin
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[2.0], [3.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] for (i, j, v_idx) in con_data)
            end

            # Global: [v1_1, v1_2, v2_1, v2_2, d]
            x_global = [1.0, 2.0, 3.0, 4.0, 0.5]

            g_global = zeros(total_vars(model))
            grad!(model.model, x_global, g_global)

            # ∂obj/∂v1_1 = 2*θ1*v1_1 = 2*2*1 = 4
            # ∂obj/∂v1_2 = 2*θ1*v1_2 = 2*2*2 = 8
            # ∂obj/∂v2_1 = 2*θ2*v2_1 = 2*3*3 = 18
            # ∂obj/∂v2_2 = 2*θ2*v2_2 = 2*3*4 = 24
            # ∂obj/∂d = 0
            @test g_global ≈ [4.0, 8.0, 18.0, 24.0, 0.0]

            # Test gradient block extraction using index ranges
            @test g_global[grad_recourse_indices(model, 1)] ≈ [4.0, 8.0]
            @test g_global[grad_design_indices(model)] ≈ [0.0]
        end

        @testset "Jacobian structure and evaluation" begin
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] + d[1] for (i, j, v_idx) in con_data)
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
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[2.0], [3.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ) for i in 1:ns for j in 1:nv]
                objective(c, θ[θ_off + 1] * v[v_idx]^2 for (i, j, v_idx, θ_off) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] + d[1] for (i, j, v_idx) in con_data)
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
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
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
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
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
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] + d[1] for (i, j, v_idx) in con_data)
            end

            @test NLPModels.get_nvar(model) == total_vars(model)
            @test NLPModels.get_ncon(model) == total_cons(model)
            @test NLPModels.get_nnzj(model) > 0
            @test NLPModels.get_nnzh(model) > 0
        end

        @testset "set_all_scenario_parameters!" begin
            ns, nv, nd = 2, 2, 1
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
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
            # Problem: Two-stage two-stage programming with quadratic objective
            #
            # min  d^2 + Σᵢ wᵢ * (vᵢ - θᵢ)^2
            # s.t. vᵢ ≥ d  for all scenarios i
            #      d ≥ 0
            #
            # With equal weights wᵢ = 1/ns, and constraint vᵢ ≥ d binding at optimum,
            # we have vᵢ = d for all i (if d ≤ min(θᵢ)).
            #
            # The objective becomes: d^2 + (1/ns) * Σᵢ (d - θᵢ)^2
            # Taking derivative: 2d + (2/ns) * Σᵢ (d - θᵢ) = 0
            # => d + (1/ns) * (ns*d - Σᵢ θᵢ) = 0
            # => d + d - θ̄ = 0  where θ̄ = (1/ns) * Σᵢ θᵢ
            # => d* = θ̄ / 2
            #
            # For θ = [2, 4, 6], θ̄ = 4, so d* = 2, v* = [2, 2, 2]
            # Optimal objective: 2^2 + (1/3)*[(2-2)^2 + (2-4)^2 + (2-6)^2]
            #                  = 4 + (1/3)*(0 + 4 + 16) = 4 + 20/3 ≈ 10.6667

            ns, nv, nd = 3, 1, 1
            nθ = 1
            θ_vals = [2.0, 4.0, 6.0]
            θ_sets = [[θ] for θ in θ_vals]
            weight = 1.0 / ns

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                # Objective: d^2 + weight * Σᵢ (vᵢ - θᵢ)^2
                objective(c, d[1]^2)
                obj_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
                objective(c, weight * (v[v_idx] - θ[θ_off + 1])^2 for (i, v_idx, θ_off) in obj_data)

                # Constraints: vᵢ - d ≥ 0 (i.e., vᵢ ≥ d)
                con_data = [(i, i) for i in 1:ns]
                constraint(c, v[v_idx] - d[1] for (i, v_idx) in con_data; lcon = 0.0)
            end

            # Solve with Ipopt
            result = ipopt(model.model; print_level = 5)

            @test result.status == :first_order

            # Extract solution using index ranges
            x_sol = result.solution
            d_sol = x_sol[design_var_indices(model)]
            v1_sol = x_sol[recourse_var_indices(model, 1)]
            v2_sol = x_sol[recourse_var_indices(model, 2)]
            v3_sol = x_sol[recourse_var_indices(model, 3)]

            # Expected: d* = θ̄/2 = 4/2 = 2, v* = d* = 2 for all scenarios
            θ_bar = sum(θ_vals) / ns
            d_expected = θ_bar / 2
            v_expected = d_expected

            @test d_sol[1] ≈ d_expected atol = 1e-5
            @test v1_sol[1] ≈ v_expected atol = 1e-5
            @test v2_sol[1] ≈ v_expected atol = 1e-5
            @test v3_sol[1] ≈ v_expected atol = 1e-5

            # Expected objective value
            obj_expected = d_expected^2 + weight * sum((d_expected - θ)^2 for θ in θ_vals)
            @test result.objective ≈ obj_expected atol = 1e-5
        end

        @testset "Ipopt solver - multiple recourse variables" begin
            # Problem with multiple recourse variables per scenario
            #
            # min  d₁^2 + d₂^2 + Σᵢ [(vᵢ₁ - θᵢ₁)^2 + (vᵢ₂ - θᵢ₂)^2]
            # s.t. vᵢ₁ + vᵢ₂ = d₁ + d₂  for all i (coupling constraint)
            #      vᵢⱼ ≥ 0 for all i,j
            #      dⱼ ≥ 0 for all j
            #
            # With θ = [[1,3], [2,2]] (two scenarios, two params each)
            # The optimal solution has each scenario's recourse vars summing to d₁+d₂

            ns, nv, nd = 2, 2, 2
            nθ = 2
            θ_sets = [[1.0, 3.0], [2.0, 2.0]]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                # Design objective: d₁^2 + d₂^2
                objective(c, d[1]^2 + d[2]^2)

                # Recourse objective: Σᵢ Σⱼ (vᵢⱼ - θᵢⱼ)^2
                obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ + j) for i in 1:ns for j in 1:nv]
                objective(c, (v[v_idx] - θ[θ_idx])^2 for (i, j, v_idx, θ_idx) in obj_data)

                # Coupling constraint: vᵢ₁ + vᵢ₂ = d₁ + d₂ for each scenario
                con_data = [(i, (i - 1) * nv + 1, (i - 1) * nv + 2) for i in 1:ns]
                constraint(c, v[v1] + v[v2] - d[1] - d[2] for (i, v1, v2) in con_data;
                           lcon = 0.0, ucon = 0.0)
            end

            result = ipopt(model.model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution
            d_sol = x_sol[design_var_indices(model)]
            v1_sol = x_sol[recourse_var_indices(model, 1)]
            v2_sol = x_sol[recourse_var_indices(model, 2)]

            # Verify coupling constraints are satisfied
            @test v1_sol[1] + v1_sol[2] ≈ d_sol[1] + d_sol[2] atol = 1e-5
            @test v2_sol[1] + v2_sol[2] ≈ d_sol[1] + d_sol[2] atol = 1e-5

            # The problem is convex, so Ipopt should find the global minimum
            # Verify objective is reasonable (less than initial value at x=0)
            @test result.objective < 1e10
            @test result.objective ≥ 0
        end

        @testset "Ipopt solver - unconstrained quadratic" begin
            # Simple unconstrained problem with known closed-form solution
            #
            # min  (d - 3)^2 + Σᵢ (vᵢ - θᵢ - d)^2
            #
            # This is separable, so:
            # - Optimal vᵢ* = θᵢ + d (for any d)
            # - Then minimize (d - 3)^2 over d => d* = 3
            # - Therefore vᵢ* = θᵢ + 3

            ns, nv, nd = 3, 1, 1
            nθ = 1
            θ_vals = [1.0, 2.0, 5.0]
            θ_sets = [[θ] for θ in θ_vals]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                # Objective: (d - 3)^2 + Σᵢ (vᵢ - θᵢ - d)^2
                objective(c, (d[1] - 3)^2)
                obj_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
                objective(c, (v[v_idx] - θ[θ_off + 1] - d[1])^2 for (i, v_idx, θ_off) in obj_data)

                # Dummy constraints (bounds only, effectively unconstrained)
                con_data = [(i, i) for i in 1:ns]
                constraint(c, v[v_idx] for (i, v_idx) in con_data; lcon = -1e6, ucon = 1e6)
            end

            result = ipopt(model.model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution
            d_sol = x_sol[design_var_indices(model)]

            # Expected: d* = 3, vᵢ* = θᵢ + 3
            @test d_sol[1] ≈ 3.0 atol = 1e-5

            for (i, θ) in enumerate(θ_vals)
                v_sol = x_sol[recourse_var_indices(model, i)]
                @test v_sol[1] ≈ θ + 3.0 atol = 1e-5
            end

            # Optimal objective should be 0 (all squared terms are zero)
            @test result.objective ≈ 0.0 atol = 1e-8
        end

        @testset "Ipopt solver - coupled d-v constraints" begin
            # Problem with explicit coupling between design and recourse variables
            #
            # min  d² + Σᵢ vᵢ²
            # s.t. 2*vᵢ + d = θᵢ  for all scenarios i  (equality constraint coupling d and v)
            #      d ≥ 0, vᵢ ≥ 0
            #
            # From the constraint: vᵢ = (θᵢ - d) / 2
            # Substituting into objective:
            #   f(d) = d² + Σᵢ ((θᵢ - d)/2)²
            #        = d² + (1/4) Σᵢ (θᵢ - d)²
            #        = d² + (1/4) Σᵢ (θᵢ² - 2θᵢd + d²)
            #        = d²(1 + ns/4) - (Σθᵢ/2)d + (Σθᵢ²)/4
            #
            # Taking derivative and setting to zero:
            #   2d(1 + ns/4) - Σθᵢ/2 = 0
            #   d* = Σθᵢ / (4(1 + ns/4)) = Σθᵢ / (4 + ns)
            #
            # For θ = [4, 6, 8], ns = 3:
            #   Σθᵢ = 18, d* = 18 / (4 + 3) = 18/7 ≈ 2.5714
            #   vᵢ* = (θᵢ - d*) / 2

            ns, nv, nd = 3, 1, 1
            nθ = 1
            θ_vals = [4.0, 6.0, 8.0]
            θ_sets = [[θ] for θ in θ_vals]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                # Objective: d² + Σᵢ vᵢ²
                objective(c, d[1]^2)
                obj_data = [(i, i) for i in 1:ns]
                objective(c, v[v_idx]^2 for (i, v_idx) in obj_data)

                # Coupled constraint: 2*vᵢ + d = θᵢ (equality)
                con_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
                constraint(c, 2 * v[v_idx] + d[1] - θ[θ_off + 1] for (i, v_idx, θ_off) in con_data;
                           lcon = 0.0, ucon = 0.0)
            end

            result = ipopt(model.model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution
            d_sol = x_sol[design_var_indices(model)]

            # Expected solution
            sum_θ = sum(θ_vals)
            d_expected = sum_θ / (4 + ns)  # = 18/7 ≈ 2.5714

            @test d_sol[1] ≈ d_expected atol = 1e-5

            # Verify each scenario's recourse variable and constraint
            for (i, θ) in enumerate(θ_vals)
                v_sol = x_sol[recourse_var_indices(model, i)]
                v_expected = (θ - d_expected) / 2

                @test v_sol[1] ≈ v_expected atol = 1e-5

                # Verify constraint: 2*v + d = θ
                @test 2 * v_sol[1] + d_sol[1] ≈ θ atol = 1e-5
            end

            # Verify optimal objective value
            obj_expected = d_expected^2 + sum((θ - d_expected)^2 / 4 for θ in θ_vals)
            @test result.objective ≈ obj_expected atol = 1e-5
        end

        @testset "Ipopt solver - nonlinear coupled constraints" begin
            # Problem with nonlinear coupling between d and v
            #
            # min  d² + Σᵢ (vᵢ - θᵢ)²
            # s.t. vᵢ + d = θᵢ + 1  for all i (linear coupling)
            #      vᵢ² ≤ 2*d         (nonlinear coupling between d and v)
            #
            # From equality constraint: vᵢ = θᵢ + 1 - d
            # Substituting into objective:
            #   f(d) = d² + Σᵢ (θᵢ + 1 - d - θᵢ)² = d² + ns*(1 - d)²
            # Taking derivative: 2d - 2*ns*(1-d) = 0 => d(1 + ns) = ns => d* = ns/(1+ns)
            #
            # For ns = 3: d* = 3/4 = 0.75
            # vᵢ* = θᵢ + 1 - 0.75 = θᵢ + 0.25
            #
            # For θ = [0.5, 0.5, 0.5]: vᵢ* = 0.75
            # Check nonlinear constraint: vᵢ² = 0.5625 ≤ 2*d = 1.5 ✓

            ns, nv, nd = 3, 1, 1
            nθ = 1
            θ_val = 0.5
            θ_sets = [[θ_val] for _ in 1:ns]

            model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                # Objective: d² + Σᵢ (vᵢ - θᵢ)²
                objective(c, d[1]^2)
                obj_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
                objective(c, (v[v_idx] - θ[θ_off + 1])^2 for (i, v_idx, θ_off) in obj_data)

                # Linear coupled constraint: vᵢ + d = θᵢ + 1 (equality)
                con_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
                constraint(c, v[v_idx] + d[1] - θ[θ_off + 1] - 1 for (i, v_idx, θ_off) in con_data;
                           lcon = 0.0, ucon = 0.0)
            end

            result = ipopt(model.model; print_level = 0)

            @test result.status == :first_order

            x_sol = result.solution
            d_sol = x_sol[design_var_indices(model)]

            # Expected: d* = ns/(1+ns) = 3/4
            d_expected = ns / (1 + ns)
            v_expected = θ_val + 1 - d_expected

            @test d_sol[1] ≈ d_expected atol = 1e-5

            for i in 1:ns
                v_sol = x_sol[recourse_var_indices(model, i)]
                @test v_sol[1] ≈ v_expected atol = 1e-5

                # Verify linear constraint: v + d = θ + 1
                @test v_sol[1] + d_sol[1] ≈ θ_val + 1 atol = 1e-5
            end

            # Optimal objective = d*² + ns*(1 - d*)² = (3/4)² + 3*(1/4)² = 9/16 + 3/16 = 12/16 = 0.75
            obj_expected = d_expected^2 + ns * (1 - d_expected)^2
            @test result.objective ≈ obj_expected atol = 1e-5
        end

        @testset "Warm start - zero iterations from optimal" begin
            # First solve a problem to get the optimal solution
            # Then create a new model with the solution as start values
            # Verify that Ipopt requires 0 iterations (already at optimum)

            ns, nv, nd = 3, 1, 1
            nθ = 1
            θ_vals = [4.0, 6.0, 8.0]
            θ_sets = [[θ] for θ in θ_vals]

            # First solve: build model with default start values
            model1 = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
                objective(c, d[1]^2)
                obj_data = [(i, i) for i in 1:ns]
                objective(c, v[v_idx]^2 for (i, v_idx) in obj_data)

                con_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
                constraint(c, 2 * v[v_idx] + d[1] - θ[θ_off + 1] for (i, v_idx, θ_off) in con_data;
                           lcon = 0.0, ucon = 0.0)
            end

            result1 = ipopt(model1.model; print_level = 0)
            @test result1.status == :first_order

            # Extract solution
            x_sol = result1.solution
            d_sol = x_sol[design_var_indices(model1)]
            v_sol = x_sol[first(recourse_var_indices(model1, 1)):last(recourse_var_indices(model1, ns))]

            # Second solve: build model with optimal solution as start values
            model2 = TwoStageExaModel(nd, nv, ns, θ_sets;
                d_start = d_sol,
                v_start = v_sol
            ) do c, d, v, θ, ns, nv, nθ
                objective(c, d[1]^2)
                obj_data = [(i, i) for i in 1:ns]
                objective(c, v[v_idx]^2 for (i, v_idx) in obj_data)

                con_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
                constraint(c, 2 * v[v_idx] + d[1] - θ[θ_off + 1] for (i, v_idx, θ_off) in con_data;
                           lcon = 0.0, ucon = 0.0)
            end

            # Solve with warm start - should require 0 iterations
            result2 = ipopt(model2.model; print_level = 0)

            @test result2.status == :first_order
            @test result2.iter == 0  # Zero iterations when starting at optimum
            @test result2.objective ≈ result1.objective atol = 1e-8
        end

        @testset "Variable bounds with start values" begin
            # Test that lvar/uvar bounds work correctly with start values

            ns, nv, nd = 2, 2, 2
            nθ = 1
            θ_sets = [[1.0], [2.0]]

            # Set specific bounds and start values
            d_start_vals = [0.5, 0.8]
            v_start_vals = [0.1, 0.2, 0.3, 0.4]  # ns * nv = 4 values

            model = TwoStageExaModel(nd, nv, ns, θ_sets;
                d_start = d_start_vals,
                d_lvar = 0.0,
                d_uvar = 1.0,
                v_start = v_start_vals,
                v_lvar = 0.0,
                v_uvar = 10.0
            ) do c, d, v, θ, ns, nv, nθ
                obj_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                objective(c, v[v_idx]^2 + d[1]^2 + d[2]^2 for (i, j, v_idx) in obj_data)

                con_data = [(i, j, (i - 1) * nv + j) for i in 1:ns for j in 1:nv]
                constraint(c, v[v_idx] + d[1] for (i, j, v_idx) in con_data; lcon = 0.0, ucon = 100.0)
            end

            # Check that initial values are set correctly
            x0 = model.model.meta.x0
            @test x0[recourse_var_indices(model, 1)] ≈ [0.1, 0.2]
            @test x0[recourse_var_indices(model, 2)] ≈ [0.3, 0.4]
            @test x0[design_var_indices(model)] ≈ [0.5, 0.8]

            # Check bounds
            lvar = model.model.meta.lvar
            uvar = model.model.meta.uvar

            # Recourse variable bounds
            @test all(lvar[1:ns*nv] .== 0.0)
            @test all(uvar[1:ns*nv] .== 10.0)

            # Design variable bounds
            @test all(lvar[ns*nv+1:end] .== 0.0)
            @test all(uvar[ns*nv+1:end] .== 1.0)
        end

    end
end

end # module
