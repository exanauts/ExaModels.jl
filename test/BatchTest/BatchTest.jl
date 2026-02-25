module BatchTest

using Test
using ExaModels
import NLPModels
import NLPModels:
    obj, obj!, cons!, cons_nln!, grad!, jac_coord!, hess_coord!, jac_structure!,
    hess_structure!
import ExaModels:
    num_scenarios, set_scenario_parameters!, set_all_scenario_parameters!,
    var_indices, cons_block_indices, get_model

import NLPModelsIpopt: ipopt

import ..BACKENDS
using Adapt

function runtests()
    return @testset "BatchExaModel" begin

        @testset "Construction and dimensions" begin
            ns, nv = 3, 2
            nθ = 2
            θ_data = [1.0 3.0 5.0; 2.0 4.0 6.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, θ[j] * v[j]^2 for j in 1:nv)
                constraint(c, v[j] for j in 1:nv)
            end

            @test num_scenarios(model) == 3
            @test NLPModels.get_nvar(model) == nv
            @test NLPModels.get_ncon(model) == nv
            @test NLPModels.get_nbatch(model) == ns
            @test model.nv == 2
            @test model.nc == 2
        end

        @testset "Batch obj! evaluation" begin
            ns, nv = 2, 2
            θ_data = [2.0 3.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, θ[1] * v[j]^2 for j in 1:nv)
                constraint(c, v[j] for j in 1:nv)
            end

            # bx: (nv, ns) matrix
            bx = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)

            bf = zeros(ns)
            obj!(model, bx, bf)

            # scenario1: θ=2, v=[1,2], obj = 2*(1 + 4) = 10
            # scenario2: θ=3, v=[3,4], obj = 3*(9 + 16) = 75
            @test bf[1] ≈ 10.0
            @test bf[2] ≈ 75.0

            # Consistency: sum(bf) ≈ obj(get_model(m), vec(bx))
            @test sum(bf) ≈ obj(get_model(model), vec(bx))

            # Convenience obj() also works
            bf2 = obj(model, bx)
            @test bf2 ≈ bf
        end

        @testset "Batch grad! evaluation" begin
            ns, nv = 2, 2
            θ_data = [2.0 3.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, θ[1] * v[j]^2 for j in 1:nv)
                constraint(c, v[j] for j in 1:nv)
            end

            bx = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)
            bg = zeros(nv, ns)
            grad!(model, bx, bg)

            # ∂(θ*v²)/∂v = 2*θ*v
            # s1: [2*2*1, 2*2*2] = [4, 8]
            # s2: [2*3*3, 2*3*4] = [18, 24]
            @test bg[:, 1] ≈ [4.0, 8.0]
            @test bg[:, 2] ≈ [18.0, 24.0]

            # Consistency with fused model
            g_flat = zeros(ns * nv)
            grad!(get_model(model), vec(bx), g_flat)
            @test vec(bg) ≈ g_flat
        end

        @testset "Batch cons! evaluation" begin
            ns, nv = 2, 2
            θ_data = [1.0 2.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, v[j]^2 for j in 1:nv)
                constraint(c, v[j] - θ[1] for j in 1:nv)
            end

            bx = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)
            bc = zeros(nv, ns)
            cons!(model, bx, bc)

            # s1: v=[1,2], θ=1 → [0, 1]
            # s2: v=[3,4], θ=2 → [1, 2]
            @test bc[:, 1] ≈ [0.0, 1.0]
            @test bc[:, 2] ≈ [1.0, 2.0]

            # Consistency with fused model
            c_flat = zeros(ns * nv)
            cons_nln!(get_model(model), vec(bx), c_flat)
            @test vec(bc) ≈ c_flat
        end

        @testset "Batch jac_structure! and jac_coord!" begin
            ns, nv = 2, 2
            θ_data = [1.0 2.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, v[j]^2 for j in 1:nv)
                constraint(c, v[j] for j in 1:nv)
            end

            nnzj = NLPModels.get_nnzj(model)
            @test nnzj > 0

            rows = zeros(Int, nnzj)
            cols = zeros(Int, nnzj)
            jac_structure!(model, rows, cols)

            # Per-scenario local indices: rows ∈ 1:nc, cols ∈ 1:nv
            @test all(r -> 1 <= r <= model.nc, rows)
            @test all(c -> 1 <= c <= model.nv, cols)

            # Evaluate Jacobian: bjvals is (nnzj, ns)
            bx = reshape(ones(nv * ns), nv, ns)
            bjvals = zeros(nnzj, ns)
            jac_coord!(model, bx, bjvals)

            # Linear constraints → all values should be 1
            @test all(v -> v ≈ 1.0, bjvals)
        end

        @testset "Batch hess_structure! and hess_coord!" begin
            ns, nv = 2, 2
            θ_data = [2.0 3.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, θ[1] * v[j]^2 for j in 1:nv)
                constraint(c, v[j] for j in 1:nv)
            end

            nnzh = NLPModels.get_nnzh(model)
            @test nnzh > 0

            rows = zeros(Int, nnzh)
            cols = zeros(Int, nnzh)
            hess_structure!(model, rows, cols)

            # Per-scenario local indices
            @test all(r -> 1 <= r <= model.nv, rows)
            @test all(c -> 1 <= c <= model.nv, cols)

            # Evaluate with uniform obj_weight
            bx = reshape(ones(nv * ns), nv, ns)
            by = zeros(model.nc, ns)
            bobj_weight = ones(ns)
            bhvals = zeros(nnzh, ns)
            hess_coord!(model, bx, by, bobj_weight, bhvals)

            # Hessian of θ*v[j]^2 is 2*θ on diagonal
            # s1: 2*2 = 4, s2: 2*3 = 6
            @test any(v -> v ≈ 4.0, bhvals[:, 1])
            @test any(v -> v ≈ 6.0, bhvals[:, 2])
        end

        @testset "hess_coord! with varying obj_weight" begin
            ns, nv = 2, 2
            θ_data = [2.0 3.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, θ[1] * v[j]^2 for j in 1:nv)
                constraint(c, v[j]^3 for j in 1:nv)
            end

            nnzh = NLPModels.get_nnzh(model)
            bx = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)
            by = ones(model.nc, ns)

            # Uniform weight for reference
            bhvals_uniform = zeros(nnzh, ns)
            hess_coord!(model, bx, by, [1.0, 1.0], bhvals_uniform)

            # Varying weights
            bhvals_varying = zeros(nnzh, ns)
            hess_coord!(model, bx, by, [2.0, 0.5], bhvals_varying)

            # Compute reference via fused model for each scenario
            # With varying weights, obj part is scaled differently per scenario
            inner = get_model(model)
            total_nnzh = NLPModels.get_nnzh(inner)

            # obj-only hessian
            hess_obj = zeros(total_nnzh)
            hess_coord!(inner, vec(bx), zeros(ns * model.nc), hess_obj; obj_weight = 1.0)

            # con-only hessian
            hess_con = zeros(total_nnzh)
            hess_coord!(inner, vec(bx), vec(by), hess_con; obj_weight = 0.0)

            # Verify per-scenario reconstruction
            perm = model.hess_perm
            for s in 1:ns
                for k in 1:nnzh
                    idx = perm[(s - 1) * nnzh + k]
                    expected = [2.0, 0.5][s] * hess_obj[idx] + hess_con[idx]
                    @test bhvals_varying[k, s] ≈ expected
                end
            end
        end

        @testset "Multiple constraint() calls" begin
            ns, nv = 2, 3
            θ_data = reshape([1.0, 2.0], 1, ns)

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, θ[1] * v[j]^2 for j in 1:nv)
                # Two separate constraint() calls per scenario
                constraint(c, v[j] - θ[1] for j in 1:nv)
                constraint(c, v[1] + v[2] + v[3]; ucon = 10.0)
            end

            # nc = nv + 1 = 4 per scenario
            @test model.nc == nv + 1
            @test NLPModels.get_ncon(model) == nv + 1
            @test NLPModels.get_nbatch(model) == ns

            bx = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], nv, ns)
            bc = zeros(model.nc, ns)
            cons!(model, bx, bc)

            # s1: v=[1,2,3], θ=1 → [1-1, 2-1, 3-1, 1+2+3] = [0, 1, 2, 6]
            # s2: v=[4,5,6], θ=2 → [4-2, 5-2, 6-2, 4+5+6] = [2, 3, 4, 15]
            @test bc[:, 1] ≈ [0.0, 1.0, 2.0, 6.0]
            @test bc[:, 2] ≈ [2.0, 3.0, 4.0, 15.0]

            # Consistency with fused model
            c_flat = zeros(model.nc * ns)
            cons_nln!(get_model(model), vec(bx), c_flat)
            @test vec(bc) ≈ c_flat

            # Jacobian structure should have local indices
            nnzj = NLPModels.get_nnzj(model)
            rows = zeros(Int, nnzj)
            cols = zeros(Int, nnzj)
            jac_structure!(model, rows, cols)
            @test all(r -> 1 <= r <= model.nc, rows)
            @test all(c -> 1 <= c <= model.nv, cols)

            # Hessian with both obj and con contributions
            nnzh = NLPModels.get_nnzh(model)
            hrows = zeros(Int, nnzh)
            hcols = zeros(Int, nnzh)
            hess_structure!(model, hrows, hcols)
            @test all(r -> 1 <= r <= model.nv, hrows)
            @test all(c -> 1 <= c <= model.nv, hcols)

            # Evaluate hessian — both obj and con have nonzero second derivatives
            by = ones(model.nc, ns)
            bobj_weight = ones(ns)
            bhvals = zeros(nnzh, ns)
            hess_coord!(model, bx, by, bobj_weight, bhvals)
            @test any(v -> v != 0.0, bhvals[:, 1])
            @test any(v -> v != 0.0, bhvals[:, 2])
        end

        @testset "Parameter updates" begin
            ns, nv = 2, 2
            θ_data = [1.0 2.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, θ[1] * v[j]^2 for j in 1:nv)
                constraint(c, v[j] for j in 1:nv)
            end

            x_global = ones(ns * nv)

            # Initial: θ1=1, θ2=2, so obj = 1*2 + 2*2 = 6
            @test obj(get_model(model), x_global) ≈ 6.0

            # Update scenario 1: θ1 = 5
            set_scenario_parameters!(model, 1, [5.0])
            @test obj(get_model(model), x_global) ≈ 14.0

            # set_all_scenario_parameters!
            set_all_scenario_parameters!(model, [[10.0], [20.0]])
            @test obj(get_model(model), x_global) ≈ 60.0
        end

        @testset "Get underlying model" begin
            ns, nv = 2, 2
            θ_data = [1.0 2.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, v[j]^2 for j in 1:nv)
                constraint(c, v[j] for j in 1:nv)
            end

            inner = get_model(model)
            @test inner isa ExaModels.ExaModel
            @test NLPModels.get_nvar(inner) == ns * nv
            @test NLPModels.get_ncon(inner) == ns * nv
        end

        @testset "Variable bounds and start values" begin
            ns, nv = 2, 2
            θ_data = [1.0 2.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv; start = 0.5, lvar = 0.0, uvar = 10.0)
                objective(c, v[j]^2 for j in 1:nv)
                constraint(c, v[j] for j in 1:nv; lcon = 0.0, ucon = 100.0)
            end

            # Check that meta matrices have correct shape and values
            @test size(model.meta.x0) == (nv, ns)
            @test all(model.meta.x0 .== 0.5)

            @test all(model.meta.lvar .== 0.0)
            @test all(model.meta.uvar .== 10.0)
        end

        @testset "Ipopt solver with known solution" begin
            ns, nv = 3, 1
            θ_vals = [2.0, 4.0, 6.0]
            θ_data = reshape(θ_vals, 1, ns)

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, (v[1] - θ[1])^2)
                constraint(c, v[1]; lcon = 0.0, ucon = Inf)
            end

            # Solve via fused model
            result = ipopt(get_model(model); print_level = 0)
            @test result.status == :first_order

            x_sol = result.solution
            for (i, θ) in enumerate(θ_vals)
                @test x_sol[var_indices(model, i)] ≈ [θ] atol = 1.0e-5
            end
            @test result.objective ≈ 0.0 atol = 1.0e-8
        end

        @testset "Ipopt solver - multiple variables per scenario" begin
            ns, nv = 2, 2
            θ_data = [1.0 2.0; 3.0 2.0]

            c = ExaCore()
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, (v[j] - θ[j])^2 for j in 1:nv)
                constraint(c, v[1] + v[2]; ucon = 10.0)
            end

            result = ipopt(get_model(model); print_level = 0)
            @test result.status == :first_order

            x_sol = result.solution
            @test x_sol[var_indices(model, 1)] ≈ [1.0, 3.0] atol = 1.0e-5
            @test x_sol[var_indices(model, 2)] ≈ [2.0, 2.0] atol = 1.0e-5
            @test result.objective ≈ 0.0 atol = 1.0e-8
        end

        @testset "Multi-backend: $backend" for backend in BACKENDS
            ns, nv = 2, 2
            θ_data = [2.0 3.0]

            c = ExaCore(; backend = backend)
            model = BatchExaModel(c, ns, θ_data) do c, θ
                v = variable(c, nv)
                objective(c, θ[1] * v[j]^2 for j in 1:nv)
                constraint(c, v[j] - θ[1] for j in 1:nv)
            end

            # Create test matrix on the right device
            bx_cpu = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)
            bx = backend === nothing ? bx_cpu : adapt(backend, bx_cpu)

            # Batch obj!
            bf = similar(bx, ns)
            obj!(model, bx, bf)
            @test Array(bf) ≈ [10.0, 75.0]
            @test sum(Array(bf)) ≈ 85.0

            # Consistency with fused model
            @test sum(Array(bf)) ≈ obj(get_model(model), vec(bx))

            # Batch grad!
            bg = similar(bx, nv, ns)
            grad!(model, bx, bg)
            @test Array(bg) ≈ [4.0 18.0; 8.0 24.0]

            # Batch cons!
            bc = similar(bx, model.nc, ns)
            cons!(model, bx, bc)
            @test Array(bc) ≈ [-1.0 0.0; 0.0 1.0]

            # Batch jac_coord!
            nnzj = NLPModels.get_nnzj(model)
            bjvals = similar(bx, nnzj, ns)
            jac_coord!(model, bx, bjvals)
            @test all(v -> v ≈ 1.0, Array(bjvals))

            # Batch hess_coord!
            nnzh = NLPModels.get_nnzh(model)
            by = similar(bx, model.nc, ns)
            fill!(by, zero(eltype(by)))
            bobj_weight = similar(bx, ns)
            fill!(bobj_weight, one(eltype(bx)))
            bhvals = similar(bx, nnzh, ns)
            hess_coord!(model, bx, by, bobj_weight, bhvals)
            hv = Array(bhvals)
            @test any(v -> v ≈ 4.0, hv[:, 1])
            @test any(v -> v ≈ 6.0, hv[:, 2])
        end

    end
end

end # module
