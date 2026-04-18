module BatchTest

using Test
using ExaModels
import NLPModels
import NLPModels:
    obj, cons!, cons_nln!, grad!, jac_coord!, hess_coord!, jac_structure!,
    hess_structure!
import ExaModels: obj!, var_indices, cons_block_indices, get_model, get_nbatch

import NLPModelsIpopt: ipopt

import ..BACKENDS
using Adapt

function runtests()
    return @testset "Batch ExaModel" begin

        @testset "Construction and dimensions" begin
            ns, nv = 3, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [1.0, 2.0])
            @add_obj(c, θ[j, s] * v[j, s]^2 for j in 1:nv, s in 1:ns)
            @add_con(c, EachInstance(), v[j, s] for j in 1:nv, s in 1:ns)
            model = ExaModel(c)

            @test get_nbatch(model) == 3
            @test NLPModels.get_nvar(model) == nv
            @test NLPModels.get_ncon(model) == nv

            # ExaModel <: AbstractNLPModel
            @test model isa NLPModels.AbstractNLPModel
        end

        @testset "Batch obj! evaluation" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [2.0])
            nb = get_nbatch(c)
            c, _ = add_obj(c, θ[1, s] * v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s] for j in 1:nv, s in 1:nb)
            model = ExaModel(c)

            # bx: (nv, ns) matrix
            bx = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)

            bf = zeros(ns)
            obj!(model, bx, bf)

            # Both instances have θ=2
            # instance1: v=[1,2], obj = 2*(1 + 4) = 10
            # instance2: v=[3,4], obj = 2*(9 + 16) = 50
            @test bf[1] ≈ 10.0
            @test bf[2] ≈ 50.0

            # Consistency: sum(bf) ≈ obj(get_model(m), vec(bx))
            @test sum(bf) ≈ obj(get_model(model), vec(bx))

            # Convenience obj() also works
            bf2 = obj(model, bx)
            @test bf2 ≈ bf
        end

        @testset "Batch grad! evaluation" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [2.0])
            nb = get_nbatch(c)
            c, _ = add_obj(c, θ[1, s] * v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s] for j in 1:nv, s in 1:nb)
            model = ExaModel(c)

            bx = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)
            bg = zeros(nv, ns)
            grad!(model, bx, bg)

            # ∂(θ*v²)/∂v = 2*θ*v, θ=2 for both instances
            # s1: [2*2*1, 2*2*2] = [4, 8]
            # s2: [2*2*3, 2*2*4] = [12, 16]
            @test bg[:, 1] ≈ [4.0, 8.0]
            @test bg[:, 2] ≈ [12.0, 16.0]

            # Consistency with fused model
            g_flat = zeros(ns * nv)
            grad!(get_model(model), vec(bx), g_flat)
            @test vec(bg) ≈ g_flat
        end

        @testset "Batch cons! evaluation" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [1.0])
            nb = get_nbatch(c)
            c, _ = add_obj(c, v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s] - θ[1, s] for j in 1:nv, s in 1:nb)
            model = ExaModel(c)

            bx = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)
            bc = zeros(nv, ns)
            cons!(model, bx, bc)

            # Both instances have θ=1
            # s1: v=[1,2], θ=1 → [0, 1]
            # s2: v=[3,4], θ=1 → [2, 3]
            @test bc[:, 1] ≈ [0.0, 1.0]
            @test bc[:, 2] ≈ [2.0, 3.0]

            # Consistency with fused model
            c_flat = zeros(ns * nv)
            cons_nln!(get_model(model), vec(bx), c_flat)
            @test vec(bc) ≈ c_flat
        end

        @testset "Batch jac_structure! and jac_coord!" begin
            ns, nv, nc = 2, 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            nb = get_nbatch(c)
            c, _ = add_obj(c, v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s] for j in 1:nv, s in 1:nb)
            model = ExaModel(c)

            nnzj = NLPModels.get_nnzj(model)
            @test nnzj > 0

            rows = zeros(Int, nnzj)
            cols = zeros(Int, nnzj)
            jac_structure!(model, rows, cols)

            # Per-instance local indices: rows ∈ 1:nc, cols ∈ 1:nv
            @test all(r -> 1 <= r <= nc, rows)
            @test all(c -> 1 <= c <= nv, cols)

            # Evaluate Jacobian: jvals is flat vector (nnzj * ns)
            bx = reshape(ones(nv * ns), nv, ns)
            jvals = zeros(nnzj * ns)
            jac_coord!(model, bx, jvals)

            # Linear constraints → all values should be 1
            @test all(v -> v ≈ 1.0, jvals)
        end

        @testset "Batch hess_structure! and hess_coord!" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [2.0])
            nb = get_nbatch(c)
            c, _ = add_obj(c, θ[1, s] * v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s] for j in 1:nv, s in 1:nb)
            model = ExaModel(c)

            nnzh = NLPModels.get_nnzh(model)
            @test nnzh > 0

            rows = zeros(Int, nnzh)
            cols = zeros(Int, nnzh)
            hess_structure!(model, rows, cols)

            # Per-instance local indices
            @test all(r -> 1 <= r <= nv, rows)
            @test all(c -> 1 <= c <= nv, cols)

            # Evaluate with uniform obj_weight
            bx = reshape(ones(nv * ns), nv, ns)
            by = zeros(nv, ns)
            bobj_weight = ones(ns)
            hvals = zeros(nnzh * ns)
            hess_coord!(model, bx, by, bobj_weight, hvals)

            # Hessian of θ*v[j]^2 is 2*θ on diagonal, θ=2 for all instances
            # Both instances: 2*2 = 4
            hvals_s1 = hvals[1:nnzh]
            hvals_s2 = hvals[nnzh+1:2*nnzh]
            @test any(v -> v ≈ 4.0, hvals_s1)
            @test any(v -> v ≈ 4.0, hvals_s2)
        end

        @testset "hess_coord! with varying obj_weight" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [2.0])
            nb = get_nbatch(c)
            c, _ = add_obj(c, θ[1, s] * v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s]^3 for j in 1:nv, s in 1:nb)
            model = ExaModel(c)

            nc = NLPModels.get_ncon(model)
            nnzh = NLPModels.get_nnzh(model)
            bx = reshape([1.0, 2.0, 3.0, 4.0], nv, ns)
            by = ones(nc, ns)

            # Uniform weight for reference
            hvals_uniform = zeros(nnzh * ns)
            hess_coord!(model, bx, by, [1.0, 1.0], hvals_uniform)

            # Varying weights
            hvals_varying = zeros(nnzh * ns)
            hess_coord!(model, bx, by, [2.0, 0.5], hvals_varying)

            # Compute reference via fused model for each instance
            inner = get_model(model)
            total_nnzh = NLPModels.get_nnzh(inner)

            # obj-only hessian
            hess_obj = zeros(total_nnzh)
            hess_coord!(inner, vec(bx), zeros(ns * nc), hess_obj; obj_weight = 1.0)

            # con-only hessian
            hess_con = zeros(total_nnzh)
            hess_coord!(inner, vec(bx), vec(by), hess_con; obj_weight = 0.0)

            # Verify per-instance reconstruction
            perm = ExaModels._batch_hess_perm(model)
            for s in 1:ns
                for k in 1:nnzh
                    idx = perm[(s - 1) * nnzh + k]
                    expected = [2.0, 0.5][s] * hess_obj[idx] + hess_con[idx]
                    @test hvals_varying[(s - 1) * nnzh + k] ≈ expected
                end
            end
        end

        @testset "Multiple constraint calls" begin
            ns, nv = 2, 3

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [1.0])
            nb = get_nbatch(c)
            c, _ = add_obj(c, θ[1, s] * v[j, s]^2 for j in 1:nv, s in 1:nb)
            # Two separate constraint calls per instance
            c, _ = add_con(c, EachInstance(), v[j, s] - θ[1, s] for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[1, s] + v[2, s] + v[3, s] for s in 1:nb; ucon = 10.0)
            model = ExaModel(c)

            nc = NLPModels.get_ncon(model)
            # nc = nv + 1 = 4 per instance
            @test nc == nv + 1
            @test get_nbatch(model) == ns

            bx = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], nv, ns)
            bc = zeros(nc, ns)
            cons!(model, bx, bc)

            # Consistency with fused model — this is the definitive check
            c_flat = zeros(nc * ns)
            cons_nln!(get_model(model), vec(bx), c_flat)
            @test vec(bc) ≈ c_flat

            # Hessian with both obj and con contributions
            nnzh = NLPModels.get_nnzh(model)

            # Evaluate hessian
            by = ones(nc, ns)
            bobj_weight = ones(ns)
            hvals = zeros(nnzh * ns)
            hess_coord!(model, bx, by, bobj_weight, hvals)
            @test any(v -> v != 0.0, hvals[1:nnzh])
            @test any(v -> v != 0.0, hvals[nnzh+1:2*nnzh])
        end

        @testset "Get underlying model" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            nb = get_nbatch(c)
            c, _ = add_obj(c, v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s] for j in 1:nv, s in 1:nb)
            model = ExaModel(c)

            inner = get_model(model)
            @test inner isa ExaModels.ExaModel
            @test NLPModels.get_nvar(inner) == ns * nv
            @test NLPModels.get_ncon(inner) == ns * nv
        end

        @testset "Variable bounds and start values" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv; start = 0.5, lvar = 0.0, uvar = 10.0)
            nb = get_nbatch(c)
            c, _ = add_obj(c, v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s] for j in 1:nv, s in 1:nb; lcon = 0.0, ucon = 100.0)
            model = ExaModel(c)

            # Check that meta matrices have correct shape and values
            @test size(model.meta.x0) == (nv, ns)
            @test all(model.meta.x0 .== 0.5)

            @test all(model.meta.lvar .== 0.0)
            @test all(model.meta.uvar .== 10.0)
        end

        @testset "Error on vector arguments" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            nb = get_nbatch(c)
            c, _ = add_obj(c, v[j, s]^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[j, s] for j in 1:nv, s in 1:nb)
            model = ExaModel(c)

            x_vec = ones(nv)
            c_vec = zeros(nv)
            g_vec = zeros(nv)

            @test_throws ArgumentError obj(model, x_vec)
            @test_throws ArgumentError cons!(model, x_vec, c_vec)
            @test_throws ArgumentError grad!(model, x_vec, g_vec)
        end

        @testset "Ipopt solver with known solution" begin
            ns, nv = 3, 1

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [2.0])
            nb = get_nbatch(c)
            # Each instance minimizes (v[1,s] - θ[1,s])^2
            # But θ is the same (2.0) for all instances with this API
            c, _ = add_obj(c, (v[1, s] - θ[1, s])^2 for s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[1, s] for s in 1:nb; lcon = 0.0, ucon = Inf)
            model = ExaModel(c)

            # Solve via fused model
            result = ipopt(get_model(model); print_level = 0)
            @test result.status == :first_order

            x_sol = result.solution
            # All instances have θ=2, so optimal v*=2 for each
            for i in 1:ns
                @test x_sol[var_indices(model, i)] ≈ [2.0] atol = 1.0e-5
            end
            @test result.objective ≈ 0.0 atol = 1.0e-8
        end

        @testset "Ipopt solver - multiple variables per instance" begin
            ns, nv = 2, 2

            c = BatchExaCore(ns)
            @add_var(c, v, EachInstance(), nv)
            @add_par(c, θ, EachInstance(), [1.0, 3.0])
            nb = get_nbatch(c)
            # Each instance minimizes sum of (v[j,s] - θ[j,s])^2
            c, _ = add_obj(c, (v[j, s] - θ[j, s])^2 for j in 1:nv, s in 1:nb)
            c, _ = add_con(c, EachInstance(), v[1, s] + v[2, s] for s in 1:nb; ucon = 10.0)
            model = ExaModel(c)

            result = ipopt(get_model(model); print_level = 0)
            @test result.status == :first_order

            x_sol = result.solution
            # Both instances have θ=[1,3], so optimal v*=[1,3] for each
            @test x_sol[var_indices(model, 1)] ≈ [1.0, 3.0] atol = 1.0e-5
            @test x_sol[var_indices(model, 2)] ≈ [1.0, 3.0] atol = 1.0e-5
            @test result.objective ≈ 0.0 atol = 1.0e-8
        end

    end
end

end # module
