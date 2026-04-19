module BatchTest

using Test
using ExaModels
import NLPModels
import NLPModels:
    obj, cons!, cons_nln!, grad!, jac_coord!, hess_coord!, jac_structure!,
    hess_structure!
import NLPModels: obj!
import ExaModels: var_indices, cons_block_indices, get_model, get_nbatch,
    get_start, get_lvar, get_uvar, get_lcon, get_ucon

import NLPModelsIpopt: ipopt

import ..BACKENDS
using Adapt

# ============================================================================
# Helper: build a standard test problem
# ============================================================================

function build_batch_model(; ns=2, nv=2, θ_val=[2.0])
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, θ_val)
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j] - θ[1] for j in 1:nv; lcon = 0.0)
    return ExaModel(c)
end

# ============================================================================
# Extract test logic into functions to avoid Julia 1.12 GC/compiler segfault
# ============================================================================

function test_construction()
    model = build_batch_model(ns=3)
    @test get_nbatch(model) == 3
    @test NLPModels.get_nvar(model) == 2
    @test NLPModels.get_ncon(model) == 2
    @test model isa NLPModels.AbstractNLPModel
    @test size(model.meta.x0) == (2, 3)
end

function test_obj()
    model = build_batch_model()
    bx = [1.0 3.0; 2.0 4.0]
    bf = zeros(2)
    obj!(model, bx, bf)
    @test bf[1] ≈ 10.0
    @test bf[2] ≈ 50.0
    @test obj(model, bx) ≈ bf
    flat = get_model(model)
    @test sum(bf) ≈ obj(flat, vec(bx))
end

function test_grad()
    model = build_batch_model()
    bx = [1.0 3.0; 2.0 4.0]
    bg = zeros(2, 2)
    grad!(model, bx, bg)
    @test bg[:, 1] ≈ [4.0, 8.0]
    @test bg[:, 2] ≈ [12.0, 16.0]
    g_flat = zeros(4)
    grad!(get_model(model), vec(bx), g_flat)
    @test vec(bg) ≈ g_flat
end

function test_cons()
    model = build_batch_model()
    bx = [1.0 3.0; 2.0 4.0]
    bc = zeros(2, 2)
    cons!(model, bx, bc)
    @test bc[:, 1] ≈ [-1.0, 0.0]
    @test bc[:, 2] ≈ [1.0, 2.0]
    c_flat = zeros(4)
    cons_nln!(get_model(model), vec(bx), c_flat)
    @test vec(bc) ≈ c_flat
end

function test_jac_hess()
    model = build_batch_model()
    ns, nv = 2, 2
    flat = get_model(model)
    bx = [1.0 3.0; 2.0 4.0]

    # --- Jacobian values ---
    nnzj = NLPModels.get_nnzj(model)
    jvals = zeros(nnzj, ns)
    jac_coord!(model, bx, jvals)
    jvals_flat = zeros(NLPModels.get_nnzj(flat))
    jac_coord!(flat, vec(bx), jvals_flat)
    @test vec(jvals) ≈ jvals_flat

    # --- Hessian values ---
    nnzh = NLPModels.get_nnzh(model)
    by = ones(nv, ns)
    hvals = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals)
    hvals_flat = zeros(NLPModels.get_nnzh(flat))
    hess_coord!(flat, vec(bx), vec(by), hvals_flat)
    @test vec(hvals) ≈ hvals_flat
end

function test_hess_obj_weight()
    ns, nv = 2, 2
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j]^2 for j in 1:nv)
    model = ExaModel(c)

    nc = NLPModels.get_ncon(model)
    nnzh = NLPModels.get_nnzh(model)
    bx = [1.0 3.0; 2.0 4.0]
    by = ones(nc, ns)
    flat = get_model(model)

    hvals_w1 = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals_w1; obj_weight = 1.0)
    hvals_flat_w1 = zeros(NLPModels.get_nnzh(flat))
    hess_coord!(flat, vec(bx), vec(by), hvals_flat_w1; obj_weight = 1.0)
    @test vec(hvals_w1) ≈ hvals_flat_w1

    hvals_w2 = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals_w2; obj_weight = 2.0)
    hvals_flat_w2 = zeros(NLPModels.get_nnzh(flat))
    hess_coord!(flat, vec(bx), vec(by), hvals_flat_w2; obj_weight = 2.0)
    @test vec(hvals_w2) ≈ hvals_flat_w2

    @test hvals_w1 != hvals_w2
end

function test_hess_vector_obj_weight()
    ns, nv = 2, 2
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j]^2 for j in 1:nv)
    model = ExaModel(c)

    nc = NLPModels.get_ncon(model)
    nnzh = NLPModels.get_nnzh(model)
    bx = [1.0 3.0; 2.0 4.0]
    by = ones(nc, ns)

    # Vector obj_weight = [w1, w2]
    wvec = [1.5, 3.0]
    hvals_vec = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals_vec; obj_weight = wvec)

    # Uniform scalar weights for comparison
    hvals_w1 = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals_w1; obj_weight = wvec[1])
    hvals_w2 = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals_w2; obj_weight = wvec[2])

    # With uniform weight, both instances get the same obj contribution.
    # With vector weight, instance 1 gets w1, instance 2 gets w2.
    # The constraint hessian is unaffected by obj_weight, so it is the same.
    # Check: vector result differs from both uniform-scalar results.
    @test hvals_vec != hvals_w1
    @test hvals_vec != hvals_w2

    # Verify consistency: uniform weight = special case of vector weight
    hvals_uniform = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals_uniform; obj_weight = [2.0, 2.0])
    hvals_scalar = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals_scalar; obj_weight = 2.0)
    @test hvals_uniform ≈ hvals_scalar
end

function test_multiple_constraints()
    ns, nv = 2, 3
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, [1.0])
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j] - θ[1] for j in 1:nv)
    c, _ = add_con(c, v[1] + v[2] + v[3] for _ in 1:1; ucon = 10.0)
    model = ExaModel(c)
    flat = get_model(model)

    nc = NLPModels.get_ncon(model)
    @test nc == nv + 1
    @test get_nbatch(model) == ns

    bx = reshape(Float64[1, 2, 3, 4, 5, 6], nv, ns)

    # cons!
    bc = zeros(nc, ns)
    cons!(model, bx, bc)
    c_flat = zeros(nc * ns)
    cons_nln!(flat, vec(bx), c_flat)
    @test vec(bc) ≈ c_flat

    # jac
    nnzj = NLPModels.get_nnzj(model)
    jvals = zeros(nnzj, ns)
    jac_coord!(model, bx, jvals)
    jvals_flat = zeros(NLPModels.get_nnzj(flat))
    jac_coord!(flat, vec(bx), jvals_flat)
    @test vec(jvals) ≈ jvals_flat

    # hess
    nnzh = NLPModels.get_nnzh(model)
    by = ones(nc, ns)
    hvals = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals)
    hvals_flat = zeros(NLPModels.get_nnzh(flat))
    hess_coord!(flat, vec(bx), vec(by), hvals_flat)
    @test vec(hvals) ≈ hvals_flat
end

function test_error_guards()
    model = build_batch_model()
    x_vec = ones(2)
    @test_throws ArgumentError obj(model, x_vec)
    @test_throws ArgumentError cons!(model, x_vec, zeros(2))
    @test_throws ArgumentError grad!(model, x_vec, zeros(2))
end

function test_bounds()
    ns, nv = 2, 2
    c = BatchExaCore(ns)
    @add_var(c, v, nv; start = 0.5, lvar = 0.0, uvar = 10.0)
    c, _ = add_obj(c, v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j] for j in 1:nv; lcon = 0.0, ucon = 100.0)
    model = ExaModel(c)

    @test size(model.meta.x0) == (nv, ns)
    @test model.meta.x0 ≈ fill(0.5, nv, ns)
    @test model.meta.lvar ≈ fill(0.0, nv, ns)
    @test model.meta.uvar ≈ fill(10.0, nv, ns)

    flat = get_model(model)
    @test NLPModels.get_nvar(flat) == nv * ns
    @test flat.meta.x0 ≈ fill(0.5, nv * ns)
    @test flat.meta.lvar ≈ fill(0.0, nv * ns)
    @test flat.meta.uvar ≈ fill(10.0, nv * ns)
end

function test_flatten_model()
    model = build_batch_model()
    flat = get_model(model)
    @test flat isa ExaModels.BatchNLPModels.FlattenNLPModel
    @test NLPModels.get_nvar(flat) == 2 * 2
    @test NLPModels.get_ncon(flat) == 2 * 2

    c = ExaCore(concrete = Val(true))
    c, x = add_var(c, 2)
    c, _ = add_obj(c, x[i]^2 for i in 1:2)
    m = ExaModel(c)
    @test get_model(m) === m
end

function test_ipopt_simple()
    ns, nv = 3, 1
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, (v[1] - θ[1])^2 for _ in 1:1)
    c, _ = add_con(c, v[1] for _ in 1:1; lcon = 0.0, ucon = Inf)
    model = ExaModel(c)

    result = ipopt(get_model(model); print_level = 0)
    @test result.status == :first_order
    for i in 1:ns
        @test result.solution[var_indices(model, i)] ≈ [2.0] atol = 1e-5
    end
    @test isapprox(result.objective, 0.0; atol = 1e-8)
end

function test_ipopt_multi()
    ns, nv = 2, 2
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, [1.0, 3.0])
    c, _ = add_obj(c, (v[j] - θ[j])^2 for j in 1:nv)
    c, _ = add_con(c, v[1] + v[2] for _ in 1:1; ucon = 10.0)
    model = ExaModel(c)

    result = ipopt(get_model(model); print_level = 0)
    @test result.status == :first_order
    @test result.solution[var_indices(model, 1)] ≈ [1.0, 3.0] atol = 1e-5
    @test result.solution[var_indices(model, 2)] ≈ [1.0, 3.0] atol = 1e-5
    @test isapprox(result.objective, 0.0; atol = 1e-8)
end

function test_set_parameter()
    ns, nv = 2, 1
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, (v[1] - θ[1])^2 for _ in 1:1)
    model = ExaModel(c)

    # Default: both instances share θ = [2.0]
    bx = reshape([1.0, 3.0], nv, ns)
    bf = zeros(ns)
    obj!(model, bx, bf)
    @test bf[1] ≈ 1.0  # (1-2)^2
    @test bf[2] ≈ 1.0  # (3-2)^2

    # Update parameters via set_value! on the model
    set_value!(model, θ, [10.0])
    bf2 = zeros(ns)
    obj!(model, bx, bf2)
    @test bf2[1] ≈ (1.0 - 10.0)^2
    @test bf2[2] ≈ (3.0 - 10.0)^2
end

function test_multidim_vars()
    ns = 2
    nh, nc = 3, 2  # multi-dimensional variable
    c = BatchExaCore(ns)
    @add_var(c, w, nh, nc; start = zeros(nh, nc))
    @add_par(c, θ, [1.0])
    c, _ = add_obj(c, w[i, j]^2 for i in 1:nh, j in 1:nc)
    model = ExaModel(c)

    @test NLPModels.get_nvar(model) == nh * nc
    @test get_nbatch(model) == ns

    bx = reshape(Float64.(1:(nh*nc*ns)), nh * nc, ns)
    bf = zeros(ns)
    obj!(model, bx, bf)
    @test bf[1] ≈ sum(bx[:, 1] .^ 2)
    @test bf[2] ≈ sum(bx[:, 2] .^ 2)
end

function test_add_con_aug()
    ns, nv = 2, 3
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, [1.0])
    c, _ = add_obj(c, v[j]^2 for j in 1:nv)
    # Create a constraint, then augment it
    @add_con(c, g, v[j] for j in 1:nv; lcon = -10.0, ucon = 10.0)
    @add_con!(c, g, j => θ[1] * v[j] for j in 1:nv)
    model = ExaModel(c)
    flat = get_model(model)

    nc = NLPModels.get_ncon(model)
    @test nc == nv

    bx = reshape(Float64.(1:(nv*ns)), nv, ns)

    # cons! — augmented constraint = v[j] + θ[1]*v[j] = 2*v[j]
    bc = zeros(nc, ns)
    cons!(model, bx, bc)
    c_flat = zeros(nc * ns)
    cons_nln!(flat, vec(bx), c_flat)
    @test vec(bc) ≈ c_flat

    # Each constraint value should be v[j] + 1.0*v[j] = 2*v[j]
    @test bc[:, 1] ≈ 2.0 .* bx[:, 1]
    @test bc[:, 2] ≈ 2.0 .* bx[:, 2]

    # jac
    nnzj = NLPModels.get_nnzj(model)
    jvals = zeros(nnzj, ns)
    jac_coord!(model, bx, jvals)
    jvals_flat = zeros(NLPModels.get_nnzj(flat))
    jac_coord!(flat, vec(bx), jvals_flat)
    @test vec(jvals) ≈ jvals_flat

    # hess
    nnzh = NLPModels.get_nnzh(model)
    by = ones(nc, ns)
    hvals = zeros(nnzh, ns)
    hess_coord!(model, bx, by, hvals)
    hvals_flat = zeros(NLPModels.get_nnzh(flat))
    hess_coord!(flat, vec(bx), vec(by), hvals_flat)
    @test vec(hvals) ≈ hvals_flat
end

function test_add_expr()
    ns, nv = 2, 3
    c = BatchExaCore(ns)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])

    # Create a subexpression and use it in objective and constraint
    @add_expr(c, s, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_obj(c, s[j] for j in 1:nv)
    c, _ = add_con(c, s[j] - v[j] for j in 1:nv; lcon = 0.0)
    model = ExaModel(c)
    flat = get_model(model)

    nc = NLPModels.get_ncon(model)
    @test nc == nv

    bx = reshape(Float64.(1:(nv*ns)), nv, ns)

    # obj — should be sum of θ[1]*v[j]^2 = 2*v[j]^2
    bf = zeros(ns)
    obj!(model, bx, bf)
    @test bf[1] ≈ 2.0 * sum(bx[:, 1] .^ 2)
    @test bf[2] ≈ 2.0 * sum(bx[:, 2] .^ 2)
    @test sum(bf) ≈ obj(flat, vec(bx))

    # cons — s[j] - v[j] = 2*v[j]^2 - v[j]
    bc = zeros(nc, ns)
    cons!(model, bx, bc)
    c_flat = zeros(nc * ns)
    cons_nln!(flat, vec(bx), c_flat)
    @test vec(bc) ≈ c_flat
    @test bc[:, 1] ≈ 2.0 .* bx[:, 1] .^ 2 .- bx[:, 1]

    # grad
    bg = zeros(nv, ns)
    grad!(model, bx, bg)
    g_flat = zeros(nv * ns)
    grad!(flat, vec(bx), g_flat)
    @test vec(bg) ≈ g_flat
end

function test_per_instance_accessors()
    ns, nv = 2, 3
    c = BatchExaCore(ns)
    @add_var(c, v, nv; start = 1.0, lvar = -5.0, uvar = 5.0)
    c, _ = add_obj(c, v[j]^2 for j in 1:nv)
    c, con = add_con(c, v[j] for j in 1:nv; lcon = 0.0, ucon = 10.0)
    model = ExaModel(c)

    # Test per-instance variable accessors
    @test get_start(model, v, 1) ≈ fill(1.0, nv)
    @test get_lvar(model, v, 1) ≈ fill(-5.0, nv)
    @test get_uvar(model, v, 2) ≈ fill(5.0, nv)

    # Test per-instance constraint accessors
    @test get_lcon(model, con, 1) ≈ fill(0.0, nv)
    @test get_ucon(model, con, 2) ≈ fill(10.0, nv)

    # Test cons_block_indices
    @test cons_block_indices(model, 1) == 1:nv
    @test cons_block_indices(model, 2) == (nv+1):(2*nv)
end

# ============================================================================
# Backend-parameterized tests — verify batch evaluation on all backends (incl. KA/GPU)
# ============================================================================

function test_batch_backend(backend)
    ns, nv = 2, 3
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    @add_con(c, g, v[j] - θ[1] for j in 1:nv; lcon = 0.0)
    @add_con!(c, g, j => θ[1] * v[j] for j in 1:nv)
    model = ExaModel(c)
    flat = get_model(model)

    nc = NLPModels.get_ncon(model)
    bx = ExaModels.convert_array(reshape(Float64.(1:(nv*ns)), nv, ns), backend)

    # obj
    bf_gpu = obj(model, bx)
    @test sum(Array(bf_gpu)) ≈ obj(flat, vec(bx))

    # grad
    bg = ExaModels.convert_array(zeros(nv, ns), backend)
    grad!(model, bx, bg)
    g_flat = ExaModels.convert_array(zeros(nv * ns), backend)
    grad!(flat, vec(bx), g_flat)
    @test vec(Array(bg)) ≈ Array(g_flat)

    # cons
    bc = ExaModels.convert_array(zeros(nc, ns), backend)
    cons!(model, bx, bc)
    c_flat = ExaModels.convert_array(zeros(nc * ns), backend)
    cons_nln!(flat, vec(bx), c_flat)
    @test vec(Array(bc)) ≈ Array(c_flat)

    # jac
    nnzj = NLPModels.get_nnzj(model)
    jvals = ExaModels.convert_array(zeros(nnzj, ns), backend)
    jac_coord!(model, bx, jvals)
    jvals_flat = ExaModels.convert_array(zeros(NLPModels.get_nnzj(flat)), backend)
    jac_coord!(flat, vec(bx), jvals_flat)
    @test vec(Array(jvals)) ≈ Array(jvals_flat)

    # hess
    nnzh = NLPModels.get_nnzh(model)
    by = ExaModels.convert_array(ones(nc, ns), backend)
    hvals = ExaModels.convert_array(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals)
    hvals_flat = ExaModels.convert_array(zeros(NLPModels.get_nnzh(flat)), backend)
    hess_coord!(flat, vec(bx), vec(by), hvals_flat)
    @test vec(Array(hvals)) ≈ Array(hvals_flat)
end

# ============================================================================

function runtests()
    return @testset "Batch ExaModel" begin
        @testset "Construction" test_construction()
        @testset "obj!" test_obj()
        @testset "grad!" test_grad()
        @testset "cons!" test_cons()
        @testset "jac and hess" test_jac_hess()
        @testset "hess obj_weight" test_hess_obj_weight()
        @testset "hess vector obj_weight" test_hess_vector_obj_weight()
        @testset "Multiple constraints" test_multiple_constraints()
        @testset "Error guards" test_error_guards()
        @testset "Bounds" test_bounds()
        @testset "flatten_model" test_flatten_model()
        @testset "Ipopt simple" test_ipopt_simple()
        @testset "Ipopt multi" test_ipopt_multi()
        @testset "set_value!" test_set_parameter()
        @testset "Multidim vars" test_multidim_vars()
        @testset "add_con!" test_add_con_aug()
        @testset "add_expr" test_add_expr()
        @testset "Per-instance accessors" test_per_instance_accessors()
        for backend in BACKENDS
            backend === nothing && continue
            @testset "Backend: $(typeof(backend))" test_batch_backend(backend)
        end
    end
end

end # module
