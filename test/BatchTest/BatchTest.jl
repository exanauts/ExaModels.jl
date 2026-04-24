module BatchTest

using Test
using ExaModels
import NLPModels
import NLPModels:
    obj, cons!, cons_nln!, grad!, jac_coord!, hess_coord!, jac_structure!,
    hess_structure!
import NLPModels: obj!
import ExaModels: var_indices, cons_block_indices, get_nbatch,
    get_start, get_lvar, get_uvar, get_lcon, get_ucon, WrapperNLPModel, FlatNLPModel

import NLPModelsIpopt: ipopt
import MadNLP
import MadNLP: madnlp, ERROR as MADNLP_ERROR
import PowerModels
import Downloads

import ..BACKENDS
using Adapt

# ============================================================================
# Helper: build a standard test problem
# ============================================================================

function build_batch_model(; ns=2, nv=2, θ_val=[2.0], backend = nothing)
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, θ_val)
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j] - θ[1] for j in 1:nv; lcon = 0.0)
    return ExaModel(c)
end

# ============================================================================
# Helper utilities for backend-aware allocation
# ============================================================================

_ca(x, ::Nothing) = x
_ca(x, backend) = ExaModels.convert_array(x, backend)
_to_cpu(x) = Array(x)
_to_cpu(x::Array) = x

# ============================================================================
# Test functions — all accept backend parameter
# ============================================================================

function test_construction(; backend = nothing)
    model = build_batch_model(ns=3; backend)
    @test get_nbatch(model) == 3
    @test NLPModels.get_nvar(model) == 2
    @test NLPModels.get_ncon(model) == 2
    @test model isa NLPModels.AbstractNLPModel
    @test size(model.meta.x0) == (2, 3)
end

function test_obj(; backend = nothing)
    model = build_batch_model(; backend)
    bx = _ca([1.0 3.0; 2.0 4.0], backend)
    bf = _ca(zeros(2), backend)
    obj!(model, bx, bf)
    @test _to_cpu(bf)[1] ≈ 10.0
    @test _to_cpu(bf)[2] ≈ 50.0
    @test _to_cpu(obj(model, bx)) ≈ _to_cpu(bf)
    flat = FlatNLPModel(model)
    @test sum(_to_cpu(bf)) ≈ obj(flat, vec(bx))
end

function test_grad(; backend = nothing)
    model = build_batch_model(; backend)
    bx = _ca([1.0 3.0; 2.0 4.0], backend)
    bg = _ca(zeros(2, 2), backend)
    grad!(model, bx, bg)
    bg_cpu = _to_cpu(bg)
    @test bg_cpu[:, 1] ≈ [4.0, 8.0]
    @test bg_cpu[:, 2] ≈ [12.0, 16.0]
    flat = FlatNLPModel(model)
    g_flat = _ca(zeros(4), backend)
    grad!(flat, vec(bx), g_flat)
    @test vec(bg_cpu) ≈ _to_cpu(g_flat)
end

function test_cons(; backend = nothing)
    model = build_batch_model(; backend)
    bx = _ca([1.0 3.0; 2.0 4.0], backend)
    bc = _ca(zeros(2, 2), backend)
    cons!(model, bx, bc)
    bc_cpu = _to_cpu(bc)
    @test bc_cpu[:, 1] ≈ [-1.0, 0.0]
    @test bc_cpu[:, 2] ≈ [1.0, 2.0]
    flat = FlatNLPModel(model)
    c_flat = _ca(zeros(4), backend)
    cons_nln!(flat, vec(bx), c_flat)
    @test vec(bc_cpu) ≈ _to_cpu(c_flat)
end

function test_jac_hess(; backend = nothing)
    model = build_batch_model(; backend)
    ns, nv = 2, 2
    flat = FlatNLPModel(model)
    bx = _ca([1.0 3.0; 2.0 4.0], backend)

    # --- Jacobian values ---
    nnzj = NLPModels.get_nnzj(model)
    jvals = _ca(zeros(nnzj, ns), backend)
    jac_coord!(model, bx, jvals)
    jvals_flat = _ca(zeros(NLPModels.get_nnzj(flat)), backend)
    jac_coord!(flat, vec(bx), jvals_flat)
    @test vec(_to_cpu(jvals)) ≈ _to_cpu(jvals_flat)

    # --- Hessian values ---
    nnzh = NLPModels.get_nnzh(model)
    by = _ca(ones(nv, ns), backend)
    hvals = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals)
    hvals_flat = _ca(zeros(NLPModels.get_nnzh(flat)), backend)
    hess_coord!(flat, vec(bx), vec(by), hvals_flat)
    @test vec(_to_cpu(hvals)) ≈ _to_cpu(hvals_flat)
end

function test_hess_obj_weight(; backend = nothing)
    ns, nv = 2, 2
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j]^2 for j in 1:nv)
    model = ExaModel(c)

    nc = NLPModels.get_ncon(model)
    nnzh = NLPModels.get_nnzh(model)
    bx = _ca([1.0 3.0; 2.0 4.0], backend)
    by = _ca(ones(nc, ns), backend)
    flat = FlatNLPModel(model)

    hvals_w1 = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals_w1; obj_weight = 1.0)
    hvals_flat_w1 = _ca(zeros(NLPModels.get_nnzh(flat)), backend)
    hess_coord!(flat, vec(bx), vec(by), hvals_flat_w1; obj_weight = 1.0)
    @test vec(_to_cpu(hvals_w1)) ≈ _to_cpu(hvals_flat_w1)

    hvals_w2 = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals_w2; obj_weight = 2.0)
    hvals_flat_w2 = _ca(zeros(NLPModels.get_nnzh(flat)), backend)
    hess_coord!(flat, vec(bx), vec(by), hvals_flat_w2; obj_weight = 2.0)
    @test vec(_to_cpu(hvals_w2)) ≈ _to_cpu(hvals_flat_w2)

    @test _to_cpu(hvals_w1) != _to_cpu(hvals_w2)
end

function test_hess_vector_obj_weight(; backend = nothing)
    ns, nv = 2, 2
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j]^2 for j in 1:nv)
    model = ExaModel(c)

    nc = NLPModels.get_ncon(model)
    nnzh = NLPModels.get_nnzh(model)
    bx = _ca([1.0 3.0; 2.0 4.0], backend)
    by = _ca(ones(nc, ns), backend)

    # Vector obj_weight = [w1, w2]
    wvec = _ca([1.5, 3.0], backend)
    hvals_vec = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals_vec; obj_weight = wvec)

    # Uniform scalar weights for comparison
    hvals_w1 = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals_w1; obj_weight = 1.5)
    hvals_w2 = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals_w2; obj_weight = 3.0)

    @test _to_cpu(hvals_vec) != _to_cpu(hvals_w1)
    @test _to_cpu(hvals_vec) != _to_cpu(hvals_w2)

    # Verify consistency: uniform weight = special case of vector weight
    hvals_uniform = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals_uniform; obj_weight = _ca([2.0, 2.0], backend))
    hvals_scalar = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals_scalar; obj_weight = 2.0)
    @test _to_cpu(hvals_uniform) ≈ _to_cpu(hvals_scalar)
end

function test_multiple_constraints(; backend = nothing)
    ns, nv = 2, 3
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [1.0])
    c, _ = add_obj(c, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j] - θ[1] for j in 1:nv)
    c, _ = add_con(c, v[1] + v[2] + v[3] for _ in 1:1; ucon = 10.0)
    model = ExaModel(c)
    flat = FlatNLPModel(model)

    nc = NLPModels.get_ncon(model)
    @test nc == nv + 1
    @test get_nbatch(model) == ns

    bx = _ca(reshape(Float64[1, 2, 3, 4, 5, 6], nv, ns), backend)

    # cons!
    bc = _ca(zeros(nc, ns), backend)
    cons!(model, bx, bc)
    c_flat = _ca(zeros(nc * ns), backend)
    cons_nln!(flat, vec(bx), c_flat)
    @test vec(_to_cpu(bc)) ≈ _to_cpu(c_flat)

    # jac
    nnzj = NLPModels.get_nnzj(model)
    jvals = _ca(zeros(nnzj, ns), backend)
    jac_coord!(model, bx, jvals)
    jvals_flat = _ca(zeros(NLPModels.get_nnzj(flat)), backend)
    jac_coord!(flat, vec(bx), jvals_flat)
    @test vec(_to_cpu(jvals)) ≈ _to_cpu(jvals_flat)

    # hess
    nnzh = NLPModels.get_nnzh(model)
    by = _ca(ones(nc, ns), backend)
    hvals = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals)
    hvals_flat = _ca(zeros(NLPModels.get_nnzh(flat)), backend)
    hess_coord!(flat, vec(bx), vec(by), hvals_flat)
    @test vec(_to_cpu(hvals)) ≈ _to_cpu(hvals_flat)
end

function test_error_guards(; backend = nothing)
    model = build_batch_model(; backend)
    x_vec = _ca(ones(2), backend)
    @test_throws ArgumentError obj(model, x_vec)
    @test_throws ArgumentError cons!(model, x_vec, _ca(zeros(2), backend))
    @test_throws ArgumentError grad!(model, x_vec, _ca(zeros(2), backend))
end

function test_bounds(; backend = nothing)
    ns, nv = 2, 2
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv; start = 0.5, lvar = 0.0, uvar = 10.0)
    c, _ = add_obj(c, v[j]^2 for j in 1:nv)
    c, _ = add_con(c, v[j] for j in 1:nv; lcon = 0.0, ucon = 100.0)
    model = ExaModel(c)

    @test size(model.meta.x0) == (nv, ns)
    @test _to_cpu(model.meta.x0) ≈ fill(0.5, nv, ns)
    @test _to_cpu(model.meta.lvar) ≈ fill(0.0, nv, ns)
    @test _to_cpu(model.meta.uvar) ≈ fill(10.0, nv, ns)

    flat = FlatNLPModel(model)
    @test NLPModels.get_nvar(flat) == nv * ns
    @test _to_cpu(flat.meta.x0) ≈ fill(0.5, nv * ns)
    @test _to_cpu(flat.meta.lvar) ≈ fill(0.0, nv * ns)
    @test _to_cpu(flat.meta.uvar) ≈ fill(10.0, nv * ns)
end

function test_flatten_model(; backend = nothing)
    model = build_batch_model(; backend)
    flat = FlatNLPModel(model)
    @test flat isa FlatNLPModel
    @test NLPModels.get_nvar(flat) == 2 * 2
    @test NLPModels.get_ncon(flat) == 2 * 2
end

function test_ipopt_simple(; backend = nothing)
    ns, nv = 3, 1
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, (v[1] - θ[1])^2 for _ in 1:1)
    c, _ = add_con(c, v[1] for _ in 1:1; lcon = 0.0, ucon = Inf)
    model = ExaModel(c)

    result = ipopt(WrapperNLPModel(FlatNLPModel(model)); print_level = 0)
    @test result.status == :first_order
    for i in 1:ns
        @test result.solution[var_indices(model, i)] ≈ [2.0] atol = 1e-5
    end
    @test isapprox(result.objective, 0.0; atol = 1e-8)
end

function test_ipopt_multi(; backend = nothing)
    ns, nv = 2, 2
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [1.0, 3.0])
    c, _ = add_obj(c, (v[j] - θ[j])^2 for j in 1:nv)
    c, _ = add_con(c, v[1] + v[2] for _ in 1:1; ucon = 10.0)
    model = ExaModel(c)

    result = ipopt(WrapperNLPModel(FlatNLPModel(model)); print_level = 0)
    @test result.status == :first_order
    @test result.solution[var_indices(model, 1)] ≈ [1.0, 3.0] atol = 1e-5
    @test result.solution[var_indices(model, 2)] ≈ [1.0, 3.0] atol = 1e-5
    @test isapprox(result.objective, 0.0; atol = 1e-8)
end

function test_set_parameter(; backend = nothing)
    ns, nv = 2, 1
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])
    c, _ = add_obj(c, (v[1] - θ[1])^2 for _ in 1:1)
    model = ExaModel(c)

    # Default: both instances share θ = [2.0]
    bx = _ca(reshape([1.0, 3.0], nv, ns), backend)
    bf = _ca(zeros(ns), backend)
    obj!(model, bx, bf)
    bf_cpu = _to_cpu(bf)
    @test bf_cpu[1] ≈ 1.0  # (1-2)^2
    @test bf_cpu[2] ≈ 1.0  # (3-2)^2

    # Update parameters via set_value! on the model
    set_value!(model, θ, [10.0])
    bf2 = _ca(zeros(ns), backend)
    obj!(model, bx, bf2)
    bf2_cpu = _to_cpu(bf2)
    @test bf2_cpu[1] ≈ (1.0 - 10.0)^2
    @test bf2_cpu[2] ≈ (3.0 - 10.0)^2
end

function test_multidim_vars(; backend = nothing)
    ns = 2
    nh, nc = 3, 2  # multi-dimensional variable
    c = BatchExaCore(ns; backend)
    @add_var(c, w, nh, nc; start = zeros(nh, nc))
    @add_par(c, θ, [1.0])
    c, _ = add_obj(c, w[i, j]^2 for i in 1:nh, j in 1:nc)
    model = ExaModel(c)

    @test NLPModels.get_nvar(model) == nh * nc
    @test get_nbatch(model) == ns

    bx = _ca(reshape(Float64.(1:(nh*nc*ns)), nh * nc, ns), backend)
    bf = _ca(zeros(ns), backend)
    obj!(model, bx, bf)
    bx_cpu = _to_cpu(bx)
    bf_cpu = _to_cpu(bf)
    @test bf_cpu[1] ≈ sum(bx_cpu[:, 1] .^ 2)
    @test bf_cpu[2] ≈ sum(bx_cpu[:, 2] .^ 2)
end

function test_add_con_aug(; backend = nothing)
    ns, nv = 2, 3
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [1.0])
    c, _ = add_obj(c, v[j]^2 for j in 1:nv)
    # Create a constraint, then augment it
    @add_con(c, g, v[j] for j in 1:nv; lcon = -10.0, ucon = 10.0)
    @add_con!(c, g, j => θ[1] * v[j] for j in 1:nv)
    model = ExaModel(c)
    flat = FlatNLPModel(model)

    nc = NLPModels.get_ncon(model)
    @test nc == nv

    bx = _ca(reshape(Float64.(1:(nv*ns)), nv, ns), backend)

    # cons! — augmented constraint = v[j] + θ[1]*v[j] = 2*v[j]
    bc = _ca(zeros(nc, ns), backend)
    cons!(model, bx, bc)
    c_flat = _ca(zeros(nc * ns), backend)
    cons_nln!(flat, vec(bx), c_flat)
    bc_cpu = _to_cpu(bc)
    bx_cpu = _to_cpu(bx)
    @test vec(bc_cpu) ≈ _to_cpu(c_flat)

    # Each constraint value should be v[j] + 1.0*v[j] = 2*v[j]
    @test bc_cpu[:, 1] ≈ 2.0 .* bx_cpu[:, 1]
    @test bc_cpu[:, 2] ≈ 2.0 .* bx_cpu[:, 2]

    # jac
    nnzj = NLPModels.get_nnzj(model)
    jvals = _ca(zeros(nnzj, ns), backend)
    jac_coord!(model, bx, jvals)
    jvals_flat = _ca(zeros(NLPModels.get_nnzj(flat)), backend)
    jac_coord!(flat, vec(bx), jvals_flat)
    @test vec(_to_cpu(jvals)) ≈ _to_cpu(jvals_flat)

    # hess
    nnzh = NLPModels.get_nnzh(model)
    by = _ca(ones(nc, ns), backend)
    hvals = _ca(zeros(nnzh, ns), backend)
    hess_coord!(model, bx, by, hvals)
    hvals_flat = _ca(zeros(NLPModels.get_nnzh(flat)), backend)
    hess_coord!(flat, vec(bx), vec(by), hvals_flat)
    @test vec(_to_cpu(hvals)) ≈ _to_cpu(hvals_flat)
end

function test_add_expr(; backend = nothing)
    ns, nv = 2, 3
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv)
    @add_par(c, θ, [2.0])

    # Create a subexpression and use it in objective and constraint
    @add_expr(c, s, θ[1] * v[j]^2 for j in 1:nv)
    c, _ = add_obj(c, s[j] for j in 1:nv)
    c, _ = add_con(c, s[j] - v[j] for j in 1:nv; lcon = 0.0)
    model = ExaModel(c)
    flat = FlatNLPModel(model)

    nc = NLPModels.get_ncon(model)
    @test nc == nv

    bx = _ca(reshape(Float64.(1:(nv*ns)), nv, ns), backend)

    # obj — should be sum of θ[1]*v[j]^2 = 2*v[j]^2
    bf = _ca(zeros(ns), backend)
    obj!(model, bx, bf)
    bx_cpu = _to_cpu(bx)
    bf_cpu = _to_cpu(bf)
    @test bf_cpu[1] ≈ 2.0 * sum(bx_cpu[:, 1] .^ 2)
    @test bf_cpu[2] ≈ 2.0 * sum(bx_cpu[:, 2] .^ 2)
    @test sum(bf_cpu) ≈ obj(flat, vec(bx))

    # cons — s[j] - v[j] = 2*v[j]^2 - v[j]
    bc = _ca(zeros(nc, ns), backend)
    cons!(model, bx, bc)
    c_flat = _ca(zeros(nc * ns), backend)
    cons_nln!(flat, vec(bx), c_flat)
    bc_cpu = _to_cpu(bc)
    @test vec(bc_cpu) ≈ _to_cpu(c_flat)
    @test bc_cpu[:, 1] ≈ 2.0 .* bx_cpu[:, 1] .^ 2 .- bx_cpu[:, 1]

    # grad
    bg = _ca(zeros(nv, ns), backend)
    grad!(model, bx, bg)
    g_flat = _ca(zeros(nv * ns), backend)
    grad!(flat, vec(bx), g_flat)
    @test vec(_to_cpu(bg)) ≈ _to_cpu(g_flat)
end

function _get_opf_case(filename)
    isfile(filename) && return filename
    tmpdir = tempname()
    mkdir(tmpdir)
    ff = joinpath(tmpdir, filename)
    Downloads.download(
        "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3/$filename",
        ff,
    )
    return ff
end

function _parse_opf_data(filename)
    data = PowerModels.parse_file(_get_opf_case(filename))
    PowerModels.standardize_cost_terms!(data, order = 2)
    PowerModels.calc_thermal_limits!(data)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    arcdict    = Dict(a => k for (k, a) in enumerate(ref[:arcs]))
    busdict    = Dict(k => i for (i, (k, v)) in enumerate(ref[:bus]))
    branchdict = Dict(k => i for (i, (k, v)) in enumerate(ref[:branch]))

    bus = [
        begin
            bus_loads  = [ref[:load][l]  for l in ref[:bus_loads][k]]
            bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][k]]
            pd = sum(load["pd"] for load in bus_loads;  init = 0.0)
            gs = sum(sh["gs"]   for sh   in bus_shunts; init = 0.0)
            qd = sum(load["qd"] for load in bus_loads;  init = 0.0)
            bs = sum(sh["bs"]   for sh   in bus_shunts; init = 0.0)
            (i = busdict[k], pd = pd, gs = gs, qd = qd, bs = bs)
        end for (k, v) in ref[:bus]
    ]
    gen = [
        (i = gendict_i, cost1 = v["cost"][1], cost2 = v["cost"][2], cost3 = v["cost"][3],
         bus = busdict[v["gen_bus"]])
        for (gendict_i, (k, v)) in enumerate(ref[:gen])
    ]
    arc = [
        (i = k, rate_a = ref[:branch][arc_l]["rate_a"], bus = busdict[arc_i])
        for (k, (arc_l, arc_i, arc_j)) in enumerate(ref[:arcs])
    ]
    branch = [
        begin
            g, b = PowerModels.calc_branch_y(br)
            tr, ti = PowerModels.calc_branch_t(br)
            ttm = tr^2 + ti^2
            (
                i = branchdict[bi], j = 1,
                f_idx = arcdict[bi, br["f_bus"], br["t_bus"]],
                t_idx = arcdict[bi, br["t_bus"], br["f_bus"]],
                f_bus = busdict[br["f_bus"]], t_bus = busdict[br["t_bus"]],
                c1 = (-g*tr - b*ti)/ttm, c2 = (-b*tr + g*ti)/ttm,
                c3 = (-g*tr + b*ti)/ttm, c4 = (-b*tr - g*ti)/ttm,
                c5 = (g + br["g_fr"])/ttm, c6 = (b + br["b_fr"])/ttm,
                c7 = (g + br["g_to"]),     c8 = (b + br["b_to"]),
                rate_a_sq = br["rate_a"]^2,
            )
        end for (bi, br) in ref[:branch]
    ]
    return (
        bus      = bus,
        gen      = gen,
        arc      = arc,
        branch   = branch,
        ref_buses = [busdict[i] for (i, _) in ref[:ref_buses]],
        vmax = [v["vmax"] for (k, v) in ref[:bus]],
        vmin = [v["vmin"] for (k, v) in ref[:bus]],
        pmax = [v["pmax"] for (k, v) in ref[:gen]],
        pmin = [v["pmin"] for (k, v) in ref[:gen]],
        qmax = [v["qmax"] for (k, v) in ref[:gen]],
        qmin = [v["qmin"] for (k, v) in ref[:gen]],
        rate_a  = [ref[:branch][arc_l]["rate_a"] for (arc_l, arc_i, arc_j) in ref[:arcs]],
        angmax  = [b["angmax"] for (k, b) in ref[:branch]],
        angmin  = [b["angmin"] for (k, b) in ref[:branch]],
    )
end

function _build_batch_opf(data; backend, nbatch)
    core = BatchExaCore(nbatch; backend)
    @add_var(core, va, length(data.bus))
    @add_var(core, vm, length(data.bus);
             start = fill!(similar(data.bus, Float64), 1.0),
             lvar = data.vmin, uvar = data.vmax)
    @add_var(core, pg, length(data.gen); lvar = data.pmin, uvar = data.pmax)
    @add_var(core, qg, length(data.gen); lvar = data.qmin, uvar = data.qmax)
    @add_var(core, p,  length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    @add_var(core, q,  length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    @add_obj(core, g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3 for g in data.gen)

    @add_con(core, c_ref_angle, va[i] for i in data.ref_buses)
    @add_con(core, c_from_p,
             p[b.f_idx] - b.c5*vm[b.f_bus]^2 -
             b.c3*(vm[b.f_bus]*vm[b.t_bus]*cos(va[b.f_bus]-va[b.t_bus])) -
             b.c4*(vm[b.f_bus]*vm[b.t_bus]*sin(va[b.f_bus]-va[b.t_bus]))
             for b in data.branch)
    @add_con(core, c_from_q,
             q[b.f_idx] + b.c6*vm[b.f_bus]^2 +
             b.c4*(vm[b.f_bus]*vm[b.t_bus]*cos(va[b.f_bus]-va[b.t_bus])) -
             b.c3*(vm[b.f_bus]*vm[b.t_bus]*sin(va[b.f_bus]-va[b.t_bus]))
             for b in data.branch)
    @add_con(core, c_to_p,
             p[b.t_idx] - b.c7*vm[b.t_bus]^2 -
             b.c1*(vm[b.t_bus]*vm[b.f_bus]*cos(va[b.t_bus]-va[b.f_bus])) -
             b.c2*(vm[b.t_bus]*vm[b.f_bus]*sin(va[b.t_bus]-va[b.f_bus]))
             for b in data.branch)
    @add_con(core, c_to_q,
             q[b.t_idx] + b.c8*vm[b.t_bus]^2 +
             b.c2*(vm[b.t_bus]*vm[b.f_bus]*cos(va[b.t_bus]-va[b.f_bus])) -
             b.c1*(vm[b.t_bus]*vm[b.f_bus]*sin(va[b.t_bus]-va[b.f_bus]))
             for b in data.branch)
    @add_con(core, c_angle_diff,
             va[b.f_bus] - va[b.t_bus] for b in data.branch;
             lcon = data.angmin, ucon = data.angmax)
    @add_con(core, c_thermal_f,
             p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in data.branch;
             lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf))
    @add_con(core, c_thermal_t,
             p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in data.branch;
             lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf))
    @add_con(core, c_p_balance, b.pd + b.gs*vm[b.i]^2 for b in data.bus)
    @add_con(core, c_q_balance, b.qd - b.bs*vm[b.i]^2 for b in data.bus)
    @add_con!(core, c_p_balance, a.bus => p[a.i]    for a in data.arc)
    @add_con!(core, c_q_balance, a.bus => q[a.i]    for a in data.arc)
    @add_con!(core, c_p_balance, g.bus => -pg[g.i]  for g in data.gen)
    @add_con!(core, c_q_balance, g.bus => -qg[g.i]  for g in data.gen)

    return FlatNLPModel(ExaModel(core; prod = true))
end

function test_batch_opf_flat(; backend = nothing)
    data = _parse_opf_data("pglib_opf_case3_lmbd.m")
    m = _build_batch_opf(data; backend, nbatch = 3)
    result = madnlp(m; print_level = MADNLP_ERROR, fixed_variable_treatment = MadNLP.RelaxBound)
    @test result.status == MadNLP.SOLVE_SUCCEEDED
end

function test_per_instance_accessors(; backend = nothing)
    ns, nv = 2, 3
    c = BatchExaCore(ns; backend)
    @add_var(c, v, nv; start = 1.0, lvar = -5.0, uvar = 5.0)
    c, _ = add_obj(c, v[j]^2 for j in 1:nv)
    c, con = add_con(c, v[j] for j in 1:nv; lcon = 0.0, ucon = 10.0)
    model = ExaModel(c)

    # Test per-instance variable accessors
    @test _to_cpu(get_start(model, v, 1)) ≈ fill(1.0, nv)
    @test _to_cpu(get_lvar(model, v, 1)) ≈ fill(-5.0, nv)
    @test _to_cpu(get_uvar(model, v, 2)) ≈ fill(5.0, nv)

    # Test per-instance constraint accessors
    @test _to_cpu(get_lcon(model, con, 1)) ≈ fill(0.0, nv)
    @test _to_cpu(get_ucon(model, con, 2)) ≈ fill(10.0, nv)

    # Test cons_block_indices
    @test cons_block_indices(model, 1) == 1:nv
    @test cons_block_indices(model, 2) == (nv+1):(2*nv)
end

# ============================================================================

function runtests()
    return @testset "Batch ExaModel" begin
        for backend in BACKENDS
            @testset "Backend: $(something(backend, :CPU))" begin
                @testset "Construction" test_construction(; backend)
                @testset "obj!" test_obj(; backend)
                @testset "grad!" test_grad(; backend)
                @testset "cons!" test_cons(; backend)
                @testset "jac and hess" test_jac_hess(; backend)
                @testset "hess obj_weight" test_hess_obj_weight(; backend)
                @testset "hess vector obj_weight" test_hess_vector_obj_weight(; backend)
                @testset "Multiple constraints" test_multiple_constraints(; backend)
                @testset "Error guards" test_error_guards(; backend)
                @testset "Bounds" test_bounds(; backend)
                @testset "flatten_model" test_flatten_model(; backend)
                @testset "Ipopt simple" test_ipopt_simple(; backend)
                @testset "Ipopt multi" test_ipopt_multi(; backend)
                @testset "set_value!" test_set_parameter(; backend)
                @testset "Multidim vars" test_multidim_vars(; backend)
                @testset "add_con!" test_add_con_aug(; backend)
                @testset "add_expr" test_add_expr(; backend)
                @testset "Per-instance accessors" test_per_instance_accessors(; backend)
                @testset "Batch OPF FlatNLPModel" test_batch_opf_flat(; backend)
            end
        end
    end
end

end # module
