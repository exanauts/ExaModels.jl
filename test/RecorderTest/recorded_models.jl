# Recorder-side builders for the expanded test set. Each mirrors an
# independently-implemented direct builder (LuksanVlcekBenchmark's ExaModels
# extension, or test/NLPTest's models), expressed against (tape, data).
#
# Transcription idioms (documented in docs/design/recorder.md):
# - Single-expression constraints/objectives become 1-element generators
#   (`expr for _ in 1:1`), matching the real API's own `pars = 1:1` semantics.
# - A structural scalar used inside an expression (`constraint(x, n)` at the
#   boundary) is injected through the iterable: `for n in data.N:data.N`.
# - An instance scalar computed from data (`h = 1/(N+1)`) becomes a parameter:
#   `add_par(c, 1; value = 1/(data.N + 1))`.

# ── LuksanVlcekBenchmark set (schema: (; N)) ─────────────────────────────────

const LV_TEMPLATE = (; N = 20)

lv_case(name, tape, sizes = (50, 300)) = (name = name, tape = tape, sizes = sizes)

const LV_CASES = [
    lv_case("rosenrock", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = (LVB.rosenrock_start(i) for i = 1:data.N))
        @add_con(c, LVB.rosenrock_constraint(x, i) for i = 1:data.N-2)
        @add_obj(c, LVB.rosenrock_objective(x, i) for i = 1:data.N-1)
        c
    end),
    lv_case("wood", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = (LVB.wood_start(i) for i = 1:data.N))
        con = @add_con(c, LVB.wood_constraint(x, k) for k in 1:data.N-7)
        @add_con!(c, con, k => LVB.wood_constraint_aug(x, k) for k in 1:data.N-7)
        @add_obj(c, LVB.wood_objective(x, i) for i = 1:data.N÷2-1)
        c
    end),
    lv_case("chained_powell", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = (LVB.chained_powell_start(i) for i = 1:data.N))
        @add_con(c, LVB.chained_powell_constraint1(x) for _ in 1:1)
        @add_con(c, LVB.chained_powell_constraint2(x, n) for n in data.N:data.N)
        @add_obj(c, LVB.chained_powell_objective(x, i) for i = 1:data.N÷2-1)
        c
    end),
    lv_case("cragg_levy", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = (LVB.cragg_levy_start(i) for i = 1:data.N))
        @add_con(c, LVB.cragg_levy_constraint(x, k) for k = 1:data.N-2)
        @add_obj(c, LVB.cragg_levy_objective(x, i) for i = 1:data.N÷2-1)
        c
    end),
    lv_case("broyden_tridiagonal", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = fill(-1, data.N))
        @add_con(c, LVB.broyden_tridiagonal_constraint(x, k) for k = 1:data.N-4)
        @add_obj(c, LVB.broyden_tridiagonal_objective(x, i) for i = 2:data.N-1)
        @add_obj(c, LVB.broyden_tridiagonal_objective_boundary(x, n) for n in data.N:data.N)
        c
    end),
    lv_case("broyden_banded", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = fill(3, data.N))
        @add_var(c, y, data.N; start = fill(0, data.N))
        @add_con(c, LVB.broyden_banded_kconstraint(x, k) for k = 1:data.N÷2)
        @add_con(c, y[1] - LVB.broyden_banded_xi(x, 1) - LVB.broyden_banded_xi(x, 2) for _ in 1:1)
        @add_con(c, y[2] - LVB.broyden_banded_xi(x, 2) - LVB.broyden_banded_xi(x, 3) - LVB.broyden_banded_xi(x, 1) for _ in 1:1)
        @add_con(c, y[3] - LVB.broyden_banded_xi(x, 2) - LVB.broyden_banded_xi(x, 3) - LVB.broyden_banded_xi(x, 1) - LVB.broyden_banded_xi(x, 4) for _ in 1:1)
        @add_con(c, y[4] - LVB.broyden_banded_xi(x, 2) - LVB.broyden_banded_xi(x, 3) - LVB.broyden_banded_xi(x, 1) - LVB.broyden_banded_xi(x, 4) - LVB.broyden_banded_xi(x, 5) for _ in 1:1)
        @add_con(c, y[5] - LVB.broyden_banded_xi(x, 2) - LVB.broyden_banded_xi(x, 3) - LVB.broyden_banded_xi(x, 1) - LVB.broyden_banded_xi(x, 4) - LVB.broyden_banded_xi(x, 5) - LVB.broyden_banded_xi(x, 6) for _ in 1:1)
        @add_con(c, LVB.broyden_banded_yconstraint(y, x, i) for i in 6:data.N-1)
        @add_obj(c, LVB.broyden_banded_objective(x, y, i) for i in 1:data.N)
        c
    end),
    lv_case("trigo_tridiagonal", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = fill(1, data.N))
        @add_con(c, LVB.trigo_tridiagonal_constraint1(x) for _ in 1:1)
        @add_con(c, LVB.trigo_tridiagonal_constraint2(x) for _ in 1:1)
        @add_con(c, LVB.trigo_tridiagonal_constraint3(x, n) for n in data.N:data.N)
        @add_con(c, LVB.trigo_tridiagonal_constraint4(x, n) for n in data.N:data.N)
        @add_obj(c, LVB.trigo_tridiagonal_objective(x, i) for i = 2:data.N-1)
        c
    end),
    lv_case("generalized_brown", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = fill(-1, data.N))
        @add_con(c, LVB.generalized_brown_constraint(x, k) for k = 1:data.N-2)
        @add_obj(c, LVB.generalized_brown_objective(x, i) for i = 1:data.N÷2)
        c
    end),
    lv_case("modified_brown", () -> record(LV_TEMPLATE) do c, data
        @add_var(c, x, data.N; start = fill(-1, data.N))
        @add_con(c, LVB.modified_brown_constraint1(x) for _ in 1:1)
        @add_con(c, LVB.modified_brown_constraint2(x) for _ in 1:1)
        @add_con(c, LVB.modified_brown_constraint3(x) for _ in 1:1)
        @add_con(c, LVB.modified_brown_constraint4(x, n) for n in data.N:data.N)
        @add_con(c, LVB.modified_brown_constraint5(x, n) for n in data.N:data.N)
        @add_con(c, LVB.modified_brown_constraint6(x, n) for n in data.N:data.N)
        @add_obj(c, LVB.modified_brown_objective(x, i) for i = 1:data.N÷2)
        c
    end),
    # h = 1/(N+1) is an instance scalar used inside expressions; recorded as a
    # parameter (values match the direct model, where h is baked as a constant;
    # expression trees differ, so this case is compared densely).
    (name = "augmented_lagrangian", sizes = (50, 300),
     dense = true,
     tape = () -> record(LV_TEMPLATE) do c, data
        c, h = add_par(c, 1; value = 1 / (data.N + 1))
        @add_var(c, x, data.N; start = (LVB.augmented_lagrangian_start(i) for i = 1:data.N))
        @add_con(c, LVB.augmented_lagrangian_constraint(x, h[1], k) for k = 1:data.N-2)
        @add_obj(c, LVB.augmented_lagrangian_objective(
            x, LVB.augmented_lagrangian_l1, LVB.augmented_lagrangian_l2,
            LVB.augmented_lagrangian_l3, i) for i = 1:data.N÷5)
        c
    end),
]

for hs in (:Chained_HS46, :Chained_HS48, :Chained_HS49)
    con1 = Symbol(hs, :_constraint1); con2 = Symbol(hs, :_constraint2)
    st = Symbol(hs, :_start); ob = Symbol(hs, :_objective)
    @eval push!(LV_CASES, lv_case($(string(hs)), () -> record(LV_TEMPLATE) do c, data
        nC = 2 * (data.N - 2) ÷ 3
        It_L1 = [3 * div(i-1, 2) for i in 1:2:nC-($(hs == :Chained_HS46 ? 2 : 1))]
        It_L2 = [3 * div(i-1, 2) for i in 2:2:nC]
        @add_var(c, x, data.N; start = (LVB.$st(i) for i = 1:data.N))
        @add_con(c, LVB.$con1(x, l) for l in $(hs == :Chained_HS46 ? :It_L2 : :It_L1))
        @add_con(c, LVB.$con2(x, l) for l in $(hs == :Chained_HS46 ? :It_L1 : :It_L2))
        @add_obj(c, LVB.$ob(x, i) for i in 1:floor(Int, (data.N-2)/3))
        c
    end))
end

for hs in (:Chained_HS47, :Chained_HS50, :Chained_HS51, :Chained_HS52, :Chained_HS53)
    con1 = Symbol(hs, :_constraint1); con2 = Symbol(hs, :_constraint2); con3 = Symbol(hs, :_constraint3)
    st = Symbol(hs, :_start); ob = Symbol(hs, :_objective)
    fillstart = hs in (:Chained_HS52, :Chained_HS53)
    startex = fillstart ? :(fill(LVB.$st(1), data.N)) : :((LVB.$st(i) for i = 1:data.N))
    @eval push!(LV_CASES, lv_case($(string(hs)), () -> record(LV_TEMPLATE) do c, data
        nC = 3 * (data.N - 1) ÷ 4
        It_L1 = [4 * div(i-1, 3) for i in 1:3:nC-3]
        It_L2 = [4 * div(i-1, 3) for i in 2:3:nC-3]
        It_L3 = [4 * div(i-1, 3) for i in 3:3:nC-3]
        @add_var(c, x, data.N; start = $startex)
        @add_con(c, LVB.$con1(x, l) for l in It_L1)
        @add_con(c, LVB.$con2(x, l) for l in It_L2)
        @add_con(c, LVB.$con3(x, l) for l in It_L3)
        @add_obj(c, LVB.$ob(x, i) for i in 1:floor(Int, (data.N-1)/4))
        c
    end))
end

lv_direct(name, N) = getfield(LVB, Symbol(name, :_model))(LVB.ExaModelsBackend(), N)

# ── 2-D Luksan (mirrors test/NLPTest/luksan.jl) — schema: (; N, M) ───────────

luksan2d_tape() = record((; N = 6, M = 2)) do c, data
    @add_var(c, x, data.N, data.M; start = (luksan_vlcek_x0(i) for i = 1:data.N, j = 1:data.M))
    @add_con(c, s, luksan_vlcek_con1(x, i, j) for i = 1:(data.N-2), j = 1:data.M)
    @add_con!(c, s, (i, j) => luksan_vlcek_con2(x, i, j) for i = 1:(data.N-2), j = 1:data.M)
    @add_obj(c, luksan_vlcek_obj(x, i, j) for i = 2:data.N, j = 1:data.M)
    c
end

# ── AC power flow (mirrors __exa_ac_power_model in test/NLPTest/power.jl) ────
# Schema: the parsed pglib named tuple. One tape serves every grid whose data
# has the same field types.

function opf_build(c, data)
    @add_var(c, va, length(data.bus);)
    @add_var(c, vm, length(data.bus);
        start = fill(1.0, length(data.bus)), lvar = data.vmin, uvar = data.vmax)
    @add_var(c, pg, length(data.gen); lvar = data.pmin, uvar = data.pmax)
    @add_var(c, qg, length(data.gen); lvar = data.qmin, uvar = data.qmax)
    @add_var(c, p, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    @add_var(c, q, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    @add_obj(c, g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3 for g in data.gen)

    @add_con(c, c1, va[i] for i in data.ref_buses)
    @add_con(c, c2,
        p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch)
    @add_con(c, c3,
        q[b.f_idx] +
        b.c6 * vm[b.f_bus]^2 +
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch)
    @add_con(c, c4,
        p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch)
    @add_con(c, c5,
        q[b.t_idx] +
        b.c8 * vm[b.t_bus]^2 +
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch)
    @add_con(c, c6, va[b.f_bus] - va[b.t_bus] for b in data.branch;
        lcon = data.angmin, ucon = data.angmax)
    @add_con(c, c7, p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill(-Inf, length(data.branch)))
    @add_con(c, c8, p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill(-Inf, length(data.branch)))
    @add_con(c, c9, b.pd + b.gs * vm[b.i]^2 for b in data.bus)
    @add_con(c, c10, b.qd - b.bs * vm[b.i]^2 for b in data.bus)

    @add_con!(c, c9, a.bus => p[a.i] for a in data.arc)
    @add_con!(c, c10, a.bus => q[a.i] for a in data.arc)
    @add_con!(c, c9, g.bus => -pg[g.i] for g in data.gen)
    @add_con!(c, c10, g.bus => -qg[g.i] for g in data.gen)
    return c
end

# ── COPS models (chain, camshape) — schema: (; n) ────────────────────────────
# Instance scalars that appear inside expressions (h = tf/nh, cos(d_theta),
# the objective coefficient) become parameters, so recorded trees differ from
# the direct models' baked constants: compared densely.

cops_chain_tape() = record((; n = 20)) do c, data
    nh = max(2, div(data.n - 4, 4))
    nh1 = nh + 1
    L = 4
    a = 1
    b = 3
    tmin = 1 / 4   # b > a for these constants, matching the direct model
    tf = 1.0
    c, h = add_par(c, 1; value = tf / nh)
    @add_var(c, u, nh1; start = [4 * abs(b - a) * (k / nh - tmin) for k in 1:nh1])
    @add_var(c, x1, nh1; start = [4 * abs(b - a) * k / nh * (1 / 2 * k / nh - tmin) + a for k in 1:nh1])
    @add_var(c, x2, nh1; start = [(4 * abs(b - a) * k / nh * (1 / 2 * k / nh - tmin) + a) *
        (4 * abs(b - a) * (k / nh - tmin)) for k in 1:nh1])
    @add_var(c, x3, nh1; start = [4 * abs(b - a) * (k / nh - tmin) for k in 1:nh1])
    @add_obj(c, x2[m] for m in nh1:nh1)
    @add_con(c, x1[j + 1] - x1[j] - 1 / 2 * h[1] * (u[j] + u[j + 1]) for j in 1:nh)
    @add_con(c, x1[1] - a for _ in 1:1)
    @add_con(c, x1[m] - b for m in nh1:nh1)
    @add_con(c, x2[1] for _ in 1:1)
    @add_con(c, x3[1] for _ in 1:1)
    @add_con(c, x3[m] - L for m in nh1:nh1)
    @add_con(c, x2[j + 1] - x2[j] - 1 / 2 * h[1] * (x1[j] * sqrt(1 + u[j]^2) + x1[j + 1] * sqrt(1 + u[j + 1]^2)) for j in 1:nh)
    @add_con(c, x3[j + 1] - x3[j] - 1 / 2 * h[1] * (sqrt(1 + u[j]^2) + sqrt(1 + u[j + 1]^2)) for j in 1:nh)
    c
end

cops_camshape_tape() = record((; n = 20); minimize = false) do c, data
    R_v = 1.0
    R_max = 2.0
    R_min = 1.0
    alpha = 1.5
    d_theta = 2 * pi / (5 * (data.n + 1))
    c, dt = add_par(c, 1; value = d_theta)
    c, ocoef = add_par(c, 1; value = pi * R_v / data.n)
    @add_var(c, r, 1:data.n; lvar = R_min, uvar = R_max, start = (R_min + R_max) / 2.0)
    @add_obj(c, ocoef[1] * r[i] for i in 1:data.n)
    @add_con(c, -r[i-1] * r[i] - r[i] * r[i+1] + 2 * r[i-1] * r[i+1] * cos(dt[1]) for i = 2:data.n-1; lcon = -Inf, ucon = 0.0)
    @add_con(c, -R_min * r[1] - r[1] * r[2] + 2 * R_min * r[2] * cos(dt[1]) for _ in 1:1; lcon = -Inf, ucon = 0.0)
    @add_con(c, -R_min^2 - R_min * r[1] + 2 * R_min * r[1] * cos(dt[1]) for _ in 1:1; lcon = -Inf, ucon = 0.0)
    @add_con(c, -r[m-1] * r[m] - r[m] * R_max + 2 * r[m-1] * R_max * cos(dt[1]) for m in data.n:data.n; lcon = -Inf, ucon = 0.0)
    @add_con(c, -2 * R_max * r[m] + 2 * r[m]^2 * cos(dt[1]) for m in data.n:data.n; lcon = -Inf, ucon = 0.0)
    @add_con(c, (r[i+1] - r[i]) for i = 1:data.n-1; lcon = -alpha * d_theta, ucon = alpha * d_theta)
    @add_con(c, (r[1] - R_min) for _ in 1:1; lcon = -alpha * d_theta, ucon = alpha * d_theta)
    @add_con(c, (R_max - r[m]) for m in data.n:data.n; lcon = -alpha * d_theta, ucon = alpha * d_theta)
    c
end
