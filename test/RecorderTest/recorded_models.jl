# Recorder-side tape builders and case lists for the expanded test set.
#
# The LuksanVlcek tape builders live in LuksanVlcekBenchmark's ExaModels
# extension (`LVB.<name>_tape()`) and the COPS ones in COPSBenchmark's
# (`COPSBenchmark.<name>_tape()`); each records the model once against a
# symbolic schema and is replayed here against the corresponding direct
# constructor. The builders local to this file mirror models that live in
# this repo's own test suite (test/NLPTest). Transcription idioms are
# documented in docs/design/recorder.md.

# ── LuksanVlcekBenchmark set (schema: (; N)) ─────────────────────────────────
# augmented_lagrangian records the instance scalar h = 1/(N+1) as a parameter
# where the direct model bakes it as a constant, so its trees differ and it is
# compared densely.

const LV_CASES = [
    let mname = replace(string(tname), r"_tape$" => "")
        (name = mname, tape = getfield(LVB, tname), sizes = (50, 300),
         dense = mname == "augmented_lagrangian")
    end for tname in LVB.TAPE_NAMES
]

lv_direct(name, N) = getfield(LVB, Symbol(name, :_model))(LVB.ExaModelsBackend(), N)

# ── 2-D Luksan (mirrors test/NLPTest/luksan.jl) — schema: (; N, M) ───────────

luksan2d_tape() = let data = DataTracer((; N = 6, M = 2)), c = ExaTape()
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
