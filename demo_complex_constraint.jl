using ExaModels

println("=" ^ 70)
println("  Complex Constraint Pretty Print Demo")
println("=" ^ 70)

# ═══════════════════════════════════════════════════════════════════
# AC Optimal Power Flow – a classic nonlinear optimization problem
# ═══════════════════════════════════════════════════════════════════

c = ExaCore()

# Bus voltage variables
@var(c, va, 5)     # voltage angles
@var(c, vm, 5)     # voltage magnitudes

# Generator variables
@var(c, pg, 3)     # active power generation
@var(c, qg, 3)     # reactive power generation

# Branch flow variables
@var(c, p, 8)      # active branch flows
@var(c, q, 8)      # reactive branch flows

# Parameters
@par(c, cost, [0.11, 0.085, 0.12])

# ── Objective: generator cost ────────────────────────────────────

println("\n── Quadratic generation cost ──────────────────────────────\n")

c, obj = add_obj(c,
    cost[g.i] * pg[g.i]^2 + g.c1 * pg[g.i] + g.c0
    for g in [(i=1, c1=5.0, c0=0.0), (i=2, c1=1.7, c0=0.0), (i=3, c1=4.0, c0=0.0)]
)
display(obj)

# ── Power balance constraints (with augmentation) ───────────────

println("\n── Active power balance: ∑p_in - ∑p_out = pd ─────────────\n")

bus_data = [(i=k, pd=0.5*k) for k in 1:5]
c, pbal = add_con(c,
    -b.pd for b in bus_data;
    lcon = 0.0, ucon = 0.0,
)
display(pbal)

# Augment: inject generator active power
println("\n── Augment with generator injection ──────────────────────\n")

gen_bus = [(i=1, bus=1), (i=2, bus=3), (i=3, bus=5)]
c, pbal_gen = add_con!(c, pbal,
    g.bus => pg[g.i] for g in gen_bus
)
display(pbal_gen)

# Augment: add branch flows
println("\n── Augment with branch flows ─────────────────────────────\n")

arc_data = [(i=k, bus=mod1(k,5)) for k in 1:8]
c, pbal_arc = add_con!(c, pbal,
    a.bus => -p[a.i] for a in arc_data
)
display(pbal_arc)

# ── Branch flow constraint (complex trig expression) ────────────

println("\n── Branch active power flow (Ohm's law) ─────────────────\n")

branch_data = [
    (f_idx=k, f_bus=k, t_bus=mod1(k+1,5), g=0.1*k, b=0.05*k, tr=1.0, ttm=1.0, g_fr=0.01)
    for k in 1:4
]

c, flow_p = add_con(c,
    p[br.f_idx]
    - br.g / br.ttm * vm[br.f_bus]^2
    - br.g / br.tr * (vm[br.f_bus] * vm[br.t_bus] * cos(va[br.f_bus] - va[br.t_bus]))
    - br.b / br.tr * (vm[br.f_bus] * vm[br.t_bus] * sin(va[br.f_bus] - va[br.t_bus]))
    for br in branch_data;
    lcon = 0.0, ucon = 0.0,
)
display(flow_p)

# ── Reactive power flow constraint ──────────────────────────────

println("\n── Branch reactive power flow ────────────────────────────\n")

c, flow_q = add_con(c,
    q[br.f_idx]
    + br.b / br.ttm * vm[br.f_bus]^2
    + br.b / br.tr * (vm[br.f_bus] * vm[br.t_bus] * cos(va[br.f_bus] - va[br.t_bus]))
    - br.g / br.tr * (vm[br.f_bus] * vm[br.t_bus] * sin(va[br.f_bus] - va[br.t_bus]))
    for br in branch_data;
    lcon = 0.0, ucon = 0.0,
)
display(flow_q)

# ── Thermal limit constraint ────────────────────────────────────

println("\n── Thermal limit: p² + q² ≤ rate² ───────────────────────\n")

thermal_data = [(i=k, rate_a_sq=1.5^2) for k in 1:4]

c, thermal = add_con(c,
    p[t.i]^2 + q[t.i]^2 - t.rate_a_sq
    for t in thermal_data;
    lcon = -Inf, ucon = 0.0,
)
display(thermal)

# ── Voltage angle difference constraint ─────────────────────────

println("\n── Angle difference: -π/6 ≤ va_f - va_t ≤ π/6 ──────────\n")

angdiff_data = [(f=k, t=mod1(k+1,5)) for k in 1:4]

c, angdiff = add_con(c,
    va[a.f] - va[a.t]
    for a in angdiff_data;
    lcon = -π/6, ucon = π/6,
)
display(angdiff)

# ── Subexpression: voltage product ──────────────────────────────

println("\n── Subexpression: voltage product ────────────────────────\n")

vprod_data = [(f=k, t=mod1(k+1,5)) for k in 1:4]

c, vprod = add_expr(c,
    vm[d.f] * vm[d.t] for d in vprod_data
)
display(vprod)

# ── Full model summary ─────────────────────────────────────────

println("\n── Full Model ───────────────────────────────────────────\n")
m = ExaModel(c)
display(m)

println("\n\n", "=" ^ 70)
println("  Done!")
println("=" ^ 70)
