using ExaModels
using LuksanVlcekBenchmark
using COPSBenchmark
using Printf

function btime(f, N)
    f()  # warmup
    GC.gc()
    return minimum(begin
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end for _ in 1:N)
end

function benchmark_callbacks(m; N = 20)
    nvar = m.meta.nvar
    ncon = m.meta.ncon
    nnzj = m.meta.nnzj
    nnzh = m.meta.nnzh

    x    = copy(m.meta.x0)
    y    = zeros(ncon)
    c    = zeros(ncon)
    g    = zeros(nvar)
    jac  = zeros(nnzj)
    hess = zeros(nnzh)

    tobj  = btime(() -> ExaModels.obj(m, x), N)
    tcon  = ncon > 0 ? btime(() -> ExaModels.cons!(m, x, c), N)  : NaN
    tgrad = btime(() -> ExaModels.grad!(m, x, g), N)
    tjac  = ncon > 0 ? btime(() -> ExaModels.jac_coord!(m, x, jac), N)  : NaN
    thess = btime(() -> ExaModels.hess_coord!(m, x, y, hess), N)

    return (nvar=nvar, ncon=ncon, tobj=tobj, tcon=tcon, tgrad=tgrad, tjac=tjac, thess=thess)
end

function print_header(title)
    println()
    println("=" ^ 80)
    @printf("  %-20s  %6s %6s | %8s %8s %8s %8s %8s\n",
        title, "nvar", "ncon", "obj", "cons", "grad", "jac", "hess")
    println("=" ^ 80)
end

function print_row(name, r)
    @printf("  %-20s  %6s %6s | %8.2e %8s %8.2e %8s %8.2e\n",
        name,
        r.nvar < 1_000_000 ? @sprintf("%6d", r.nvar) : @sprintf("%5.0fk", r.nvar/1000),
        r.ncon < 1_000_000 ? @sprintf("%6d", r.ncon) : @sprintf("%5.0fk", r.ncon/1000),
        r.tobj,
        isnan(r.tcon)  ? "  N/A   " : @sprintf("%8.2e", r.tcon),
        r.tgrad,
        isnan(r.tjac)  ? "  N/A   " : @sprintf("%8.2e", r.tjac),
        r.thess,
    )
end

# ── LuksanVlcek ────────────────────────────────────────────────────────────────
print_header("LuksanVlcek (N=10000)")
for name in LuksanVlcekBenchmark.NAMES
    model_func = getfield(LuksanVlcekBenchmark, name)
    try
        m = model_func(LuksanVlcekBenchmark.ExaModelsBackend(), 10000)
        r = benchmark_callbacks(m)
        print_row(string(name), r)
    catch e
        @printf("  %-20s  ERROR: %s\n", string(name), e)
    end
end

# ── COPS ───────────────────────────────────────────────────────────────────────
const COPS_INSTANCES = [
    (:bearing_model,          (50, 50)),
    (:chain_model,            (800,)),
    (:camshape_model,         (1000,)),
    (:catmix_model,           (500,)),
    (:channel_model,          (1000,)),
    (:gasoil_model,           (500,)),
    (:glider_model,           (500,)),
    (:marine_model,           (500,)),
    (:methanol_model,         (500,)),
    (:minsurf_model,          (100, 100)),
    (:pinene_model,           (500,)),
    (:robot_model,            (500,)),
    (:rocket_model,           (2000,)),
    (:steering_model,         (1000,)),
    (:torsion_model,          (100, 100)),
    (:channel_model,          (1000,)),
]

print_header("COPS")
for (sym, params) in COPS_INSTANCES
    model_func = getfield(COPSBenchmark, sym)
    try
        m = model_func(COPSBenchmark.ExaModelsBackend(), params...)
        r = benchmark_callbacks(m)
        print_row("$sym($(join(params,',')))", r)
    catch e
        @printf("  %-20s  ERROR: %s\n", string(sym), e)
    end
end

println()
