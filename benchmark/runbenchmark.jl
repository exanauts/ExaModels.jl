import Pkg
if "main" in ARGS
    Pkg.activate(joinpath(@__DIR__, "main"))
    Pkg.instantiate()
    const rev = "main"
elseif "current" in ARGS
    Pkg.activate(joinpath(@__DIR__, "current"))
    Pkg.instantiate()
    const rev = "current"
else
    error("Please specify either 'main' or 'current' as an argument to select the environment.")
end

using ExaModels, KernelAbstractions, Printf, JLD2, ExaPowerIO, Random

# ── Backend selection ──────────────────────────────────────────────────────────
# Usage: julia runbenchmark.jl (main|current) [nothing] [cuda] [amdgpu] [oneapi] [all]
# Default (no extra args): CPU only.

const _extra = filter(a -> a ∉ ("main", "current"), ARGS)
const _all   = "all" in _extra

# Each entry: (label, backend_object, sync_fn)
const BENCH_BACKENDS = Tuple{String, Any, Function}[]

if isempty(_extra) || "nothing" in _extra || _all
    push!(BENCH_BACKENDS, ("nothing", nothing, () -> nothing))
end

if "cuda" in _extra || _all
    try
        @eval using CUDA
        if CUDA.functional()
            push!(BENCH_BACKENDS, ("CUDA", CUDABackend(), CUDA.synchronize))
        else
            @warn "CUDA loaded but not functional — skipping"
        end
    catch e
        @warn "CUDA unavailable: $e"
    end
end

if "amdgpu" in _extra || _all
    try
        @eval using AMDGPU
        if AMDGPU.functional()
            push!(BENCH_BACKENDS, ("AMDGPU", ROCBackend(), AMDGPU.synchronize))
        else
            @warn "AMDGPU loaded but not functional — skipping"
        end
    catch e
        @warn "AMDGPU unavailable: $e"
    end
end

if "oneapi" in _extra || _all
    try
        @eval using oneAPI
        if oneAPI.functional()
            push!(BENCH_BACKENDS, ("oneAPI", oneAPIBackend(), oneAPI.synchronize))
        else
            @warn "oneAPI loaded but not functional — skipping"
        end
    catch e
        @warn "oneAPI unavailable: $e"
    end
end

if "metal" in _extra || _all
    try
        @eval using Metal
        if Metal.functional()
            push!(BENCH_BACKENDS, ("Metal", MetalBackend(), Metal.synchronize))
        else
            @warn "Metal loaded but not functional — skipping"
        end
    catch e
        @warn "Metal unavailable: $e"
    end
end

isempty(BENCH_BACKENDS) && error("No functional backends available for the requested selection: $(_extra)")

# ── Timing helpers ─────────────────────────────────────────────────────────────

function belapsed(ex; samples = 1000)
    ex()
    minimum(@elapsed ex() for _ in 1:samples)
end

function benchmark_callbacks(m, sync = () -> nothing; samples = 10)
    nvar = m.meta.nvar
    ncon = m.meta.ncon
    nnzj = m.meta.nnzj
    nnzh = m.meta.nnzh

    x    = copy(m.meta.x0)
    y    = similar(x, ncon)
    c    = similar(x, ncon)
    g    = similar(x, nvar)
    jac  = similar(x, nnzj)
    hess = similar(x, nnzh)

    tobj  = belapsed(() -> (ExaModels.obj(m, x);          sync()); samples)
    tcon  = belapsed(() -> (ExaModels.cons!(m, x, c);     sync()); samples)
    tgrad = belapsed(() -> (ExaModels.grad!(m, x, g);     sync()); samples)
    tjac  = belapsed(() -> (ExaModels.jac_coord!(m, x, jac);      sync()); samples)
    thess = belapsed(() -> (ExaModels.hess_coord!(m, x, y, hess); sync()); samples)

    return (nvar=nvar, ncon=ncon, tobj=tobj, tcon=tcon, tgrad=tgrad, tjac=tjac, thess=thess)
end

function print_header(title)
    println()
    println("=" ^ 90)
    @printf("  %-26s  %6s %6s | %8s %8s %8s %8s %8s\n",
        title, "nvar", "ncon", "obj", "cons", "grad", "jac", "hess")
    println("=" ^ 90)
end

function print_row(name, r)
    @printf("  %-26s  %6s %6s | %8.2e %8s %8.2e %8s %8.2e\n",
        name,
        r.nvar < 1_000_000 ? @sprintf("%6d", r.nvar) : @sprintf("%5.0fk", r.nvar/1000),
        r.ncon < 1_000_000 ? @sprintf("%6d", r.ncon) : @sprintf("%5.0fk", r.ncon/1000),
        r.tobj,
        isnan(r.tcon) ? "  N/A   " : @sprintf("%8.2e", r.tcon),
        r.tgrad,
        isnan(r.tjac) ? "  N/A   " : @sprintf("%8.2e", r.tjac),
        r.thess,
    )
end

function parse_ac_power_data(filename)
    data = ExaPowerIO.parse_matpower(filename; library = :pglib)
    return (
        baseMVA = [data.baseMVA],
        bus = data.bus,
        gen = data.gen,
        arc = data.arc,
        branch = data.branch,
        storage = isempty(data.storage) ? empty_data = Vector{NamedTuple{(:i,), Tuple{Int64}}}() : data.storage,
        ref_buses = [i for i in 1:length(data.bus) if data.bus[i].type == 3],
        vmax = [bu.vmax for bu in data.bus],
        vmin = [bu.vmin for bu in data.bus],
        pmax = [g.pmax for g in data.gen],
        pmin = [g.pmin for g in data.gen],
        qmax = [g.qmax for g in data.gen],
        qmin = [g.qmin for g in data.gen],
        angmax = [br.angmax for br in data.branch],
        angmin = [br.angmin for br in data.branch],
        rate_a = [a.rate_a for a in data.arc],
        vm0 = [b.vm for b in data.bus],
        va0 = [b.va for b in data.bus],
        pg0 = [g.pg for g in data.gen],
        qg0 = [g.qg for g in data.gen],
        pdmax = isempty(data.storage) ? Vector{NamedTuple{(:i,), Tuple{Int64}}}() : [s.charge_rating for s in data.storage],
        pcmax = isempty(data.storage) ? Vector{NamedTuple{(:i,), Tuple{Int64}}}() : [s.discharge_rating for s in data.storage],
        srating = isempty(data.storage) ? Vector{NamedTuple{(:i,), Tuple{Int64}}}() : [s.thermal_rating for s in data.storage],
        emax = isempty(data.storage) ? Vector{NamedTuple{(:i,), Tuple{Int64}}}() : [s.energy_rating for s in data.storage],
    )
end

if pkgversion(ExaModels) > v"0.9.7"

    # ── Rosenrock ───────────────────────────────────────────────────────────────
    @eval @inline function exa_rosenrock_model(backend, N)
        c = ExaCore(; backend)
        @var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
        @con(c, s, 3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2))
        @obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
        return ExaModel(c)
    end

    # ── AC OPF ──────────────────────────────────────────────────────────────────
    @eval @inline function exa_opf_model(backend, case)
        data = parse_ac_power_data(case)
        core = ExaCore(; backend)

        @var(core, pg, length(data.gen); lvar = data.pmin, uvar = data.pmax)
        @var(core, qg, length(data.gen); lvar = data.qmin, uvar = data.qmax)
        @var(core, p,  length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
        @var(core, q,  length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
        @var(core, va, length(data.bus))
        @var(core, vm, length(data.bus);
            start = fill!(similar(data.bus, Float64), 1.0),
            lvar  = data.vmin,
            uvar  = data.vmax)

        @obj(core, g.c[1] * pg[g.i]^2 + g.c[2] * pg[g.i] + g.c[3] for g in data.gen)

        @con(core, c_ref_angle, va[i] for i in data.ref_buses)

        @con(core, c_to_active_power_flow,
            p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
            b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
            b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
            for b in data.branch)

        @con(core, c_to_reactive_power_flow,
            q[b.f_idx] +
            b.c6 * vm[b.f_bus]^2 +
            b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
            b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
            for b in data.branch)

        @con(core, c_from_active_power_flow,
            p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
            b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
            b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
            for b in data.branch)

        @con(core, c_from_reactive_power_flow,
            q[b.t_idx] +
            b.c8 * vm[b.t_bus]^2 +
            b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
            b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
            for b in data.branch)

        @con(core, c_phase_angle_diff,
            va[b.f_bus] - va[b.t_bus] for b in data.branch;
            lcon = data.angmin, ucon = data.angmax)

        @con(core, c_active_power_balance,   b.pd + b.gs * vm[b.i]^2 for b in data.bus)
        @con(core, c_reactive_power_balance, b.qd - b.bs * vm[b.i]^2 for b in data.bus)

        @con!(core, c_active_power_balance,   a.bus => p[a.i]   for a in data.arc)
        @con!(core, c_reactive_power_balance, a.bus => q[a.i]   for a in data.arc)
        @con!(core, c_active_power_balance,   g.bus => -pg[g.i] for g in data.gen)
        @con!(core, c_reactive_power_balance, g.bus => -qg[g.i] for g in data.gen)

        @con(core, c_from_thermal_limit,
            p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a^2 for b in data.branch;
            lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf))

        @con(core, c_to_thermal_limit,
            p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a^2 for b in data.branch;
            lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf))

        return ExaModel(core)
    end

    # ── COPS: Hanging Chain ─────────────────────────────────────────────────────
    @eval @inline function cops_chain_model(backend, n)
        nh = max(2, div(n - 4, 4))
        L = 4; a = 1; b = 3
        tmin = b > a ? 1/4 : 3/4
        tf = 1.0; h = tf / nh

        c = ExaCore(; backend = backend)
        @var(c, u,  nh+1; start = [4*abs(b-a)*(k/nh - tmin) for k in 1:nh+1])
        @var(c, x1, nh+1; start = [4*abs(b-a)*k/nh*(1/2*k/nh - tmin) + a for k in 1:nh+1])
        @var(c, x2, nh+1; start = [(4*abs(b-a)*k/nh*(1/2*k/nh - tmin) + a) *
            (4*abs(b-a)*(k/nh - tmin)) for k in 1:nh+1])
        @var(c, x3, nh+1; start = [4*abs(b-a)*(k/nh - tmin) for k in 1:nh+1])

        @obj(c, x2[nh+1])
        @con(c, c1, x1[j+1] - x1[j] - 1/2*h*(u[j] + u[j+1]) for j in 1:nh)
        @con(c, c2, x1[1] - a)
        @con(c, c3, x1[nh+1] - b)
        @con(c, c4, x2[1])
        @con(c, c5, x3[1])
        @con(c, c6, x3[nh+1] - L)
        @con(c, c7, x2[j+1] - x2[j] - 1/2*h*(x1[j]*sqrt(1+u[j]^2) + x1[j+1]*sqrt(1+u[j+1]^2)) for j in 1:nh)
        @con(c, c8, x3[j+1] - x3[j] - 1/2*h*(sqrt(1+u[j]^2) + sqrt(1+u[j+1]^2)) for j in 1:nh)

        return ExaModel(c)
    end

    # ── COPS: Electrons on a Sphere ─────────────────────────────────────────────
    @eval @inline function cops_elec_model(backend, np; seed = 2713)
        Random.seed!(seed)
        theta = (2pi) .* rand(np)
        phi   = pi    .* rand(np)
        itr   = [(i, j) for i in 1:np-1 for j in i+1:np]

        core = ExaCore(; backend = backend)
        @var(core, x, 1:np; start = [cos(theta[i])*sin(phi[i]) for i = 1:np])
        @var(core, y, 1:np; start = [sin(theta[i])*sin(phi[i]) for i = 1:np])
        @var(core, z, 1:np; start = [cos(phi[i]) for i = 1:np])

        @obj(core, 1.0 / sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2 + (z[i]-z[j])^2) for (i, j) in itr)
        @con(core, c1, x[i]^2 + y[i]^2 + z[i]^2 - 1 for i = 1:np)

        return ExaModel(core)
    end

else
    # ── Legacy API (ExaModels ≤ 0.9.7) ─────────────────────────────────────────

    function exa_rosenrock_model(backend, N)
        c = ExaCore(backend = backend)
        x = variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
        constraint(c, 3x[i+1]^3 + 2*x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2))
        objective(c, 100*(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
        return ExaModel(c)
    end

    function exa_opf_model(backend, case)
        data = parse_ac_power_data(case)
        core = ExaCore(backend = backend)

        pg = variable(core, length(data.gen); lvar = data.pmin, uvar = data.pmax)
        qg = variable(core, length(data.gen); lvar = data.qmin, uvar = data.qmax)
        p  = variable(core, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
        q  = variable(core, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
        va = variable(core, length(data.bus))
        vm = variable(core, length(data.bus);
            start = fill!(similar(data.bus, Float64), 1.0),
            lvar  = data.vmin,
            uvar  = data.vmax)

        objective(core, g.c[1] * pg[g.i]^2 + g.c[2] * pg[g.i] + g.c[3] for g in data.gen)

        constraint(core, va[i] for i in data.ref_buses)

        constraint(core,
            p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
            b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
            b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
            for b in data.branch)

        constraint(core,
            q[b.f_idx] +
            b.c6 * vm[b.f_bus]^2 +
            b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
            b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus]))
            for b in data.branch)

        constraint(core,
            p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
            b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
            b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
            for b in data.branch)

        constraint(core,
            q[b.t_idx] +
            b.c8 * vm[b.t_bus]^2 +
            b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
            b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus]))
            for b in data.branch)

        constraint(core,
            va[b.f_bus] - va[b.t_bus] for b in data.branch;
            lcon = data.angmin, ucon = data.angmax)

        c_bal_p = constraint(core, b.pd + b.gs * vm[b.i]^2 for b in data.bus)
        c_bal_q = constraint(core, b.qd - b.bs * vm[b.i]^2 for b in data.bus)

        constraint!(core, c_bal_p, a.bus => p[a.i]   for a in data.arc)
        constraint!(core, c_bal_q, a.bus => q[a.i]   for a in data.arc)
        constraint!(core, c_bal_p, g.bus => -pg[g.i] for g in data.gen)
        constraint!(core, c_bal_q, g.bus => -qg[g.i] for g in data.gen)

        constraint(core,
            p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a^2 for b in data.branch;
            lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf))

        constraint(core,
            p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a^2 for b in data.branch;
            lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf))

        return ExaModel(core)
    end

    function cops_chain_model(backend, n)
        nh = max(2, div(n - 4, 4))
        L = 4; a = 1; b = 3
        tmin = b > a ? 1/4 : 3/4
        tf = 1.0; h = tf / nh

        c = ExaCore(backend = backend)
        u  = variable(c, nh+1; start = [4*abs(b-a)*(k/nh - tmin) for k in 1:nh+1])
        x1 = variable(c, nh+1; start = [4*abs(b-a)*k/nh*(1/2*k/nh - tmin) + a for k in 1:nh+1])
        x2 = variable(c, nh+1; start = [(4*abs(b-a)*k/nh*(1/2*k/nh - tmin) + a) *
            (4*abs(b-a)*(k/nh - tmin)) for k in 1:nh+1])
        x3 = variable(c, nh+1; start = [4*abs(b-a)*(k/nh - tmin) for k in 1:nh+1])

        objective(c, x2[nh+1])
        constraint(c, x1[j+1] - x1[j] - 1/2*h*(u[j] + u[j+1]) for j in 1:nh)
        constraint(c, x1[1] - a)
        constraint(c, x1[nh+1] - b)
        constraint(c, x2[1])
        constraint(c, x3[1])
        constraint(c, x3[nh+1] - L)
        constraint(c, x2[j+1] - x2[j] - 1/2*h*(x1[j]*sqrt(1+u[j]^2) + x1[j+1]*sqrt(1+u[j+1]^2)) for j in 1:nh)
        constraint(c, x3[j+1] - x3[j] - 1/2*h*(sqrt(1+u[j]^2) + sqrt(1+u[j+1]^2)) for j in 1:nh)

        return ExaModel(c)
    end

    function cops_elec_model(backend, np; seed = 2713)
        Random.seed!(seed)
        theta = (2pi) .* rand(np)
        phi   = pi    .* rand(np)
        itr   = [(i, j) for i in 1:np-1 for j in i+1:np]

        core = ExaCore(backend = backend)
        x = variable(core, 1:np; start = [cos(theta[i])*sin(phi[i]) for i = 1:np])
        y = variable(core, 1:np; start = [sin(theta[i])*sin(phi[i]) for i = 1:np])
        z = variable(core, 1:np; start = [cos(phi[i]) for i = 1:np])

        objective(core, 1.0 / sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2 + (z[i]-z[j])^2) for (i, j) in itr)
        constraint(core, x[i]^2 + y[i]^2 + z[i]^2 - 1 for i = 1:np)

        return ExaModel(core)
    end
end

# ── Benchmark loop ─────────────────────────────────────────────────────────────

results = Dict()
print_header("Benchmarks (rev=$rev, backends=$(join(first.(BENCH_BACKENDS), '+')))")

for (bname, backend, sync) in BENCH_BACKENDS
    for (name, thunk) in [
        ("rosenrock-1000",   () -> exa_rosenrock_model(backend, 1000)),
        ("rosenrock-10000",  () -> exa_rosenrock_model(backend, 10000)),
        ("rosenrock-100000", () -> exa_rosenrock_model(backend, 100000)),
        ("OPF-case14",       () -> exa_opf_model(backend, "pglib_opf_case14_ieee.m")),
        ("OPF-case1354",     () -> exa_opf_model(backend, "pglib_opf_case1354_pegase.m")),
        ("OPF-case30000",    () -> exa_opf_model(backend, "pglib_opf_case30000_goc.m")),
        ("chain-10",         () -> cops_chain_model(backend, 10)),
        ("chain-100",        () -> cops_chain_model(backend, 100)),
        ("chain-1000",       () -> cops_chain_model(backend, 1000)),
        ("elec-10",          () -> cops_elec_model(backend, 10)),
        ("elec-100",         () -> cops_elec_model(backend, 100)),
        ("elec-1000",        () -> cops_elec_model(backend, 1000)),
        ]
        key = "$name-$bname"
        m = thunk()
        r = benchmark_callbacks(m, sync)
        print_row(key, r)
        results[key] = r
    end
end

JLD2.@save joinpath(@__DIR__, "benchmark-results-$rev.jld2") results

println()
