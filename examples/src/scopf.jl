function ac_scopf_model(
    filename, contingencies;
    backend = nothing,
    T = Float64,
    kwargs...
)
    # Raw data
    data = parse_ac_power_data(filename, backend)
    nbus = length(data.bus)
    ngen = length(data.gen)
    narc = length(data.arc)
    # Contingencies
    K = length(contingencies) + 1 # +1 accounts for base case

    # Build indexes
    idx_ref = [(r, k) for r in data.ref_buses, k in 1:K]
    idx_branch = [(b, 1) for b in data.branch]
    idx_branch_down = similar(idx_branch, 0)
    # Scan contingencies
    for k in 2:K # k=1 is base case
        line_down = contingencies[k-1] # ID of line down
        for b in data.branch
            if b.i != line_down
                push!(idx_branch, (b, k))
            else
                push!(idx_branch_down, (b, k))
            end
        end
    end
    display(idx_branch_down)
    idx_bus = [(b, k) for b in data.bus, k in 1:K]
    idx_gen = [(b, k) for b in data.gen, k in 1:K]
    idx_arc = [(b, k) for b in data.arc, k in 1:K]

    # Compute maximum injection in lines
    max_inj = repeat(data.rate_a, K)
    # Remove (pinj, qinj) for lines down  by setting the bounds to 0
    # so the solver can convert it as a fixed variable.
    for (b, k) in idx_branch_down
        max_inj[b.f_idx + (k-1) * narc] = 0.0
        max_inj[b.t_idx + (k-1) * narc] = 0.0
    end

    core = ExaModels.ExaCore(T, backend)

    # Voltage angle and magnitudes
    va = ExaModels.variable(core, 1:nbus, 1:K; start=zeros(nbus, K))
    vm = ExaModels.variable(
        core,
        1:nbus, 1:K;
        start = ones(nbus, K),
        lvar = repeat(data.vmin, K),
        uvar = repeat(data.vmax, K),
    )
    # Power generations
    pg = ExaModels.variable(
        core,
        1:ngen, 1:K;
        lvar = repeat(data.pmin, K),
        uvar = repeat(data.pmax, K),
    )
    qg = ExaModels.variable(
        core,
        1:ngen, 1:K;
        lvar = repeat(data.qmin, K),
        uvar = repeat(data.qmax, K),
    )
    # Power injection
    p = ExaModels.variable(
        core,
        1:narc, 1:K;
        lvar = -max_inj,
        uvar = max_inj,
    )
    q = ExaModels.variable(
        core,
        1:narc, 1:K;
        lvar = -max_inj,
        uvar = max_inj,
    )

    # Objective (cost for base case only)
    obj = ExaModels.objective(
        core,
        g.cost1 * pg[g.i, 1]^2 + g.cost2 * pg[g.i, 1] + g.cost3 for g in data.gen
    )

    # Constraints
    # Voltage angle at reference buses is 0
    c1 = ExaModels.constraint(core, va[i, k] for (i, k) in idx_ref)
    # Branch power injection
    # from, w.r.t active power
    c2 = ExaModels.constraint(
        core,
        p[b.f_idx, k] - b.c5 * vm[b.f_bus, k]^2 -
        b.c3 * (vm[b.f_bus, k] * vm[b.t_bus, k] * cos(va[b.f_bus, k] - va[b.t_bus, k])) -
        b.c4 * (vm[b.f_bus, k] * vm[b.t_bus, k] * sin(va[b.f_bus, k] - va[b.t_bus, k])) for
        (b, k) in idx_branch
    )
    # from, w.r.t reactive power
    c3 = ExaModels.constraint(
        core,
        q[b.f_idx, k] +
        b.c6 * vm[b.f_bus, k]^2 +
        b.c4 * (vm[b.f_bus, k] * vm[b.t_bus, k] * cos(va[b.f_bus, k] - va[b.t_bus, k])) -
        b.c3 * (vm[b.f_bus, k] * vm[b.t_bus, k] * sin(va[b.f_bus, k] - va[b.t_bus, k])) for
        (b, k) in idx_branch
    )
    # to, w.r.t active power
    c4 = ExaModels.constraint(
        core,
        p[b.t_idx, k] - b.c7 * vm[b.t_bus, k]^2 -
        b.c1 * (vm[b.t_bus, k] * vm[b.f_bus, k] * cos(va[b.t_bus, k] - va[b.f_bus, k])) -
        b.c2 * (vm[b.t_bus, k] * vm[b.f_bus, k] * sin(va[b.t_bus, k] - va[b.f_bus, k])) for
        (b, k) in idx_branch
    )
    # to, w.r.t reactive power
    c5 = ExaModels.constraint(
        core,
        q[b.t_idx, k] +
        b.c8 * vm[b.t_bus, k]^2 +
        b.c2 * (vm[b.t_bus, k] * vm[b.f_bus, k] * cos(va[b.t_bus, k] - va[b.f_bus, k])) -
        b.c1 * (vm[b.t_bus, k] * vm[b.f_bus, k] * sin(va[b.t_bus, k] - va[b.f_bus, k])) for
        (b, k) in idx_branch
    )
    # Difference of angles
    c6 = ExaModels.constraint(
        core,
        va[b.f_bus, k] - va[b.t_bus, k] for (b, k) in idx_branch
        lcon = repeat(data.angmin, K),
        ucon = repeat(data.angmax, K),
    )
    # Line-flow constraints
    # from
    lcon = fill!(similar(data.branch, Float64, K * length(data.branch)), -Inf)
    c7 = ExaModels.constraint(
        core,
        p[b.f_idx, k]^2 + q[b.f_idx, k]^2 - b.rate_a_sq for (b, k) in idx_branch;
        lcon = lcon,
    )
    # to
    c8 = ExaModels.constraint(
        core,
        p[b.t_idx, k]^2 + q[b.t_idx, k]^2 - b.rate_a_sq for (b, k) in idx_branch;
        lcon = lcon,
    )

    # Power flow constraints
    c9 = ExaModels.constraint(core, b.pd + b.gs * vm[b.i, k]^2 for (b, k) in idx_bus)
    c10 = ExaModels.constraint(core, b.qd - b.bs * vm[b.i, k]^2 for (b, k) in idx_bus)
    # TODO: shifts of index is hacky.
    c11 = ExaModels.constraint!(core, c9, a.bus + (k-1)*nbus => p[a.i, k] for (a, k) in idx_arc)
    c12 = ExaModels.constraint!(core, c10, a.bus + (k-1)*nbus => q[a.i, k] for (a, k) in idx_arc)
    c13 = ExaModels.constraint!(core, c9, g.bus + (k-1)*nbus => -pg[g.i, k] for (g, k) in idx_gen)
    c14 = ExaModels.constraint!(core, c10, g.bus + (k-1)*nbus=> -qg[g.i, k] for (g, k) in idx_gen)

    # Corrective OPF formulation
    Δp = 0.1 .* (data.pmax .- data.pmin)
    idx_gen_cont = [(b, k) for b in data.gen, k in 2:K] # discard base case
    c_corrective_gen = ExaModels.constraint(
        core,
        pg[g.i, k] - pg[g.i, 1] for (g, k) in idx_gen_cont;
        lcon = repeat(-Δp, K),
        ucon = repeat(Δp, K),
    )
    # Select buses with generators (=PV and REF buses)
    gen_buses = [b for b in data.bus if b[:bus_type] in [2, 3]]
    idx_bus_cont = [(b, k) for b in gen_buses, k in 2:K]
    c_corrective_vmag = ExaModels.constraint(
        core,
        vm[b.i, k] - vm[b.i, 1] for (b, k) in idx_bus_cont;
    )

    return ExaModels.ExaModel(core; kwargs...)
end

