# AC power flow on the lazy core: the structured-args model. One core built
# against args' record vectors (bus/gen/arc/branch) materializes on any grid
# whose records have the same field types.
function opf_lazy(args)
    c = ExaCore(concrete = Val(true))
    c, va = add_var(c, length(args.bus))
    c, vm = add_var(c, length(args.bus);
        start = fill(1.0, length(args.bus)), lvar = args.vmin, uvar = args.vmax)
    c, pg = add_var(c, length(args.gen); lvar = args.pmin, uvar = args.pmax)
    c, qg = add_var(c, length(args.gen); lvar = args.qmin, uvar = args.qmax)
    c, p = add_var(c, length(args.arc); lvar = -args.rate_a, uvar = args.rate_a)
    c, q = add_var(c, length(args.arc); lvar = -args.rate_a, uvar = args.rate_a)

    c, _ = add_obj(c, g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3 for g in args.gen)

    c, c1 = add_con(c, va[i] for i in args.ref_buses)
    c, c2 = add_con(c,
        p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in args.branch)
    c, c3 = add_con(c,
        q[b.f_idx] +
        b.c6 * vm[b.f_bus]^2 +
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in args.branch)
    c, c4 = add_con(c,
        p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in args.branch)
    c, c5 = add_con(c,
        q[b.t_idx] +
        b.c8 * vm[b.t_bus]^2 +
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in args.branch)
    c, c6 = add_con(c, va[b.f_bus] - va[b.t_bus] for b in args.branch;
        lcon = args.angmin, ucon = args.angmax)
    c, c7 = add_con(c, p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in args.branch;
        lcon = fill(-Inf, length(args.branch)))
    c, c8 = add_con(c, p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in args.branch;
        lcon = fill(-Inf, length(args.branch)))
    c, c9 = add_con(c, b.pd + b.gs * vm[b.i]^2 for b in args.bus)
    c, c10 = add_con(c, b.qd - b.bs * vm[b.i]^2 for b in args.bus)

    c, _ = add_con!(c, c9, a.bus => p[a.i] for a in args.arc)
    c, _ = add_con!(c, c10, a.bus => q[a.i] for a in args.arc)
    c, _ = add_con!(c, c9, g.bus => -pg[g.i] for g in args.gen)
    c, _ = add_con!(c, c10, g.bus => -qg[g.i] for g in args.gen)
    return c
end
