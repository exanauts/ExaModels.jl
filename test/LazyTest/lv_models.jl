# Lazy builders for the full LuksanVlcekBenchmark set: each builds against
# args = ArgTracer() with the same LVB expression functions and call order
# as the eager constructors, so materialized models must match them exactly.
# Instance scalars (augmented_lagrangian's h) sit directly in expressions —
# no parameter idiom.

lv_lazy(name, build, sizes = (50, 300)) = (name = name, build = build, sizes = sizes)

const LV_LAZY = [
    lv_lazy("rosenrock", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = (LVB.rosenrock_start(i) for i = 1:args.N))
        c, _ = add_con(c, LVB.rosenrock_constraint(x, i) for i = 1:args.N-2)
        c, _ = add_obj(c, LVB.rosenrock_objective(x, i) for i = 1:args.N-1)
        c
    end),
    lv_lazy("wood", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = (LVB.wood_start(i) for i = 1:args.N))
        c, con = add_con(c, LVB.wood_constraint(x, k) for k in 1:args.N-7)
        c, _ = add_con!(c, con, k => LVB.wood_constraint_aug(x, k) for k in 1:args.N-7)
        c, _ = add_obj(c, LVB.wood_objective(x, i) for i = 1:args.N÷2-1)
        c
    end),
    lv_lazy("chained_powell", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = (LVB.chained_powell_start(i) for i = 1:args.N))
        c, _ = add_con(c, LVB.chained_powell_constraint1(x) for _ in 1:1)
        c, _ = add_con(c, LVB.chained_powell_constraint2(x, n) for n in args.N:args.N)
        c, _ = add_obj(c, LVB.chained_powell_objective(x, i) for i = 1:args.N÷2-1)
        c
    end),
    lv_lazy("cragg_levy", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = (LVB.cragg_levy_start(i) for i = 1:args.N))
        c, _ = add_con(c, LVB.cragg_levy_constraint(x, k) for k = 1:args.N-2)
        c, _ = add_obj(c, LVB.cragg_levy_objective(x, i) for i = 1:args.N÷2-1)
        c
    end),
    lv_lazy("broyden_tridiagonal", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = fill(-1, args.N))
        c, _ = add_con(c, LVB.broyden_tridiagonal_constraint(x, k) for k = 1:args.N-4)
        c, _ = add_obj(c, LVB.broyden_tridiagonal_objective(x, i) for i = 2:args.N-1)
        c, _ = add_obj(c, LVB.broyden_tridiagonal_objective_boundary(x, n) for n in args.N:args.N)
        c
    end),
    lv_lazy("broyden_banded", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = fill(3, args.N))
        c, y = add_var(c, args.N; start = fill(0, args.N))
        c, _ = add_con(c, LVB.broyden_banded_kconstraint(x, k) for k = 1:(args.N-1)÷2)
        c, _ = add_con(c, y[1] - LVB.broyden_banded_xi(x, 1) - LVB.broyden_banded_xi(x, 2) for _ in 1:1)
        c, _ = add_con(c, y[2] - LVB.broyden_banded_xi(x, 2) - LVB.broyden_banded_xi(x, 3) - LVB.broyden_banded_xi(x, 1) for _ in 1:1)
        c, _ = add_con(c, y[3] - LVB.broyden_banded_xi(x, 2) - LVB.broyden_banded_xi(x, 3) - LVB.broyden_banded_xi(x, 1) - LVB.broyden_banded_xi(x, 4) for _ in 1:1)
        c, _ = add_con(c, y[4] - LVB.broyden_banded_xi(x, 2) - LVB.broyden_banded_xi(x, 3) - LVB.broyden_banded_xi(x, 1) - LVB.broyden_banded_xi(x, 4) - LVB.broyden_banded_xi(x, 5) for _ in 1:1)
        c, _ = add_con(c, y[5] - LVB.broyden_banded_xi(x, 2) - LVB.broyden_banded_xi(x, 3) - LVB.broyden_banded_xi(x, 1) - LVB.broyden_banded_xi(x, 4) - LVB.broyden_banded_xi(x, 5) - LVB.broyden_banded_xi(x, 6) for _ in 1:1)
        c, _ = add_con(c, LVB.broyden_banded_yconstraint(y, x, i) for i in 6:args.N-1)
        c, _ = add_con(c, LVB.broyden_banded_ylast(y, x, n) for n in args.N:args.N)
        c, _ = add_obj(c, LVB.broyden_banded_objective(x, y, i) for i in 1:args.N)
        c
    end),
    lv_lazy("trigo_tridiagonal", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = fill(1, args.N))
        c, _ = add_con(c, LVB.trigo_tridiagonal_constraint1(x) for _ in 1:1)
        c, _ = add_con(c, LVB.trigo_tridiagonal_constraint2(x) for _ in 1:1)
        c, _ = add_con(c, LVB.trigo_tridiagonal_constraint3(x, n) for n in args.N:args.N)
        c, _ = add_con(c, LVB.trigo_tridiagonal_constraint4(x, n) for n in args.N:args.N)
        c, _ = add_obj(c, LVB.trigo_tridiagonal_objective(x, i) for i = 2:args.N-1)
        c
    end),
    lv_lazy("generalized_brown", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = fill(-1, args.N))
        c, _ = add_con(c, LVB.generalized_brown_constraint(x, k) for k = 1:args.N-2)
        c, _ = add_obj(c, LVB.generalized_brown_objective(x, i) for i = 1:args.N÷2)
        c
    end),
    lv_lazy("modified_brown", args -> begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, args.N; start = fill(-1, args.N))
        c, _ = add_con(c, LVB.modified_brown_constraint1(x) for _ in 1:1)
        c, _ = add_con(c, LVB.modified_brown_constraint2(x) for _ in 1:1)
        c, _ = add_con(c, LVB.modified_brown_constraint3(x) for _ in 1:1)
        c, _ = add_con(c, LVB.modified_brown_constraint4(x, n) for n in args.N:args.N)
        c, _ = add_con(c, LVB.modified_brown_constraint5(x, n) for n in args.N:args.N)
        c, _ = add_con(c, LVB.modified_brown_constraint6(x, n) for n in args.N:args.N)
        c, _ = add_obj(c, LVB.modified_brown_objective(x, i) for i = 1:args.N÷2)
        c
    end),
    lv_lazy("augmented_lagrangian", args -> begin
        c = ExaCore(concrete = Val(true))
        h = 1 / (args.N + 1)      # instance scalar, directly in expressions
        c, x = add_var(c, args.N; start = (LVB.augmented_lagrangian_start(i) for i = 1:args.N))
        c, _ = add_con(c, LVB.augmented_lagrangian_constraint(x, h, k) for k = 1:args.N-2)
        c, _ = add_obj(c, LVB.augmented_lagrangian_objective(
            x, LVB.augmented_lagrangian_l1, LVB.augmented_lagrangian_l2,
            LVB.augmented_lagrangian_l3, i) for i = 1:args.N÷5)
        c
    end),
]

for hs in (:Chained_HS46, :Chained_HS48, :Chained_HS49)
    con1 = Symbol(hs, :_constraint1); con2 = Symbol(hs, :_constraint2)
    st = Symbol(hs, :_start); ob = Symbol(hs, :_objective)
    @eval push!(LV_LAZY, lv_lazy($(string(hs)), args -> begin
        c = ExaCore(concrete = Val(true))
        nC = 2 * (args.N - 2) ÷ 3
        It_L1 = [3 * div(i-1, 2) for i in 1:2:nC-($(hs == :Chained_HS46 ? 2 : 1))]
        It_L2 = [3 * div(i-1, 2) for i in 2:2:nC]
        c, x = add_var(c, args.N; start = (LVB.$st(i) for i = 1:args.N))
        c, _ = add_con(c, LVB.$con1(x, l) for l in $(hs == :Chained_HS46 ? :It_L2 : :It_L1))
        c, _ = add_con(c, LVB.$con2(x, l) for l in $(hs == :Chained_HS46 ? :It_L1 : :It_L2))
        c, _ = add_obj(c, LVB.$ob(x, i) for i in 1:floor(Int, (args.N-2)/3))
        c
    end))
end

for hs in (:Chained_HS47, :Chained_HS50, :Chained_HS51, :Chained_HS52, :Chained_HS53)
    con1 = Symbol(hs, :_constraint1); con2 = Symbol(hs, :_constraint2); con3 = Symbol(hs, :_constraint3)
    st = Symbol(hs, :_start); ob = Symbol(hs, :_objective)
    fillstart = hs in (:Chained_HS52, :Chained_HS53)
    startex = fillstart ? :(fill(LVB.$st(1), args.N)) : :((LVB.$st(i) for i = 1:args.N))
    @eval push!(LV_LAZY, lv_lazy($(string(hs)), args -> begin
        c = ExaCore(concrete = Val(true))
        nC = 3 * (args.N - 1) ÷ 4
        It_L1 = [4 * div(i-1, 3) for i in 1:3:nC-3]
        It_L2 = [4 * div(i-1, 3) for i in 2:3:nC-3]
        It_L3 = [4 * div(i-1, 3) for i in 3:3:nC-3]
        c, x = add_var(c, args.N; start = $startex)
        c, _ = add_con(c, LVB.$con1(x, l) for l in It_L1)
        c, _ = add_con(c, LVB.$con2(x, l) for l in It_L2)
        c, _ = add_con(c, LVB.$con3(x, l) for l in It_L3)
        c, _ = add_obj(c, LVB.$ob(x, i) for i in 1:floor(Int, (args.N-1)/4))
        c
    end))
end
