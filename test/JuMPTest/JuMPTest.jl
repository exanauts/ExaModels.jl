module JuMPTest

using Test, JuMP, ExaModels, PowerModels, NLPModelsIpopt, ..NLPTest

import ..BACKENDS

const JUMP_INTERFACE_INSTANCES = [
    (:jump_luksan_vlcek_model, [3, 10]),
    (:jump_ac_power_model, ["pglib_opf_case3_lmbd.m", "pglib_opf_case14_ieee.m"]),
]

function jump_luksan_vlcek_model(N)
    jm = JuMP.Model()

    JuMP.@variable(jm, x[i=1:N], start = mod(i, 2) == 1 ? -1.2 : 1.0)
    JuMP.@constraint(
        jm,
        s[i=1:(N-2)],
        3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
        x[i]exp(x[i] - x[i+1]) - 3 == 0.0
    )
    JuMP.@objective(jm, Min, sum(100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N))

    return jm
end
function fixed_variable_e2etest()
    N=5
    jm = JuMP.Model()

    JuMP.@variable(jm, x[1:N])
    JuMP.fix(x[1], 1.0)
    JuMP.@constraint(jm, sum(x) == 1.0)
    JuMP.@objective(jm, Min, sum(2*x[i]^2 for i in 1:N))

    em = ExaModel(jm)
    @test only(em.meta.lcon) == only(em.meta.ucon) == 1.0

    # FIXME: two ConstraintAug?
    wcons = em.cons
    @test typeof(wcons) <: ExaModels.ConstraintAug
    @test typeof(wcons.f.f) <: ExaModels.Null

    cons = wcons.inner
    @test typeof(cons) <: ExaModels.ConstraintAug
    @test typeof(cons.f.f) <: Pair

    @test typeof(cons.f.f.second) <: ExaModels.Node2{typeof(*), ExaModels.Var{T1}, T2} where {T1<:ExaModels.ParIndexed,T2<:ExaModels.ParIndexed}

    @test typeof(cons.inner) <: ExaModels.Constraint
    @test typeof(cons.inner.f.f) <: ExaModels.Null{Nothing}
    @test typeof(cons.inner.inner) <: ExaModels.ConstraintNull

    @test typeof(em.objs.f.f) <: ExaModels.Null
    @test typeof(em.objs.inner.f.f) <: ExaModels.Node2{typeof(*), T1, ExaModels.Node1{typeof(abs2), ExaModels.Var{T2}}} where {T1<:ExaModels.ParIndexed,T2<:ExaModels.ParIndexed}

    jm = JuMP.Model()

    JuMP.@variable(jm, x[1:N])
    JuMP.@variable(jm, p in JuMP.Parameter(1.0))
    JuMP.@constraint(jm, sum(x) == p)
    JuMP.@objective(jm, Min, sum(x))

    @test_broken em = ExaModel(jm) # FIXME: merge #158
    # @test only(em.meta.lcon) == only(em.meta.ucon) == 0.0
    # @test only(em.θ) == 1.0
    # wcons = em.cons
    # @test typeof(wcons) <: ExaModels.ConstraintAug
    # @test typeof(wcons.f.f) <: ExaModels.Null
    # cons = wcons.inner
    # @test typeof(cons) <: ExaModels.ConstraintAug
    # @test typeof(cons.f.f) <: Pair
    # @test typeof(cons.f.f.second) <: ExaModels.Node2{typeof(*), ExaModels.ParameterNode{T1}, T2} where {T1<:ExaModels.ParIndexed,T2<:ExaModels.ParIndexed}
    # @test typeof(cons.inner) <: ExaModels.ConstraintAug
    # @test typeof(cons.inner.f.f) <: Pair
    # @test typeof(cons.inner.f.f.second) <: ExaModels.Node2{typeof(*), ExaModels.Var{T1}, T2} where {T1<:ExaModels.ParIndexed,T2<:ExaModels.ParIndexed}
    # @test typeof(cons.inner.inner) <: ExaModels.Constraint
    # @test typeof(cons.inner.inner.f.f) <: ExaModels.Null{Nothing}
    # @test typeof(cons.inner.inner.inner) <: ExaModels.ConstraintNull

    jm = JuMP.Model()
    JuMP.@variable(jm, x)
    @test_broken em = ExaModel(jm)  # FIXME: support feasibility problems?

    return jm
end
function no_constraints_e2etest()
    N=5
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:N])
    JuMP.@objective(jm, Max, sum(sin(x[i]) for i in 1:N))

    em = ExaModel(jm)

    @test typeof(em.cons) <: ExaModels.Constraint
    @test typeof(em.cons.inner) <: ExaModels.ConstraintNull

    @test typeof(em.objs.f.f) <: ExaModels.Null
    @test typeof(em.objs.inner.f.f) <: ExaModels.Node1{typeof(sin), ExaModels.Var{T1}} where {T1<:ExaModels.ParIndexed}

    N=5
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:N])
    JuMP.@objective(jm, Max, sin(sum(x[i] for i in 1:N)))

    em = ExaModel(jm)

    @test typeof(em.cons) <: ExaModels.Constraint
    @test typeof(em.cons.inner) <: ExaModels.ConstraintNull

    @test typeof(em.objs.f.f) <: ExaModels.Null
    # broken since ExaMOI fails to detect SIMD in this case
    @test_broken typeof(em.objs.inner.f.f) <: ExaModels.Node1{typeof(sin), ExaModels.Var{T1}} where {T1}
end
function generic_e2etest()
    N=5
    jm = JuMP.GenericModel{Float32}()
    JuMP.@variable(jm, x[1:N])
    JuMP.@constraint(jm, sum(x) == 1.0f0)
    JuMP.@objective(jm, Min, sum(x[i]^2 for i in 1:N))

    em = ExaModel(jm)
    
    @test typeof(em) <: ExaModel{Float32}
    @test typeof(getindex.(em.cons.inner.itr, 2)) <: Vector{Float32}
end

function jump_ac_power_model(filename = "pglib_opf_case3_lmbd.m")

    ref = NLPTest.get_power_data_ref(filename)

    model = JuMP.Model()
    #JuMP.set_optimizer_attribute(model, "print_level", 0)

    JuMP.@variable(model, va[i in keys(ref[:bus])])
    JuMP.@variable(model, will_delete)
    JuMP.@variable(
        model,
        ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"],
        start = 1.0
    )

    JuMP.@variable(
        model,
        ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"]
    )
    JuMP.@variable(
        model,
        ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"]
    )

    JuMP.@variable(
        model,
        -ref[:branch][l]["rate_a"] <=
        p[(l, i, j) in ref[:arcs]] <=
        ref[:branch][l]["rate_a"]
    )
    JuMP.@variable(
        model,
        -ref[:branch][l]["rate_a"] <=
        q[(l, i, j) in ref[:arcs]] <=
        ref[:branch][l]["rate_a"]
    )

    JuMP.@objective(
        model,
        Min,
        sum(
            gen["cost"][1] * pg[i]^2 + gen["cost"][2] * pg[i] + gen["cost"][3] for
            (i, gen) in ref[:gen]
        )
    )

    for (i, bus) in ref[:ref_buses]
        JuMP.@constraint(model, va[i] == 0)
    end

    for (i, bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        JuMP.@constraint(
            model,
            sum(p[a] for a in ref[:bus_arcs][i]) ==
            sum(pg[g] for g in ref[:bus_gens][i]) - sum(load["pd"] for load in bus_loads) -
            sum(shunt["gs"] for shunt in bus_shunts) * vm[i]^2
        )

        JuMP.@constraint(
            model,
            sum(q[a] for a in ref[:bus_arcs][i]) ==
            sum(qg[g] for g in ref[:bus_gens][i]) - sum(load["qd"] for load in bus_loads) +
            sum(shunt["bs"] for shunt in bus_shunts) * vm[i]^2
        )
    end

    # Branch power flow physics and limit constraints
    for (i, branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])

        p_fr = p[f_idx]
        q_fr = q[f_idx]
        p_to = p[t_idx]
        q_to = q[t_idx]

        vm_fr = vm[branch["f_bus"]]
        vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]
        va_to = va[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]

        # From side of the branch flow
        JuMP.@constraint(
            model,
            p_fr ==
            (g + g_fr) / ttm * vm_fr^2 +
            (-g * tr + b * ti) / ttm * (vm_fr * vm_to * cos(va_fr - va_to)) +
            (-b * tr - g * ti) / ttm * (vm_fr * vm_to * sin(va_fr - va_to))
        )
        JuMP.@constraint(
            model,
            q_fr ==
            -(b + b_fr) / ttm * vm_fr^2 -
            (-b * tr - g * ti) / ttm * (vm_fr * vm_to * cos(va_fr - va_to)) +
            (-g * tr + b * ti) / ttm * (vm_fr * vm_to * sin(va_fr - va_to))
        )

        # To side of the branch flow
        JuMP.@constraint(
            model,
            p_to ==
            (g + g_to) * vm_to^2 +
            (-g * tr - b * ti) / ttm * (vm_to * vm_fr * cos(va_to - va_fr)) +
            (-b * tr + g * ti) / ttm * (vm_to * vm_fr * sin(va_to - va_fr))
        )
        JuMP.@constraint(
            model,
            q_to ==
            -(b + b_to) * vm_to^2 -
            (-b * tr + g * ti) / ttm * (vm_to * vm_fr * cos(va_to - va_fr)) +
            (-g * tr - b * ti) / ttm * (vm_to * vm_fr * sin(va_to - va_fr))
        )

        # Voltage angle difference limit
        JuMP.@constraint(model, branch["angmin"] <= va_fr - va_to <= branch["angmax"])

        # Apparent power limit, from side and to side
        JuMP.@constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        JuMP.@constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end

    JuMP.delete(model, will_delete)

    return model
end

function runtests()
    @testset "JuMP Interface test" begin
        for (model, cases) in JUMP_INTERFACE_INSTANCES
            for case in cases
                @testset "$model $case" begin
                    modelfunction = getfield(@__MODULE__, model)

                    # solve JuMP problem
                    jm = modelfunction(case)
                    set_optimizer(jm, NLPModelsIpopt.Ipopt.Optimizer)
                    set_optimizer_attribute(jm, "print_level", 0)
                    optimize!(jm)
                    sol = value.(all_variables(jm))

                    for backend in BACKENDS
                        @testset "$backend" begin
                            m = WrapperNLPModel(ExaModel(jm; backend = backend))
                            result = ipopt(m; print_level = 0)

                            @test sol ≈ result.solution atol = 1e-6
                        end
                    end
                end
            end
        end
        @testset "E2E tests" begin
            generic_e2etest()
            fixed_variable_e2etest()
            no_constraints_e2etest()
        end
    end
end

end # module
