module JuMPTest

using Test, JuMP, ExaModels, PowerModels, NLPModels, NLPModelsIpopt, NLPModelsJuMP, ..NLPTest

import ..BACKENDS
import ..ad_tolerance, ..sol_tolerance, ..solver_tolerance

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

function nlp_legacy_runtests()
    jm = JuMP.Model()

    JuMP.@variable(jm, x[1:10])
    JuMP.@NLobjective(jm, Min, sum(x[i] for i=1:10))

    @test_throws ErrorException ExaModel(jm)

    jm = JuMP.Model(() -> ExaModels.Optimizer(NLPModelsIpopt.ipopt))
    @test_throws ErrorException optimize!(jm)
end
    
function fixed_variable_e2etest()
    N=5
    jm = JuMP.Model()

    JuMP.@variable(jm, x[1:N])
    JuMP.fix(x[1], 1.0)
    JuMP.@constraint(jm, sum(x) == 1.0)
    JuMP.@objective(jm, Min, sum(2*x[i]^2 for i = 1:N))

    em = ExaModel(jm)
    @test only(em.meta.lcon) == only(em.meta.ucon) == 1.0

    # em.cons is a Tuple: (ConstraintAugmentation{Null}, ConstraintAugmentation{Pair}, Constraint{Null{Nothing}})
    @test em.cons[1] isa ExaModels.ConstraintAugmentation
    @test em.cons[1].f.f isa ExaModels.Null

    @test em.cons[2] isa ExaModels.ConstraintAugmentation
    @test em.cons[2].f.f isa Pair

    @test typeof(em.cons[2].f.f.second) <: ExaModels.Node2{
        typeof(*),
        ExaModels.Var{T1},
        T2,
    } where {T1<:ExaModels.DataIndexed,T2<:ExaModels.DataIndexed}

    @test em.cons[3] isa ExaModels.Constraint
    @test em.cons[3].f.f isa ExaModels.Null{Nothing}

    @test em.objs[1].f.f isa ExaModels.Null
    @test typeof(em.objs[2].f.f) <: ExaModels.Node2{
        typeof(*),
        T1,
        ExaModels.Node1{typeof(abs2),ExaModels.Var{T2}},
    } where {T1<:ExaModels.DataIndexed,T2<:ExaModels.DataIndexed}

    jm = JuMP.Model()

    JuMP.@variable(jm, x[1:N])
    JuMP.@variable(jm, p in JuMP.Parameter(1.0))
    JuMP.@constraint(jm, sum(x) == p)
    JuMP.@objective(jm, Min, sum(x))

    em = ExaModel(jm)
    @test only(em.meta.lcon) == only(em.meta.ucon) == 0.0
    @test only(em.θ) == 1.0
    # em.cons: (ConstraintAugmentation{Null}, ConstraintAugmentation{Pair/Param}, ConstraintAugmentation{Pair/Var}, Constraint{Null{Nothing}})
    @test em.cons[1] isa ExaModels.ConstraintAugmentation
    @test em.cons[1].f.f isa ExaModels.Null
    @test em.cons[2] isa ExaModels.ConstraintAugmentation
    @test em.cons[2].f.f isa Pair
    @test typeof(em.cons[2].f.f.second) <: ExaModels.Node2{
        typeof(*),
        ExaModels.ParameterNode{T1},
        T2,
    } where {T1<:ExaModels.DataIndexed,T2<:ExaModels.DataIndexed}
    @test em.cons[3] isa ExaModels.ConstraintAugmentation
    @test em.cons[3].f.f isa Pair
    @test typeof(em.cons[3].f.f.second) <: ExaModels.Node2{
        typeof(*),
        ExaModels.Var{T1},
        T2,
    } where {T1<:ExaModels.DataIndexed,T2<:ExaModels.DataIndexed}
    @test em.cons[4] isa ExaModels.Constraint
    @test em.cons[4].f.f isa ExaModels.Null{Nothing}

    jm = JuMP.Model()
    JuMP.@variable(jm, x)
    @test_broken em = ExaModel(jm)  # FIXME: support feasibility problems?

    return jm
end
function no_constraints_e2etest()
    N=5
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:N])
    JuMP.@objective(jm, Max, sum(sin(x[i]) for i = 1:N))

    em = ExaModel(jm)

    @test length(em.cons) == 1
    @test em.cons[1] isa ExaModels.Constraint

    @test length(em.objs) == 1
    @test em.meta.nnzo == N
    @test em.meta.nnzh == N
    point = collect(range(-0.4, 0.4; length = N))
    @test NLPModels.obj(em, point) ≈ sum(sin, point)
    @test NLPModels.grad(em, point) ≈ cos.(point)

    N=5
    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:N])
    JuMP.@objective(jm, Max, sin(sum(x[i] for i = 1:N)))

    em = ExaModel(jm)

    @test length(em.cons) == 1
    @test em.cons[1] isa ExaModels.Constraint

    @test length(em.objs) == 1
    @test em.meta.nnzo == N
    @test em.meta.nnzh == N * (N + 1) ÷ 2
    point = collect(range(-0.4, 0.4; length = N))
    @test NLPModels.obj(em, point) ≈ sin(sum(point))
    @test NLPModels.grad(em, point) ≈ fill(cos(sum(point)), N)
end
function generic_e2etest()
    N=5
    jm = JuMP.GenericModel{Float32}()
    JuMP.@variable(jm, x[1:N])
    JuMP.@constraint(jm, sum(x) == 1.0f0)
    JuMP.@objective(jm, Min, sum(x[i]^2 for i = 1:N))

    em = ExaModel(jm)

    @test typeof(em) <: ExaModel{Float32}
    @test typeof(getindex.(em.cons[2].itr, 2)) <: Vector{Float32}
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

function _jacobian_matrix(model, x)
    rows = zeros(Int, model.meta.nnzj)
    cols = zeros(Int, model.meta.nnzj)
    values = zeros(eltype(x), model.meta.nnzj)
    NLPModels.jac_structure!(model, rows, cols)
    NLPModels.jac_coord!(model, x, values)

    jacobian = zeros(eltype(x), model.meta.ncon, model.meta.nvar)
    for k in eachindex(values)
        jacobian[rows[k], cols[k]] += values[k]
    end
    return jacobian
end

function _hessian_matrix(model, x, y; obj_weight)
    rows = zeros(Int, model.meta.nnzh)
    cols = zeros(Int, model.meta.nnzh)
    values = zeros(eltype(x), model.meta.nnzh)
    NLPModels.hess_structure!(model, rows, cols)
    NLPModels.hess_coord!(model, x, y, values; obj_weight = obj_weight)

    hessian = zeros(eltype(x), model.meta.nvar, model.meta.nvar)
    for k in eachindex(values)
        hessian[rows[k], cols[k]] += values[k]
        if rows[k] != cols[k]
            hessian[cols[k], rows[k]] += values[k]
        end
    end
    return hessian
end

function _test_callback_equivalence(jump_model, points)
    translated = ExaModel(jump_model)
    reference = MathOptNLPModel(jump_model)
    # Constraint rows may be ordered differently by the two adapters. Restrict
    # this helper to models with at most one constraint, where direct callback
    # comparison is unambiguous.
    @assert translated.meta.ncon <= 1
    T = eltype(translated.meta.x0)
    y = T(0.37) .* collect(T, 1:translated.meta.ncon)
    obj_weight = T(0.61)

    for x in points
        @test NLPModels.obj(translated, x) ≈ NLPModels.obj(reference, x)
        @test NLPModels.cons(translated, x) ≈ NLPModels.cons(reference, x)
        @test NLPModels.grad(translated, x) ≈ NLPModels.grad(reference, x)
        @test _jacobian_matrix(translated, x) ≈ _jacobian_matrix(reference, x)
        @test _hessian_matrix(translated, x, y; obj_weight) ≈
              _hessian_matrix(reference, x, y; obj_weight)
    end
end

function derivative_sparsity_tests()
    @testset "nonlinear constraint derivative sparsity" begin
        branch = JuMP.Model()
        @variable(branch, p)
        @variable(branch, vmf)
        @variable(branch, vmt)
        @variable(branch, vaf)
        @variable(branch, vat)
        @constraint(
            branch,
            p -
            1.2vmf^2 -
            0.7vmf * vmt * cos(vaf - vat) -
            0.3vmf * vmt * sin(vaf - vat) == 0.0,
        )
        @objective(branch, Min, p)

        translated_branch = ExaModel(branch)
        @test translated_branch.meta.nnzj == 5
        @test translated_branch.meta.nnzh == 10

        jacobian_rows = zeros(Int, translated_branch.meta.nnzj)
        jacobian_cols = zeros(Int, translated_branch.meta.nnzj)
        NLPModels.jac_structure!(
            translated_branch,
            jacobian_rows,
            jacobian_cols,
        )
        @test length(unique(zip(jacobian_rows, jacobian_cols))) ==
              translated_branch.meta.nnzj

        # Hessian coordinates are unique here because this model has one
        # constraint row. Different rows may legitimately repeat coordinates.
        hessian_rows = zeros(Int, translated_branch.meta.nnzh)
        hessian_cols = zeros(Int, translated_branch.meta.nnzh)
        NLPModels.hess_structure!(
            translated_branch,
            hessian_rows,
            hessian_cols,
        )
        @test length(unique(zip(hessian_rows, hessian_cols))) ==
              translated_branch.meta.nnzh

        _test_callback_equivalence(
            branch,
            [
                [0.2, 1.0, 0.9, 0.1, -0.2],
                [-0.4, 1.1, 1.05, -0.3, 0.25],
            ],
        )

        repeated = JuMP.Model()
        @variable(repeated, x)
        @variable(repeated, y)
        @constraint(repeated, sin(x) + x^2 + cos(x - y) == 0.0)
        @objective(repeated, Min, x + y)

        translated_repeated = ExaModel(repeated)
        @test translated_repeated.meta.nnzj == 2
        @test translated_repeated.meta.nnzh == 3
        _test_callback_equivalence(
            repeated,
            [[0.3, -0.7], [1.2, 0.4]],
        )
    end

    @testset "nonlinear objective derivative sparsity" begin
        objective = JuMP.Model()
        @variable(objective, x)
        @variable(objective, y)
        @objective(objective, Min, sin(x) + x^2 + cos(x - y) + 2.5)

        translated_objective = ExaModel(objective)
        @test translated_objective.meta.nnzo == 2
        @test translated_objective.meta.nnzh == 3
        _test_callback_equivalence(
            objective,
            [[0.3, -0.7], [1.2, 0.4]],
        )

        nested_objective = JuMP.Model()
        @variable(nested_objective, x)
        @variable(nested_objective, y)
        @objective(
            nested_objective,
            Min,
            exp(sin(x) + x^2 + cos(x - y)),
        )

        translated_nested = ExaModel(nested_objective)
        @test translated_nested.meta.nnzo == 2
        @test translated_nested.meta.nnzh == 3
        _test_callback_equivalence(
            nested_objective,
            [[0.3, -0.7], [1.2, 0.4]],
        )

        parameter_objective = JuMP.Model()
        @variable(parameter_objective, x[1:8])
        @variable(parameter_objective, p in JuMP.Parameter(0.4))
        @objective(
            parameter_objective,
            Min,
            sum(sin(p * x[i]) for i = 1:8),
        )

        translated_parameter_objective = ExaModel(parameter_objective)
        @test length(translated_parameter_objective.objs) == 1
        @test translated_parameter_objective.meta.nnzo == 8
        @test translated_parameter_objective.meta.nnzh == 8
        parameter_point = collect(range(-0.7, 0.7; length = 8))
        @test NLPModels.obj(
            translated_parameter_objective,
            parameter_point,
        ) ≈ sum(sin.(0.4 .* parameter_point))
        @test NLPModels.grad(
            translated_parameter_objective,
            parameter_point,
        ) ≈ 0.4 .* cos.(0.4 .* parameter_point)

        N = 20
        coupled_objective = JuMP.Model()
        @variable(coupled_objective, x[1:N])
        @objective(
            coupled_objective,
            Min,
            sum(
                100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for
                i = 2:N
            ),
        )
        translated_coupled = ExaModel(coupled_objective)
        @test 2N - 1 < translated_coupled.meta.nnzh < 4(N - 1)
        @test length(translated_coupled.objs) == 2
        _test_callback_equivalence(
            coupled_objective,
            [collect(range(-0.5, 0.5; length = N))],
        )
    end

    @testset "nonlinear aliasing and batching" begin
        mixed = JuMP.Model()
        @variable(mixed, z[1:4])
        @constraint(mixed, sin(z[1] * z[2]) == 0.0)
        @constraint(mixed, sin(z[3] * z[3]) == 0.0)
        @constraint(mixed, sin(z[3] * z[4]) == 0.0)
        @objective(mixed, Min, sum(z))

        translated_mixed = ExaModel(mixed)
        @test translated_mixed.meta.nnzj == 5
        @test translated_mixed.meta.nnzh == 7
        mixed_point = [0.2, -0.4, 0.7, 1.1]
        @test NLPModels.cons(translated_mixed, mixed_point) ≈
              [
            sin(mixed_point[1] * mixed_point[2]),
            sin(mixed_point[3]^2),
            sin(mixed_point[3] * mixed_point[4]),
        ]

        K = 4
        batched = JuMP.Model()
        @variable(batched, p[1:K])
        @variable(batched, vmf[1:K])
        @variable(batched, vmt[1:K])
        @variable(batched, vaf[1:K])
        @variable(batched, vat[1:K])
        @constraint(
            batched,
            [i = 1:K],
            p[i] -
            1.2vmf[i]^2 -
            0.7vmf[i] * vmt[i] * cos(vaf[i] - vat[i]) -
            0.3vmf[i] * vmt[i] * sin(vaf[i] - vat[i]) == 0.0,
        )
        @objective(batched, Min, sum(p))

        translated_batched = ExaModel(batched)
        @test length(translated_batched.cons) == 2
        @test translated_batched.meta.nnzj == 5K
        @test translated_batched.meta.nnzh == 10K
        batched_point = vcat(
            collect(0.1:0.1:0.4),
            fill(1.0, K),
            fill(0.9, K),
            collect(0.05:0.05:0.2),
            collect(-0.2:0.05:-0.05),
        )
        @test NLPModels.cons(translated_batched, batched_point) ≈
              [
            batched_point[i] -
            1.2batched_point[K+i]^2 -
            0.7batched_point[K+i] *
            batched_point[2K+i] *
            cos(batched_point[3K+i] - batched_point[4K+i]) -
            0.3batched_point[K+i] *
            batched_point[2K+i] *
            sin(batched_point[3K+i] - batched_point[4K+i]) for i = 1:K
        ]
    end

    @testset "nonlinear parameters and nested expressions" begin
        repeated_parameter = JuMP.Model()
        @variable(repeated_parameter, x)
        @variable(repeated_parameter, p in JuMP.Parameter(0.4))
        @constraint(
            repeated_parameter,
            sin(p * x) + cos(p * x) + p == 0.0,
        )
        @objective(repeated_parameter, Min, x)

        translated_parameter = ExaModel(repeated_parameter)
        @test translated_parameter.meta.nnzj == 1
        @test NLPModels.cons(translated_parameter, [0.7]) ≈
              [sin(0.4 * 0.7) + cos(0.4 * 0.7) + 0.4]

        parameter_only = JuMP.Model()
        @variable(parameter_only, x)
        @variable(parameter_only, p in JuMP.Parameter(0.4))
        @constraint(parameter_only, sin(p) + p^2 == 0.0)
        @objective(parameter_only, Min, x)

        translated_parameter_only = ExaModel(parameter_only)
        @test translated_parameter_only.meta.nnzj == 0
        @test NLPModels.cons(translated_parameter_only, [0.7]) ≈
              [sin(0.4) + 0.4^2]

        nested_affine = JuMP.Model()
        @variable(nested_affine, x)
        @variable(nested_affine, y)
        @constraint(nested_affine, sin(2x + 3y) + x == 0.0)
        @objective(nested_affine, Min, x + y)

        translated_affine = ExaModel(nested_affine)
        @test translated_affine.meta.nnzj == 2
        @test translated_affine.meta.nnzh == 3
        _test_callback_equivalence(
            nested_affine,
            [[0.3, -0.7], [1.2, 0.4]],
        )

        nested_quadratic = JuMP.Model()
        @variable(nested_quadratic, x)
        @variable(nested_quadratic, y)
        @constraint(nested_quadratic, sin(x^2 + x * y) + x == 0.0)
        @objective(nested_quadratic, Min, x + y)

        translated_quadratic = ExaModel(nested_quadratic)
        @test translated_quadratic.meta.nnzj == 2
        @test translated_quadratic.meta.nnzh == 3
        _test_callback_equivalence(
            nested_quadratic,
            [[0.3, -0.7], [1.2, 0.4]],
        )
    end

    @testset "multiple nonlinear shapes and Float32" begin
        shapes = JuMP.Model()
        @variable(shapes, x)
        @variable(shapes, y)
        @variable(shapes, z)
        @constraint(shapes, sin(x) == 0.0)
        @constraint(shapes, exp(y + z) - 1.0 == 0.0)
        @constraint(shapes, cos(x * y) + z == 0.0)
        @objective(shapes, Min, x + y + z)

        translated_shapes = ExaModel(shapes)
        @test length(translated_shapes.cons) == 4
        @test translated_shapes.meta.nnzj == 6
        shapes_point = [0.2, -0.4, 0.7]
        @test NLPModels.cons(translated_shapes, shapes_point) ≈
              [
            sin(shapes_point[1]),
            exp(shapes_point[2] + shapes_point[3]) - 1.0,
            cos(shapes_point[1] * shapes_point[2]) + shapes_point[3],
        ]

        float_model = JuMP.GenericModel{Float32}()
        @variable(float_model, x)
        @variable(float_model, y)
        @constraint(
            float_model,
            sin(x) + x^2 + cos(x - y) == 0.0f0,
        )
        @objective(float_model, Min, x + y)

        translated_float = ExaModel(float_model)
        @test typeof(translated_float) <: ExaModel{Float32}
        @test translated_float.meta.nnzj == 2
        @test translated_float.meta.nnzh == 3
        float_point = Float32[0.3, -0.7]
        @test eltype(NLPModels.cons(translated_float, float_point)) ==
              Float32
        difference = float_point[1] - float_point[2]
        @test NLPModels.cons(translated_float, float_point) ≈
              Float32[
            sin(float_point[1]) +
            float_point[1]^2 +
            cos(difference)
        ]
        @test NLPModels.grad(translated_float, float_point) ≈
              ones(Float32, 2)
        @test _jacobian_matrix(translated_float, float_point) ≈
              reshape(
            Float32[
                cos(float_point[1]) +
                2float_point[1] -
                sin(difference),
                sin(difference),
            ],
            1,
            2,
        )
        expected_hessian = Float32[
            -sin(float_point[1])+2-cos(difference) cos(difference)
            cos(difference) -cos(difference)
        ]
        @test _hessian_matrix(
            translated_float,
            float_point,
            Float32[0.37];
            obj_weight = 0.61f0,
        ) ≈ 0.37f0 .* expected_hessian
    end
end

function runtests()
    @testset "JuMP Interface test" begin
        derivative_sparsity_tests()
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
                    dsol = dual.(all_constraints(jm, include_variable_in_set_constraints = true))

                    set_optimizer(jm, () -> ExaModels.Optimizer(ipopt))
                    set_optimizer_attribute(jm, "print_level", 0)
                    optimize!(jm)
                    sol2 = value.(all_variables(jm))
                    dsol2 = dual.(all_constraints(jm, include_variable_in_set_constraints = true))
                    @test sol ≈ sol2 atol = sol_tolerance(eltype(sol), eltype(sol2))
                    @test dsol ≈ dsol2 atol = sol_tolerance(eltype(sol), eltype(sol2))

                    for backend in BACKENDS
                        @testset "$backend" begin
                            m = WrapperNLPModel(ExaModel(jm; backend = backend))
                            result = ipopt(m; print_level = 0, tol = solver_tolerance(eltype(m.inner.meta.x0)))

                            @test sol ≈ result.solution atol = sol_tolerance(eltype(m.inner.meta.x0))
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
        @testset "NLP legacy test" begin
            nlp_legacy_runtests()
        end
    end
end

end # module
