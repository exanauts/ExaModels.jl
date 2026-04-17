module GenOptTest

using Test, JuMP, ExaModels, MadNLP, GenOpt

function quadrotor_test()
    container = GenOpt.ParametrizedArray

    N = 3
    n = 9
    p = 4
    dt = 0.1

    x0_val = zeros(n)
    xf = [1.0, 0, 0, 1, 0, 0, 0, 0, 0]

    Q = zeros(n)
    Q[1] = 1.0; Q[4] = 1.0
    Qf = ones(n)
    R = ones(p)

    g = 9.81

    itr1 = [(i, j, xf[j]) for i in 1:N for j in 1:n if Q[j] != 0]
    itr2 = [(j, xf[j]) for j in 1:n]

    model = Model()
    @variable(model, x[1:(N+1), 1:n])
    @variable(model, u[1:N, 1:p])

    @constraint(model, start[i in 1:n], x[1, i] == x0_val[i], container = container)

    @constraint(model, [i in 1:N], x[i+1, 1] == x[i, 1] + x[i, 2] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 2] == x[i, 2] + (u[i, 1]) * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 3] == x[i, 3] + x[i, 4] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 4] == x[i, 4] + (u[i, 2]) * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 5] == x[i, 5] + x[i, 6] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 6] == x[i, 6] + (u[i, 3] - g) * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 7] == x[i, 7] + u[i, 1] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 8] == x[i, 8] + u[i, 2] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 9] == x[i, 9] + u[i, 4] * dt, container = container)

    @objective(model, Min,
        lazy_sum(0.5 * R[j] * (u[i, j]^2) for i in 1:N, j in 1:p) +
        lazy_sum(0.5 * Q[it[2]] * (x[it[1], it[2]] - it[3])^2 for it in itr1) +
        lazy_sum(0.5 * Qf[it[1]] * (x[N+1, it[1]] - it[2])^2 for it in itr2),
    )

    return model
end

function _get_exa_model(model)
    be = backend(model)
    return be.optimizer.model.optimizer.model
end

function parametric_genopt_test()
    # Same structure as quadrotor_test but with MOI parameters declared first,
    # which shifts MOI variable indices relative to ExaModels internal indices.
    # This tests that the GenOpt extension does NOT assume var_to_idx is the identity.
    container = GenOpt.ParametrizedArray

    N = 3
    n = 9
    p = 4
    dt = 0.1

    x0_val = zeros(n)
    xf = [1.0, 0, 0, 1, 0, 0, 0, 0, 0]

    Q = zeros(n)
    Q[1] = 1.0; Q[4] = 1.0
    Qf = ones(n)
    R = ones(p)

    g = 9.81

    itr1 = [(i, j, xf[j]) for i in 1:N for j in 1:n if Q[j] != 0]
    itr2 = [(j, xf[j]) for j in 1:n]

    model = Model()

    # Parameters come first, shifting MOI variable indices by 2
    @variable(model, par1 in MOI.Parameter(2.0))
    @variable(model, par2 in MOI.Parameter(3.0))

    @variable(model, x[1:(N+1), 1:n])
    @variable(model, u[1:N, 1:p])

    @constraint(model, start[i in 1:n], x[1, i] == x0_val[i], container = container)

    @constraint(model, [i in 1:N], x[i+1, 1] == x[i, 1] + x[i, 2] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 2] == x[i, 2] + (u[i, 1]) * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 3] == x[i, 3] + x[i, 4] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 4] == x[i, 4] + (u[i, 2]) * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 5] == x[i, 5] + x[i, 6] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 6] == x[i, 6] + (u[i, 3] - g) * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 7] == x[i, 7] + u[i, 1] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 8] == x[i, 8] + u[i, 2] * dt, container = container)
    @constraint(model, [i in 1:N], x[i+1, 9] == x[i, 9] + u[i, 4] * dt, container = container)

    @objective(model, Min,
        lazy_sum(0.5 * R[j] * (u[i, j]^2) for i in 1:N, j in 1:p) +
        lazy_sum(0.5 * Q[it[2]] * (x[it[1], it[2]] - it[3])^2 for it in itr1) +
        lazy_sum(0.5 * Qf[it[1]] * (x[N+1, it[1]] - it[2])^2 for it in itr2),
    )

    return model
end

function runtests()
    @testset "GenOpt extension test" begin
        @testset "Quadrotor with ExaModels.Optimizer" begin
            model = quadrotor_test()

            set_optimizer(model, () -> ExaModels.Optimizer(MadNLP.madnlp))
            set_optimizer_attribute(model, "print_level", MadNLP.ERROR)
            optimize!(model)

            obj = objective_value(model)
            @test isapprox(obj, 8.1797, atol = 1e-3)

            # Verify that FunctionGenerator structure is retained in ExaModels
            # (not dismantled into individual scalar constraints).
            # The test has 10 @constraint calls with `container = GenOpt.ParametrizedArray`:
            #  - 1 start constraint with n=9 iterations
            #  - 9 dynamics constraints each with N=3 iterations
            # Plus 1 empty block for standard (non-generator) constraints.
            # If structure were lost, we'd see 36 entries each with 1 iteration.
            m = _get_exa_model(model)
            @test m.meta.ncon == 36
            cons = m.cons
            # Filter out the empty standard-constraint block
            nonempty = filter(c -> length(c.itr) > 0, cons)
            @test length(nonempty) == 10
            itr_lengths = sort([length(c.itr) for c in nonempty])
            @test itr_lengths == [3, 3, 3, 3, 3, 3, 3, 3, 3, 9]
        end

        @testset "GenOpt with MOI parameters" begin
            # Tests that var_to_idx is not assumed to be the identity.
            # Parameters before variables shift MOI indices.
            model = parametric_genopt_test()

            set_optimizer(model, () -> ExaModels.Optimizer(MadNLP.madnlp))
            set_optimizer_attribute(model, "print_level", MadNLP.ERROR)
            optimize!(model)

            # Same problem as quadrotor, should give same objective
            @test isapprox(objective_value(model), 8.1797, atol = 1e-3)
        end
    end
end

end # module
