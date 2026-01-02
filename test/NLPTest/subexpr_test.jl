# Test subexpression functionality

"""
Test basic 1D subexpression creation and solving.
Compares a model with subexpressions to an equivalent model without.
"""
function test_subexpr_basic(backend)
    # Model WITHOUT subexpressions
    c1 = ExaCore(; backend = backend)
    x1 = variable(c1, 10; start = 1.0)
    # Objective: sum of (x[i]^2 + x[i+1]^2)^2
    objective(c1, (x1[i]^2 + x1[i + 1]^2)^2 for i in 1:9)
    # Constraint: x[i]^2 - 1 >= 0
    constraint(c1, x1[i]^2 - 1 for i in 1:10; lcon = 0.0)
    m1 = ExaModel(c1)

    # Model WITH subexpressions
    c2 = ExaCore(; backend = backend)
    x2 = variable(c2, 10; start = 1.0)
    # Create subexpression for x[i]^2
    s = subexpr(c2, x2[i]^2 for i in 1:10)
    # Objective using subexpression
    objective(c2, (s[i] + s[i + 1])^2 for i in 1:9)
    # Constraint using subexpression
    constraint(c2, s[i] - 1 for i in 1:10; lcon = 0.0)
    m2 = ExaModel(c2)

    # The model with subexpressions has more variables and constraints
    @test m2.meta.nvar == m1.meta.nvar + 10  # 10 auxiliary variables
    @test m2.meta.ncon == m1.meta.ncon + 10  # 10 defining constraints

    # Solve both models (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result1 = NLPModelsIpopt.ipopt(WrapperNLPModel(m1); print_level = 0)
    result2 = NLPModelsIpopt.ipopt(WrapperNLPModel(m2); print_level = 0)

    # Solutions for original variables should match
    @test result1.status == result2.status
    @test solution(result1, x1) ≈ solution(result2, x2) atol = 1.0e-4

    # Subexpression values should equal x^2
    subexpr_vals = solution(result2, s)
    x_vals = solution(result2, x2)
    return @test subexpr_vals ≈ x_vals .^ 2 atol = 1.0e-6
end

"""
Test multi-dimensional subexpressions with automatic dimension inference.
"""
function test_subexpr_multidim(backend)
    T, N = 5, 4

    c = ExaCore(; backend = backend)
    x = variable(c, 0:T, 0:N; start = 0.5)

    # Create 2D subexpression with Cartesian product - dimensions inferred automatically
    dx = subexpr(c, x[t, i] - x[t - 1, i] for t in 1:T, i in 1:N)

    # Test that subexpression has correct dimensions (inferred from Cartesian product)
    @test dx.size == (1:T, 1:N)
    @test dx.length == T * N

    # Use in objective
    objective(c, dx[t, i]^2 for t in 1:T, i in 1:N)

    # Add some constraints to make it non-trivial
    constraint(c, x[0, i] - 0.0 for i in 0:N)  # Initial condition
    constraint(c, x[T, i] - 1.0 for i in 0:N)  # Final condition

    m = ExaModel(c)

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # Check subexpression values match the definition
    x_sol = solution(result, x)
    dx_sol = solution(result, dx)

    for t in 1:T, i in 1:N
        expected = x_sol[t + 1, i + 1] - x_sol[t, i + 1]  # +1 for 0-based to 1-based
        @test dx_sol[t, i] ≈ expected atol = 1.0e-6
    end
    return
end

"""
Test subexpression with automatic dimension inference from Cartesian product.
"""
function test_subexpr_auto_dims(backend)
    T, N = 3, 2

    c = ExaCore(; backend = backend)
    x = variable(c, 1:T, 1:N; start = 1.0)

    # Subexpr with Cartesian product syntax - dimensions inferred automatically
    s = subexpr(c, x[t, i]^2 + t * i for t in 1:T, i in 1:N)

    # Check dimensions were inferred
    @test s.size == (1:T, 1:N)
    @test s.length == T * N

    # Use in objective
    objective(c, s[t, i] for t in 1:T, i in 1:N)

    m = ExaModel(c)

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    return @test result.status == :first_order
end

"""
Test that subexpressions can be used in both objectives and constraints.
"""
function test_subexpr_in_obj_and_con(backend)
    c = ExaCore(; backend = backend)
    x = variable(c, 5; start = 2.0, lvar = 0.0)

    # Subexpression
    s = subexpr(c, sqrt(x[i]) for i in 1:5)

    # Use in objective
    objective(c, (s[i] - 1)^2 for i in 1:5)

    # Use in constraint
    constraint(c, s[i] + s[i + 1] for i in 1:4; lcon = 1.0, ucon = 3.0)

    m = ExaModel(c)

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # sqrt(x) = 1 at optimum, so x = 1
    @test solution(result, x) ≈ ones(5) atol = 1.0e-4
    return @test solution(result, s) ≈ ones(5) atol = 1.0e-4
end

"""
Test that lifted subexpressions inherit start values from main variables.
The auxiliary variables should get start values computed from the expression
evaluated at the main variables' start values.
"""
function test_subexpr_lifted_start_values(backend)
    c = ExaCore(; backend = backend)

    # Create variables with specific start values
    x = variable(c, 5; start = 3.0)

    # Create lifted subexpression s[i] = x[i]^2
    # With x start = 3.0, the subexpr auxiliary vars should start at 9.0
    s = subexpr(c, x[i]^2 for i in 1:5)

    # Check that the subexpression auxiliary variables have computed start values
    # The start values are stored in c.x0 at the subexpr's offset
    start_vals = c.x0[(s.offset+1):(s.offset+s.length)]
    @test all(Array(start_vals) .≈ 9.0)

    # Also test with parameters
    c2 = ExaCore(; backend = backend)
    θ = parameter(c2, [1.0, 2.0, 3.0])
    x2 = variable(c2, 3; start = 2.0)

    # Subexpression uses both x and θ: s[i] = x[i] * θ[i]
    # With x start = 2.0 and θ = [1,2,3], expect start = [2,4,6]
    s2 = subexpr(c2, x2[i] * θ[i] for i in 1:3)

    start_vals2 = c2.x0[(s2.offset+1):(s2.offset+s2.length)]
    @test Array(start_vals2) ≈ [2.0, 4.0, 6.0]

    # Test that the model solves correctly with these start values
    objective(c2, (x2[i] - s2[i])^2 for i in 1:3)
    constraint(c2, x2[i] - 1 for i in 1:3; lcon = 0.0, ucon = Inf)

    m = ExaModel(c2)
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    return @test result.status == :first_order
end

"""
Test reduced subexpressions (no extra variables/constraints).
"""
function test_subexpr_reduced_basic(backend)
    # Model WITHOUT subexpressions
    c1 = ExaCore(; backend = backend)
    x1 = variable(c1, 10; start = 1.0)
    objective(c1, (x1[i]^2 + x1[i + 1]^2)^2 for i in 1:9)
    constraint(c1, x1[i]^2 - 1 for i in 1:10; lcon = 0.0)
    m1 = ExaModel(c1)

    # Model WITH reduced subexpressions
    c2 = ExaCore(; backend = backend)
    x2 = variable(c2, 10; start = 1.0)
    s = subexpr(c2, x2[i]^2 for i in 1:10; reduced = true)
    objective(c2, (s[i] + s[i + 1])^2 for i in 1:9)
    constraint(c2, s[i] - 1 for i in 1:10; lcon = 0.0)
    m2 = ExaModel(c2)

    # Reduced subexpressions should NOT add variables or constraints
    @test m2.meta.nvar == m1.meta.nvar
    @test m2.meta.ncon == m1.meta.ncon

    # Solve both models (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result1 = NLPModelsIpopt.ipopt(WrapperNLPModel(m1); print_level = 0)
    result2 = NLPModelsIpopt.ipopt(WrapperNLPModel(m2); print_level = 0)

    # Solutions should match
    @test result1.status == result2.status
    return @test solution(result1, x1) ≈ solution(result2, x2) atol = 1.0e-4
end

"""
Test that reduced and lifted subexpressions produce equivalent solutions.
"""
function test_subexpr_lifted_vs_reduced(backend)
    # Model with LIFTED subexpressions
    c1 = ExaCore(; backend = backend)
    x1 = variable(c1, 5; start = 2.0, lvar = 0.0)
    s1 = subexpr(c1, sqrt(x1[i]) for i in 1:5)  # lifted (default)
    objective(c1, (s1[i] - 1)^2 for i in 1:5)
    constraint(c1, s1[i] + s1[i + 1] for i in 1:4; lcon = 1.0, ucon = 3.0)
    m1 = ExaModel(c1)

    # Model with REDUCED subexpressions
    c2 = ExaCore(; backend = backend)
    x2 = variable(c2, 5; start = 2.0, lvar = 0.0)
    s2 = subexpr(c2, sqrt(x2[i]) for i in 1:5; reduced = true)
    objective(c2, (s2[i] - 1)^2 for i in 1:5)
    constraint(c2, s2[i] + s2[i + 1] for i in 1:4; lcon = 1.0, ucon = 3.0)
    m2 = ExaModel(c2)

    # Lifted has more vars/cons
    @test m1.meta.nvar > m2.meta.nvar
    @test m1.meta.ncon > m2.meta.ncon

    # Solve both (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result1 = NLPModelsIpopt.ipopt(WrapperNLPModel(m1); print_level = 0)
    result2 = NLPModelsIpopt.ipopt(WrapperNLPModel(m2); print_level = 0)

    # Both should converge to same solution
    @test result1.status == :first_order
    @test result2.status == :first_order
    return @test solution(result1, x1) ≈ solution(result2, x2) atol = 1.0e-4
end

"""
Test multi-dimensional reduced subexpressions.
"""
function test_subexpr_reduced_multidim(backend)
    T, N = 3, 2

    c = ExaCore(; backend = backend)
    x = variable(c, 0:T, 0:N; start = 0.5)

    # Reduced 2D subexpression
    dx = subexpr(c, x[t, i] - x[t - 1, i] for t in 1:T, i in 1:N; reduced = true)

    # Check dimensions
    @test dx.size == (1:T, 1:N)
    @test dx.length == T * N

    # No extra vars/cons from reduced subexpr
    nvar_before = c.nvar
    ncon_before = c.ncon

    # Use in objective
    objective(c, dx[t, i]^2 for t in 1:T, i in 1:N)

    # Add constraints
    constraint(c, x[0, i] - 0.0 for i in 0:N)
    constraint(c, x[T, i] - 1.0 for i in 0:N)

    m = ExaModel(c)

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    return @test result.status == :first_order
end

"""
Test nested reduced subexpressions.
"""
function test_subexpr_reduced_nested(backend)
    c = ExaCore(; backend = backend)
    x = variable(c, 5; start = 1.0, lvar = 0.1)

    # Nested reduced subexpressions
    s1 = subexpr(c, x[i]^2 for i in 1:5; reduced = true)
    s2 = subexpr(c, s1[i] + s1[i] for i in 1:5; reduced = true)  # 2*x[i]^2

    objective(c, (s2[i] - 2)^2 for i in 1:5)  # minimize (2*x^2 - 2)^2

    m = ExaModel(c)

    # No extra vars/cons from reduced subexprs
    @test m.meta.nvar == 5
    @test m.meta.ncon == 0

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # 2*x^2 = 2 => x = 1
    return @test solution(result, x) ≈ ones(5) atol = 1.0e-4
end

"""
Test mixed lifted and reduced subexpressions.
"""
function test_subexpr_mixed(backend)
    c = ExaCore(; backend = backend)
    x = variable(c, 5; start = 1.0, lvar = 0.1)

    # First subexpr is lifted
    s_lifted = subexpr(c, x[i]^2 for i in 1:5)  # adds 5 vars + 5 cons

    # Second subexpr is reduced, uses the lifted one
    s_reduced = subexpr(c, s_lifted[i] * 2 for i in 1:5; reduced = true)

    objective(c, (s_reduced[i] - 2)^2 for i in 1:5)

    m = ExaModel(c)

    # Only lifted subexpr adds vars/cons
    @test m.meta.nvar == 10  # 5 original + 5 lifted
    @test m.meta.ncon == 5   # 5 defining constraints

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # 2*x^2 = 2 => x = 1
    return @test solution(result, x) ≈ ones(5) atol = 1.0e-4
end

"""
Test reduced subexpressions with 0-based ranges (like in distillation example).
This tests the symbolic indexing when ranges don't start at 1.
"""
function test_subexpr_reduced_0based(backend)
    T = 3

    c = ExaCore(; backend = backend)
    u = variable(c, 0:T; start = 1.0)

    # Reduced subexpressions with 0-based ranges (like distillation column)
    # V[t] should equal u[t] * 2 + 1, NOT u[t-1] * 2 + 1
    V = subexpr(c, u[t] * 2 + 1 for t in 0:T; reduced = true)

    # Objective: minimize sum of (V[t] - 3)^2
    # If V[t] = u[t]*2+1, optimal is u[t] = 1 (V[t] = 3)
    objective(c, (V[t] - 3)^2 for t in 0:T)

    m = ExaModel(c)

    # No extra vars from reduced subexpr
    @test m.meta.nvar == T + 1  # just u[0:T]
    @test m.meta.ncon == 0

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # V[t] = u[t]*2+1 = 3 => u[t] = 1
    return @test solution(result, u) ≈ ones(T + 1) atol = 1.0e-4
end

"""
Test nested reduced subexpressions with 0-based ranges (like distillation VdyA).
"""
function test_subexpr_reduced_0based_nested(backend)
    T, N = 2, 2

    c = ExaCore(; backend = backend)
    x = variable(c, 0:T, 0:N; start = 1.0)

    # First reduced subexpr with 0-based range
    s1 = subexpr(c, x[t, i] + 1 for t in 0:T, i in 0:N; reduced = true)

    # Second reduced subexpr uses the first one
    # s2[t,i] should be s1[t,i] * 2 = (x[t,i] + 1) * 2
    s2 = subexpr(c, s1[t, i] * 2 for t in 0:T, i in 0:N; reduced = true)

    # Objective: minimize sum of (s2[t,i] - 4)^2
    # s2 = (x+1)*2 = 4 => x = 1
    objective(c, (s2[t, i] - 4)^2 for t in 0:T, i in 0:N)

    m = ExaModel(c)

    # No extra vars from reduced subexprs
    @test m.meta.nvar == (T + 1) * (N + 1)
    @test m.meta.ncon == 0

    # Wrap in WrapperNLPModel for GPU compatibility with Ipopt
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # (x+1)*2 = 4 => x = 1
    return @test solution(result, x) ≈ ones(T + 1, N + 1) atol = 1.0e-4
end

"""
Test basic parameter-only subexpressions.
Values should be cached and recomputed only on set_parameter!.
"""
function test_subexpr_param_only_basic(backend)
    c = ExaCore(; backend = backend)

    # Create parameters
    θ = parameter(c, [1.0, 2.0, 3.0, 4.0, 5.0])

    # Create variables
    x = variable(c, 5; start = 1.0)

    # Parameter-only subexpression: weights = θ^2
    weights = subexpr(c, θ[i]^2 for i in 1:5; parameter_only = true)

    # Check that it's a ParameterSubexpr
    @test weights isa ParameterSubexpr
    @test weights.length == 5

    # Use in objective: minimize sum of weights[i] * (x[i] - 1)^2
    objective(c, weights[i] * (x[i] - 1)^2 for i in 1:5)

    m = ExaModel(c)

    # Parameter subexpr should NOT add variables or constraints
    @test m.meta.nvar == 5
    @test m.meta.ncon == 0

    # Solve (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # Optimal solution should be x = 1 (minimizes weighted squares)
    return @test solution(result, x) ≈ ones(5) atol = 1.0e-4
end

"""
Test that parameter-only subexpressions are recomputed on set_parameter!.
"""
function test_subexpr_param_only_update(backend)
    c = ExaCore(; backend = backend)

    # Create parameters with initial values
    θ = parameter(c, [1.0, 2.0, 3.0])

    # Create variables
    x = variable(c, 3; start = 1.0)

    # Parameter-only subexpression: coeffs = θ * 2
    coeffs = subexpr(c, θ[i] * 2 for i in 1:3; parameter_only = true)

    # Use in objective: minimize sum of (x[i] - coeffs[i])^2
    objective(c, (x[i] - coeffs[i])^2 for i in 1:3)

    m = ExaModel(c)

    # Solve with initial parameters
    result1 = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result1.status == :first_order

    # With θ = [1,2,3], coeffs = [2,4,6], optimal x = [2,4,6]
    @test solution(result1, x) ≈ [2.0, 4.0, 6.0] atol = 1.0e-4

    # Update parameters
    set_parameter!(c, θ, [10.0, 20.0, 30.0])

    # Rebuild model (parameters are shared via ExaCore)
    m2 = ExaModel(c)

    # Solve with updated parameters
    result2 = NLPModelsIpopt.ipopt(WrapperNLPModel(m2); print_level = 0)
    @test result2.status == :first_order

    # With θ = [10,20,30], coeffs = [20,40,60], optimal x = [20,40,60]
    return @test solution(result2, x) ≈ [20.0, 40.0, 60.0] atol = 1.0e-4
end

"""
Test multi-dimensional parameter-only subexpressions.
"""
function test_subexpr_param_only_multidim(backend)
    T, N = 3, 2

    c = ExaCore(; backend = backend)

    # Create 2D parameters
    θ = parameter(c, ones(T, N) .* 2.0)

    # Create variables
    x = variable(c, 1:T, 1:N; start = 1.0)

    # Parameter-only subexpression with Cartesian product
    p_weights = subexpr(c, θ[t, i]^2 for t in 1:T, i in 1:N; parameter_only = true)

    # Check dimensions
    @test p_weights.size == (1:T, 1:N)
    @test p_weights.length == T * N

    # Use in objective
    objective(c, p_weights[t, i] * (x[t, i] - 1)^2 for t in 1:T, i in 1:N)

    m = ExaModel(c)

    # No extra vars/cons
    @test m.meta.nvar == T * N
    @test m.meta.ncon == 0

    # Solve (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # x = 1 minimizes the objective
    return @test solution(result, x) ≈ ones(T, N) atol = 1.0e-4
end

"""
Test parameter-only subexpressions in constraints.
"""
function test_subexpr_param_only_in_constraint(backend)
    c = ExaCore(; backend = backend)

    # Parameters define constraint targets
    θ = parameter(c, [1.0, 2.0, 3.0])

    # Variables - start at a feasible point
    x = variable(c, 3; start = 1.0)

    # Parameter-only subexpression for constraint target
    target = subexpr(c, θ[i] for i in 1:3; parameter_only = true)

    # Objective: minimize sum of (x[i] - 2)^2 (optimal at x = 2 without constraints)
    objective(c, (x[i] - 2)^2 for i in 1:3)

    # Constraint: x[i] <= target[i], i.e., x[i] <= θ[i]
    # target = [1, 2, 3], so x[1] <= 1, x[2] <= 2, x[3] <= 3
    # Note: must set ucon = Inf for inequality (default ucon = 0 makes it equality)
    constraint(c, target[i] - x[i] for i in 1:3; lcon = 0.0, ucon = Inf)

    m = ExaModel(c)

    # Solve (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # Optimal: x[1] = 1 (bound), x[2] = 2 (optimal), x[3] = 2 (optimal, bound at 3 not active)
    return @test solution(result, x) ≈ [1.0, 2.0, 2.0] atol = 1.0e-4
end

"""
Test combining parameter-only with lifted/reduced subexpressions.
"""
function test_subexpr_param_only_mixed(backend)
    c = ExaCore(; backend = backend)

    # Parameters
    θ = parameter(c, [1.0, 2.0, 3.0, 4.0, 5.0])

    # Variables
    x = variable(c, 5; start = 1.0, lvar = 0.1)

    # Parameter-only subexpression for weights
    weights = subexpr(c, θ[i] / 10 for i in 1:5; parameter_only = true)  # [0.1, 0.2, 0.3, 0.4, 0.5]

    # Lifted subexpression for x^2
    x_sq = subexpr(c, x[i]^2 for i in 1:5)  # adds 5 vars + 5 cons

    # Objective using both: minimize sum of weights[i] * x_sq[i]
    objective(c, weights[i] * x_sq[i] for i in 1:5)

    # Constraint so problem is bounded
    constraint(c, x[i] - 1 for i in 1:5; lcon = 0.0)  # x >= 1

    m = ExaModel(c)

    # Only lifted adds vars/cons
    @test m.meta.nvar == 10  # 5 original + 5 lifted
    @test m.meta.ncon == 10  # 5 lifted defining + 5 user constraints

    # Solve (wrap in WrapperNLPModel for GPU compatibility with Ipopt)
    result = NLPModelsIpopt.ipopt(WrapperNLPModel(m); print_level = 0)
    @test result.status == :first_order

    # With constraint x >= 1, optimal is x = 1
    return @test solution(result, x) ≈ ones(5) atol = 1.0e-4
end

"""
Run all subexpression tests.
"""
function test_subexpr(backend)
    @testset "Subexpr basic (lifted)" begin
        test_subexpr_basic(backend)
    end

    @testset "Subexpr multi-dim (lifted)" begin
        test_subexpr_multidim(backend)
    end

    @testset "Subexpr auto dims (lifted)" begin
        test_subexpr_auto_dims(backend)
    end

    @testset "Subexpr in obj and con (lifted)" begin
        test_subexpr_in_obj_and_con(backend)
    end

    @testset "Subexpr lifted start values" begin
        test_subexpr_lifted_start_values(backend)
    end

    @testset "Subexpr reduced basic" begin
        test_subexpr_reduced_basic(backend)
    end

    @testset "Subexpr lifted vs reduced" begin
        test_subexpr_lifted_vs_reduced(backend)
    end

    @testset "Subexpr reduced multi-dim" begin
        test_subexpr_reduced_multidim(backend)
    end

    @testset "Subexpr reduced nested" begin
        test_subexpr_reduced_nested(backend)
    end

    @testset "Subexpr mixed lifted and reduced" begin
        test_subexpr_mixed(backend)
    end

    @testset "Subexpr reduced 0-based ranges" begin
        test_subexpr_reduced_0based(backend)
    end

    @testset "Subexpr reduced 0-based nested" begin
        test_subexpr_reduced_0based_nested(backend)
    end

    @testset "Subexpr parameter-only basic" begin
        test_subexpr_param_only_basic(backend)
    end

    @testset "Subexpr parameter-only update" begin
        test_subexpr_param_only_update(backend)
    end

    @testset "Subexpr parameter-only multi-dim" begin
        test_subexpr_param_only_multidim(backend)
    end

    @testset "Subexpr parameter-only in constraint" begin
        test_subexpr_param_only_in_constraint(backend)
    end

    return @testset "Subexpr parameter-only mixed" begin
        test_subexpr_param_only_mixed(backend)
    end
end

