# # [Two-Stage Optimization](@id two_stage)
# ExaModels supports two-stage optimization problems through the `TwoStageExaModel`. This feature enables efficient modeling of optimization problems where decisions are made in two stages:
#
# - **Design (first-stage) variables**: Decisions made before uncertainty is revealed, shared across all scenarios
# - **Recourse (second-stage) variables**: Scenario-specific decisions made after uncertainty is revealed
# - **Scenarios**: Each scenario has its own parameters θ that affect the objective and constraints
#
# The key advantage of `TwoStageExaModel` is that all scenarios share one compiled expression pattern, achieving true SIMD parallelism on GPUs while maintaining the block-structured nature of the problem.

# ## Problem Formulation
# A typical two-stage program has the form:
# ```math
# \begin{aligned}
# \min_{d, \{v_i\}} \quad & f(d) + \sum_{i=1}^{S} w_i \cdot g_i(d, v_i; \theta_i) \\
# \text{s.t.} \quad & h_i(d, v_i; \theta_i) = 0, \quad i = 1, \ldots, S \\
# & d \in \mathcal{D}, \quad v_i \in \mathcal{V}_i
# \end{aligned}
# ```
# where:
# - $d$ are design (first-stage) variables
# - $v_i$ are recourse (second-stage) variables for scenario $i$
# - $\theta_i$ are the parameters for scenario $i$
# - $S$ is the number of scenarios
# - $w_i$ are scenario weights (e.g., probabilities)

# ## Building a Two-Stage Model
# Let's build a simple two-stage problem. Consider minimizing design costs plus expected recourse costs:
# ```math
# \begin{aligned}
# \min_{d, \{v_i\}} \quad & d^2 + \frac{1}{S} \sum_{i=1}^{S} (v_i - \theta_i)^2 \\
# \text{s.t.} \quad & v_i \geq d, \quad i = 1, \ldots, S
# \end{aligned}
# ```
# Here, the design variable $d$ sets a minimum level that all recourse variables must meet, while each scenario tries to match its target $\theta_i$.

using ExaModels, NLPModelsIpopt

# Define the problem dimensions and scenario parameters:
ns = 3   ## number of scenarios
nv = 1   ## recourse variables per scenario
nd = 1   ## design variables
θ_sets = [[2.0], [4.0], [6.0]]  ## θ₁=2, θ₂=4, θ₃=6
weight = 1.0 / ns

# Build the model using the do-block syntax. The build function receives the `ExaCore` `c`,
# variable handles `d` and `v`, parameter handle `θ`, and dimension info `ns, nv, nθ`:
model = TwoStageExaModel(nd, nv, ns, θ_sets) do c, d, v, θ, ns, nv, nθ
    ## Design objective: d²
    objective(c, d[1]^2)
    ## Recourse objective: weight * Σᵢ (vᵢ - θᵢ)²
    obj_data = [(i, i, (i - 1) * nθ) for i in 1:ns]
    objective(c, weight * (v[v_idx] - θ[θ_off + 1])^2 for (i, v_idx, θ_off) in obj_data)
    ## Constraints: vᵢ ≥ d (i.e., vᵢ - d ≥ 0)
    con_data = [(i, i) for i in 1:ns]
    constraint(c, v[v_idx] - d[1] for (i, v_idx) in con_data; lcon = 0.0)
end

# The `TwoStageExaModel` constructor takes:
# - `nd`: number of design variables
# - `nv`: number of recourse variables per scenario
# - `ns`: number of scenarios
# - `θ_sets`: vector of parameter vectors, one per scenario
#
# The build function receives:
# - `c`: the `ExaCore` to build on
# - `d`: variable handle for design variables (indices 1:nd)
# - `v`: variable handle for ALL recourse variables (use global indexing)
# - `θ`: parameter handle for ALL parameters (use global indexing)
# - `ns, nv, nθ`: dimensions for building iteration data

# ## Variable Layout
# Variables are stored in a global vector with the layout: `[v₁; v₂; ...; vₛ; d]`
# - Recourse variables for all scenarios come first
# - Design variables are at the end
#
# Use the helper functions to get index ranges:
println("Recourse var indices for scenario 1: ", ExaModels.recourse_var_indices(model, 1))
println("Recourse var indices for scenario 2: ", ExaModels.recourse_var_indices(model, 2))
println("Recourse var indices for scenario 3: ", ExaModels.recourse_var_indices(model, 3))
println("Design var indices: ", ExaModels.design_var_indices(model))
println("Total variables: ", ExaModels.total_vars(model))
println("Total constraints: ", ExaModels.total_cons(model))

# ## Solving and Extracting Solutions
# Solve the model using Ipopt:
result = ipopt(model.model; print_level = 0)
println("\nSolution status: ", result.status)
println("Optimal objective: ", round(result.objective, digits = 4))

# Extract solutions using index ranges:
x_sol = result.solution
d_sol = x_sol[ExaModels.design_var_indices(model)]
println("\nDesign variable d* = ", round(d_sol[1], digits = 4))

for i in 1:ns
    v_sol = x_sol[ExaModels.recourse_var_indices(model, i)]
    println("Recourse variable v$(i)* = ", round(v_sol[1], digits = 4))
end

# For this problem, the optimal solution is d* = θ̄/2 = 2, and all vᵢ* = d* = 2
# (the constraint vᵢ ≥ d is binding for all scenarios since d* ≤ min(θᵢ)).

# ## Updating Parameters
# One powerful feature is the ability to update scenario parameters and re-solve without rebuilding the model. This is useful for:
# - Stochastic programming with sample average approximation (SAA)
# - Sensitivity analysis
# - Online optimization with changing uncertainty

# Update parameters for a single scenario:
ExaModels.set_scenario_parameters!(model, 1, [10.0])  ## Change θ₁ from 2.0 to 10.0

# Or update all scenarios at once:
new_θ_sets = [[10.0], [12.0], [14.0]]
ExaModels.set_all_scenario_parameters!(model, new_θ_sets)

# Re-solve with new parameters:
result2 = ipopt(model.model; print_level = 0)
println("\nAfter parameter update:")
println("New optimal objective: ", round(result2.objective, digits = 4))
x_sol2 = result2.solution
d_sol2 = x_sol2[ExaModels.design_var_indices(model)]
println("New design variable d* = ", round(d_sol2[1], digits = 4))

# ## Variable Bounds and Initial Values
# You can specify bounds and initial values for both design and recourse variables:
# ```julia
# model = TwoStageExaModel(nd, nv, ns, θ_sets;
#     d_start = 1.0,      # initial value for design variables
#     d_lvar = 0.0,       # lower bound for design variables
#     d_uvar = 10.0,      # upper bound for design variables
#     v_start = 0.5,      # initial value for recourse variables
#     v_lvar = 0.0,       # lower bound for recourse variables
#     v_uvar = Inf        # upper bound for recourse variables
# ) do c, d, v, θ, ns, nv, nθ
#     # ... build model
# end
# ```

# ## Building Iteration Data
# The key to using `TwoStageExaModel` is correctly building iteration data for objectives and constraints. Since all scenarios share one compiled pattern, you must:
#
# 1. Create tuples that map scenario indices to global variable/parameter indices
# 2. Use generators that iterate over all scenarios
#
# Here's a more complex example with multiple recourse variables per scenario:
ns2, nv2, nd2 = 2, 2, 2
θ_sets2 = [[1.0, 3.0], [2.0, 2.0]]  ## 2 params per scenario

model2 = TwoStageExaModel(nd2, nv2, ns2, θ_sets2) do c, d, v, θ, ns, nv, nθ
    objective(c, d[1]^2 + d[2]^2)
    ## Recourse objective: Σᵢ Σⱼ (vᵢⱼ - θᵢⱼ)²
    obj_data = [(i, j, (i - 1) * nv + j, (i - 1) * nθ + j) for i in 1:ns for j in 1:nv]
    objective(c, (v[v_idx] - θ[θ_idx])^2 for (i, j, v_idx, θ_idx) in obj_data)
    ## Coupling constraint: v_{i,1} + v_{i,2} = d₁ + d₂ for each scenario
    con_data = [(i, (i - 1) * nv + 1, (i - 1) * nv + 2) for i in 1:ns]
    constraint(c, v[v1] + v[v2] - d[1] - d[2] for (i, v1, v2) in con_data;
               lcon = 0.0, ucon = 0.0)
end

result3 = ipopt(model2.model; print_level = 0)
println("\nMulti-variable example:")
println("Status: ", result3.status)
println("Optimal objective: ", round(result3.objective, digits = 4))

# ## Accessing the Underlying Model
# For advanced use cases, you can access the underlying `ExaModel`:
inner_model = ExaModel.get_model(model)
println("\nUnderlying model type: ", typeof(inner_model))

# This allows you to use any NLPModels-compatible solver or perform custom operations on the model.
