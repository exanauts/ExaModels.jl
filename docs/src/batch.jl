# # [Batch Optimization](@id batch)
# ExaModels supports batch optimization through the `BatchExaModel`. This feature
# enables efficient evaluation of multiple fully independent optimization scenarios
# that share identical structure but differ in parameter values.
#
# Unlike `TwoStageExaModel`, which couples scenarios through shared design variables,
# `BatchExaModel` treats each scenario as completely independent. The key advantage
# is that all scenarios share one compiled expression pattern and are fused into a
# single model for efficient SIMD evaluation.

# ## Problem Formulation
# A batch optimization problem solves `ns` independent scenarios simultaneously:
# ```math
# \begin{aligned}
# \min_{v_i} \quad & f(v_i; \theta_i), \quad i = 1, \ldots, S \\
# \text{s.t.} \quad & h(v_i; \theta_i) = 0 \\
# & v_i \in \mathcal{V}
# \end{aligned}
# ```
# where each scenario has the same structure but different parameters $\theta_i$.

# ## Building a Batch Model
# The builder function defines expressions for a **single scenario**. `BatchExaModel`
# calls it `ns` times internally with per-scenario parameter handles.
# This means you never have to compute global index offsets manually.

using ExaModels, MadNLP

# Define the problem dimensions and scenario parameters as a matrix of size `(nθ, ns)`:
ns = 3   ## number of scenarios
nv = 1   ## variables per scenario
θ_data = [2.0 4.0 6.0]  ## (1, 3) matrix: θ₁=2, θ₂=4, θ₃=6

# Build the model. First create an `ExaCore`, then pass it along with the parameter
# matrix to `BatchExaModel`:
c = ExaCore()
model = BatchExaModel(c, ns, θ_data) do c, θ
    ## Create variables — this is called once per scenario, offsets are automatic
    v = variable(c, nv)
    ## Objective: minimize (v - θ)²
    objective(c, (v[1] - θ[1])^2)
    ## Constraint: v ≥ 0
    constraint(c, v[1]; lcon = 0.0, ucon = Inf)
end

# The builder function receives:
# - `c`: the `ExaCore` — use `variable(c, ...)`, `objective(c, ...)`, `constraint(c, ...)` as usual
# - `θ`: a per-scenario parameter handle (indices 1:nθ)
#
# Variable creation via `variable(c, ...)` works exactly like in a regular `ExaModel`.
# You can set start values, lower/upper bounds, etc.

# ## Batch API (NLPModels)
# `BatchExaModel` implements the `AbstractBatchNLPModel` interface from NLPModels.jl.
# All evaluation functions use matrices of size `(dim, ns)`:
import NLPModels

println("Variables per scenario: ", NLPModels.get_nvar(model))
println("Constraints per scenario: ", NLPModels.get_ncon(model))
println("Number of scenarios: ", NLPModels.get_nbatch(model))

# Evaluate objectives for all scenarios at once:
bx = reshape([1.0, 3.0, 5.0], nv, ns)
bf = zeros(ns)
NLPModels.obj!(model, bx, bf)
println("\nObjective values: ", bf)
## scenario 1: (1-2)² = 1, scenario 2: (3-4)² = 1, scenario 3: (5-6)² = 1

# Evaluate gradients:
bg = zeros(nv, ns)
NLPModels.grad!(model, bx, bg)
println("Gradients: ", bg)

# Evaluate constraints:
bc = zeros(NLPModels.get_ncon(model), ns)
NLPModels.cons!(model, bx, bc)
println("Constraints: ", bc)

# ## Solving via the Fused Model
# For solving, access the underlying fused `ExaModel` and use any NLPModels-compatible
# solver (e.g., MadNLP):
result = madnlp(ExaModels.get_model(model); print_level = MadNLP.ERROR)
println("\nSolution status: ", result.status)
println("Optimal objective: ", round(result.objective, digits = 4))

# Extract per-scenario solutions:
x_sol = result.solution
for i in 1:ns
    v_sol = x_sol[ExaModels.var_indices(model, i)]
    println("Scenario $i: v* = ", round(v_sol[1], digits = 4))
end

# ## A More Complex Example
# Here's a batch model with multiple variables, objectives, and constraints per scenario:
ns2, nv2 = 2, 3
θ_data2 = [1.0 4.0; 2.0 5.0; 3.0 6.0]  ## (3, 2) matrix

c2 = ExaCore()
model2 = BatchExaModel(c2, ns2, θ_data2) do c, θ
    v = variable(c, nv2; start = 1.0, lvar = 0.0, uvar = 10.0)
    ## Objective: Σⱼ (vⱼ - θⱼ)²
    objective(c, (v[j] - θ[j])^2 for j in 1:nv2)
    ## Constraints: sum of all variables ≤ 20
    constraint(c, sum(v[j] for j in 1:nv2); ucon = 20.0)
end

result2 = madnlp(ExaModels.get_model(model2); print_level = MadNLP.ERROR)
println("\nMulti-variable example:")
println("Status: ", result2.status)
x_sol2 = result2.solution
for i in 1:ns2
    v_sol = x_sol2[ExaModels.var_indices(model2, i)]
    println("Scenario $i: v* = ", round.(v_sol, digits = 4))
end

# ## Updating Parameters
# You can update scenario parameters and re-solve without rebuilding the model:

# Update a single scenario:
ExaModels.set_scenario_parameters!(model, 1, [10.0])

# Or update all scenarios at once:
ExaModels.set_all_scenario_parameters!(model, [[10.0], [12.0], [14.0]])

# Re-solve with new parameters:
result3 = madnlp(ExaModels.get_model(model); print_level = MadNLP.ERROR)
println("\nAfter parameter update:")
println("Status: ", result3.status)
x_sol3 = result3.solution
for i in 1:ns
    v_sol = x_sol3[ExaModels.var_indices(model, i)]
    println("Scenario $i: v* = ", round(v_sol[1], digits = 4))
end
