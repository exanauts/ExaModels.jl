# # [Batch Optimization](@id batch)
# ExaModels supports batch optimization through `BatchExaCore`. This feature
# enables efficient evaluation of multiple fully independent optimization instances
# that share identical structure but differ in parameter values.
#
# Unlike `TwoStageExaModel`, which couples instances through shared design variables,
# batch models treat each instance as completely independent. The key advantage
# is that all instances share one compiled expression pattern and are fused into a
# single model for efficient SIMD evaluation.

# ## Problem Formulation
# A batch optimization problem solves `ns` independent instances simultaneously:
# ```math
# \begin{aligned}
# \min_{v_i} \quad & f(v_i; \theta_i), \quad i = 1, \ldots, S \\
# \text{s.t.} \quad & h(v_i; \theta_i) = 0 \\
# & v_i \in \mathcal{V}
# \end{aligned}
# ```
# where each instance has the same structure but different parameters $\theta_i$.

# ## Building a Batch Model
# Use `BatchExaCore(ns)` to create a core for `ns` instances.
# Variables, parameters, objectives, and constraints are defined once —
# the batch structure replicates them across all instances automatically.

using ExaModels, NLPModelsIpopt
import NLPModels

# Define the problem dimensions and instance parameters:
ns = 3   ## number of instances
nv = 1   ## variables per instance

# Create a batch core:
c = BatchExaCore(ns)

# Add variables and parameters:
@add_var(c, v, nv)
@add_par(c, θ, [2.0])

# Define objectives and constraints — these apply to every instance:
@add_obj(c, (v[j] - θ[1])^2 for j in 1:nv)
@add_con(c, g, v[j] for j in 1:nv; lcon = 0.0)

# Build the model:
model = ExaModel(c)

# ## Batch API (NLPModels)
# Batch models implement `AbstractNLPModel` with matrix-valued variables.
# All evaluation functions use matrices of size `(dim, ns)`:

println("Variables per instance: ", NLPModels.get_nvar(model))
println("Constraints per instance: ", NLPModels.get_ncon(model))
println("Number of instances: ", ExaModels.get_nbatch(model))

# Evaluate objectives for all instances at once:
bx = reshape([1.0, 3.0, 5.0], nv, ns)
bf = zeros(ns)
NLPModels.obj!(model, bx, bf)
println("\nObjective values: ", bf)
## instance 1: (1-2)² = 1, instance 2: (3-4)² = 1, instance 3: (5-6)² = 1

# Evaluate gradients:
bg = zeros(nv, ns)
NLPModels.grad!(model, bx, bg)
println("Gradients: ", bg)

# Evaluate constraints:
bc = zeros(NLPModels.get_ncon(model), ns)
NLPModels.cons!(model, bx, bc)
println("Constraints: ", bc)

# ## Solving Batch Models
# There is currently no dedicated batch solver. To solve all instances, wrap
# the batch model in `FlatNLPModel`, which concatenates the `ns` instances
# into a single standard `AbstractNLPModel` that any NLPModels-compatible
# solver can consume:
flat = FlatNLPModel(model)
result = ipopt(flat; print_level = 0)
println("\nSolution status: ", result.status)

# Extract per-instance solutions:
x_sol = result.solution
for i in 1:ns
    v_sol = x_sol[ExaModels.var_indices(model, i)]
    println("Instance $i: v* = ", round(v_sol[1], digits = 4))
end

# ## Per-Instance Parameters
# By default, `@add_par` replicates the same value across all instances.
# To give each instance its own parameter values, pass an `npar × ns`
# matrix to `set_value!` after building the model. Each column supplies the
# parameter vector for one instance.

c3 = BatchExaCore(ns)
@add_var(c3, w, nv)
@add_par(c3, α, [0.0])   ## placeholder — will be overwritten per-instance
@add_obj(c3, (w[j] - α[1])^2 for j in 1:nv)
model3 = ExaModel(c3)

# Set different targets for each instance (npar=1 row, ns=3 columns):
ExaModels.set_value!(model3, α, [2.0 4.0 6.0])

flat3 = FlatNLPModel(model3)
result3 = ipopt(flat3; print_level = 0)
for i in 1:ns
    v_sol = result3.solution[ExaModels.var_indices(model3, i)]
    println("Instance $i: w* = ", round(v_sol[1], digits = 4))
end

# For problems with multiple parameters per instance, supply a matrix with one
# row per parameter and one column per instance. For example, with `npar = 2`
# parameters and `ns = 3` instances:
#
# ```julia
# c4 = BatchExaCore(3)
# @add_var(c4, x, 2)
# @add_par(c4, p, [0.0, 0.0])
# @add_obj(c4, sum((x[j] - p[j])^2 for j in 1:2))
# model4 = ExaModel(c4)
#
# # 2×3 matrix: column i = parameter vector for instance i
# ExaModels.set_value!(model4, p, [1.0 3.0 5.0;
#                                   2.0 4.0 6.0])
# ```
#
# The parameter matrix `model.θ` always has shape `(npar, ns)` and can be
# inspected directly:

println("\nParameter matrix (npar × ns): ", size(model3.θ))
println("Values: ", model3.θ)
