# # [Two-Stage Optimization](@id two_stage)
# ExaModels supports two-stage optimization problems. This feature enables efficient modeling of optimization problems where decisions are made in two stages:
#
# - **Design (first-stage) variables**: Decisions made before uncertainty is revealed, shared across all scenarios
# - **Recourse (second-stage) variables**: Scenario-specific decisions made after uncertainty is revealed
# - **Scenarios**: Each scenario has its own parameters θ that affect the objective and constraints

using ExaModels, NLPModelsIpopt

# Define the problem dimensions and scenario parameters:
ns = 3   ## number of scenarios
nv = 2   ## recourse variables per scenario
nd = 1   ## design variables
weight = 1.0 / ns

# To annotate the scenario for each variable and constraint, we start with a `TwoStageExaCore` that supports scenario annotations.
core = TwoStageExaCore(ns; concrete = Val(true))

# Design variables are shared across all scenarios — add them without `EachScenario()`.
core, d = add_var(core, nd; start = 1.0, lvar = 0.0, uvar = Inf)

# Recourse variables are per-scenario — use `EachScenario()` to replicate them.
v = @add_var(core, EachScenario(), nv; start = 1.0, lvar = 0.0, uvar = Inf)

# Per-scenario constraints use `EachScenario()`.
@add_con(core, EachScenario(), (v[(s-1)*nv+1] - v[(s-1)*nv+2]^2 for s in 1:ns); lcon = 0.0)

# Objectives can mix design and recourse variables.
@add_obj(core, d[1]^2)
@add_obj(core, weight * (v[(s-1)*nv+i] - d[1])^2 for s in 1:ns, i in 1:nv)

m = ExaModel(core)

# Now we can solve the model as usual.
ipopt(m)
# If the solver knows how to exploit the scenario structure, the structure-exploiting method can be used.
