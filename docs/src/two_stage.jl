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

# Build the model using the do-block syntax. The build function receives the `ExaCore` `c`,
# variable handles `d` and `v`, parameter handle `θ`, and dimension info `ns, nv, nθ`:

core = ExaCore(two_stage = Val(true))

d = variable(core; start = 1.0, lvar = 0.0, uvar = Inf, scenario = 0)  ## design variable d
v = variable(core, ns, nv; start = 1.0, lvar = 0.0, uvar = Inf, scenario = [i for i=1:ns, j=1:nv])  ## recourse variables v

constraint(core, v[s,1] - v[s,2]^2 for s in 1:ns; lcon = 0.0, scenario = 1:ns)

objective(core, d^2)
objective(core, weight * (v[s,i] - d)^2 for s in 1:ns, i in 1:nv)

m = ExaModel(core)

ipopt(m) # if the solver knows how to exploit the scenario structure, the structure-exploiting method can be used
