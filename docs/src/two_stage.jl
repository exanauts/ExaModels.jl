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

# To annotate the scenario for each variable and constraint, we need to start with a special ExaCore that supports such scenario annotations, which can be created by calling `TwoStageExaCore(ns, concrete = Val(true))`.
core = TwoStageExaCore(ns, concrete = Val(true))

# Now we can define the design variable and recourse variables. The `scenario` keyword argument allows us to specify which scenario(s) each variable belongs to. For the design variable `d`, we set `scenario = 0` to indicate that it is shared across all scenarios. 
@add_var(core, d; start = 1.0, lvar = 0.0, uvar = Inf, scenario = 0)  ## design variable d

# For the recourse variables `v`, we specify `scenario = [i for i=1:ns, j=1:nv]` to indicate that each variable `v[s,i]` belongs to scenario `s`. This allows us to define scenario-specific constraints and objectives that involve these recourse variables.
@add_var(core, v, ns, nv; start = 1.0, lvar = 0.0, uvar = Inf, scenario = [i for i=1:ns, j=1:nv])  ## recourse variables v

# Now we can define the constraints and objective function. The `scenario` keyword argument in the `constraint` and `objective` functions allows us to specify which scenario(s) each constraint or objective term belongs to. 
@add_con(core, v[s,1] - v[s,2]^2 for s in 1:ns; lcon = 0.0, scenario = 1:ns)

@add_obj(core, d^2)
@add_obj(core, weight * (v[s,i] - d)^2 for s in 1:ns, i in 1:nv)

m = ExaModel(core)

# Now we can solve the model as usual. 
ipopt(m) 
# If the solver knows how to exploit the scenario structure, the structure-exploiting method can be used.
