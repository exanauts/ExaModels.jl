# # JuMP Interface

# We have an experimental interface to JuMP model. A JuMP model can be directly converted to a `ExaModel`. It is as simple as this:

using ExaModels, JuMP

N = 10
jm = Model()

@variable(jm, x[i = 1:N], start = mod(i, 2) == 1 ? -1.2 : 1.0)
@constraint(
    jm,
    s[i = 1:N-2],
    3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
    x[i]exp(x[i] - x[i+1]) - 3 == 0.0
)
@objective(jm, Min, sum(100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N))

em = ExaModel(jm)

# Here, note that only scalar objective/constraints created via `@constraint` and `@objective` API are supported. Older syntax like `@NLconstraint` and `@NLobjective` are not supported.
# We can solve the model using any of the solvers supported by ExaModels. For example, we can use Ipopt:

using NLPModelsIpopt

result = ipopt(em)
