# # [Parameters](@id parameters)
# Parameters act like fixed variables. Internally, ExaModels keeps track of where parameters appear in the model, making it possible to efficiently modify their value without rebuilding the entire model.

# ### Creating Parametric Models
# 
# Let's modify the example in [Getting Started](@ref index) to use parameters. Suppose we want to make the penalty coefficient in the objective function adjustable:

# First, let's create a core:
using ExaModels, NLPModelsIpopt
c_param = ExaCore()

# Adding parameters is very similar to adding variables -- just pass a vector of values denoting the initial values.
θ = parameter(c_param, [100.0, 1.0])  # [penalty_coeff, offset]

# Define the variables as before:
N = 10
x_p = variable(c_param, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))

# Now we can use the parameters in our objective function just like variables:
objective(c_param, θ[1] * (x_p[i-1]^2 - x_p[i])^2 + (x_p[i-1] - θ[2])^2 for i = 2:N)

# Add the same constraints as before:
constraint(
    c_param,
    3x_p[i+1]^3 + 2 * x_p[i+2] - 5 + sin(x_p[i+1] - x_p[i+2])sin(x_p[i+1] + x_p[i+2]) + 4x_p[i+1] -
    x_p[i]exp(x_p[i] - x_p[i+1]) - 3 for i = 1:(N-2)
)

# Create the model as before:
m_param = ExaModel(c_param)

# Solve with original parameters:
result1 = ipopt(m_param)
println("Original objective: $(result1.objective)")

# Now change the penalty coefficient and solve again:
set_parameter!(c_param, θ, [200.0, 1.0])  # Double the penalty coefficient
result2 = ipopt(m_param)
println("Modified penalty objective: $(result2.objective)")

# Try a different offset parameter:
set_parameter!(c_param, θ, [200.0, 0.5])  # Change the offset in the objective
result3 = ipopt(m_param)
println("Modified offset objective: $(result3.objective)")
