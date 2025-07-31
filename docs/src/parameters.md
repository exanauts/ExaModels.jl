```@meta
EditURL = "parameters.jl"
```

# [Parameters](@id parameters)
Parameters act like fixed variables. Internally, ExaModels keeps track of where parameters appear in the model, making it possible to efficiently modify their value without rebuilding the entire model.

### Creating Parametric Models

Let's modify the example in [Getting Started](@ref guide) to use parameters. Suppose we want to make the penalty coefficient in the objective function adjustable:

First, let's create a core:

````julia
using ExaModels, NLPModelsIpopt
c_param = ExaCore()
````

````
An ExaCore

  Float type: ...................... Float64
  Array type: ...................... Vector{Float64}
  Backend: ......................... Nothing

  number of objective patterns: .... 0
  number of constraint patterns: ... 0

````

Adding parameters is very similar to adding variables -- just pass a vector of values denoting the initial values.

````julia
θ = parameter(c_param, [100.0, 1.0])  # [penalty_coeff, offset]
````

````
Parameter

  θ ∈ R^{2}

````

Define the variables as before:

````julia
N = 10
x_p = variable(c_param, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
````

````
Variable

  x ∈ R^{10}

````

Now we can use the parameters in our objective function just like variables:

````julia
objective(c_param, θ[1] * (x_p[i-1]^2 - x_p[i])^2 + (x_p[i-1] - θ[2])^2 for i = 2:N)
````

````
Objective

  min (...) + ∑_{p ∈ P} f(x,θ,p)

  where |P| = 9

````

Add the same constraints as before:

````julia
constraint(
    c_param,
    3x_p[i+1]^3 + 2 * x_p[i+2] - 5 + sin(x_p[i+1] - x_p[i+2])sin(x_p[i+1] + x_p[i+2]) + 4x_p[i+1] -
    x_p[i]exp(x_p[i] - x_p[i+1]) - 3 for i = 1:(N-2)
)
````

````
Constraint

  s.t. (...)
       g♭ ≤ [g(x,θ,p)]_{p ∈ P} ≤ g♯

  where |P| = 8

````

Create the model as before:

````julia
m_param = ExaModel(c_param)
````

````
An ExaModel{Float64, Vector{Float64}, ...}

  Problem name: Generic
   All variables: ████████████████████ 10     All constraints: ████████████████████ 8     
            free: ████████████████████ 10                free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ████████████████████ 8     
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: (-36.36% sparsity)   75              linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                    nonlinear: ████████████████████ 8     
                                                         nnzj: ( 70.00% sparsity)   24    


````

Solve with original parameters:

````julia
result1 = ipopt(m_param)
println("Original objective: $(result1.objective)")
````

````
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:       24
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:       75

Total number of variables............................:       10
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        8
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.0570000e+03 2.48e+01 2.73e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.0953147e+03 1.49e+01 8.27e+01  -1.0 2.20e+00    -  1.00e+00 1.00e+00f  1
   2  3.2865521e+02 4.28e+00 1.36e+02  -1.0 1.43e+00    -  1.00e+00 1.00e+00f  1
   3  1.3995370e+01 3.09e-01 2.18e+01  -1.0 5.63e-01    -  1.00e+00 1.00e+00f  1
   4  6.2325715e+00 1.73e-02 8.47e-01  -1.0 2.10e-01    -  1.00e+00 1.00e+00f  1
   5  6.2324586e+00 1.15e-05 8.16e-04  -1.7 3.35e-03    -  1.00e+00 1.00e+00h  1
   6  6.2324586e+00 8.35e-12 7.97e-10  -5.7 2.00e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 6

                                   (scaled)                 (unscaled)
Objective...............:   7.8692659500473017e-01    6.2324586324374636e+00
Dual infeasibility......:   7.9746955363607132e-10    6.3159588647976857e-09
Constraint violation....:   8.3546503049092280e-12    8.3546503049092280e-12
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   7.9746955363607132e-10    6.3159588647976857e-09


Number of objective function evaluations             = 7
Number of objective gradient evaluations             = 7
Number of equality constraint evaluations            = 7
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 7
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 6
Total seconds in IPOPT                               = 0.441

EXIT: Optimal Solution Found.
Original objective: 6.232458632437464

````

Now change the penalty coefficient and solve again:

````julia
set_parameter!(c_param, θ, [200.0, 1.0])  # Double the penalty coefficient
result2 = ipopt(m_param)
println("Modified penalty objective: $(result2.objective)")
````

````
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:       24
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:       75

Total number of variables............................:       10
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        8
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  4.0898000e+03 2.48e+01 2.70e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.1810502e+03 1.49e+01 8.27e+01  -1.0 2.20e+00    -  1.00e+00 1.00e+00f  1
   2  6.5137192e+02 4.27e+00 1.36e+02  -1.0 1.43e+00    -  1.00e+00 1.00e+00f  1
   3  2.4064340e+01 3.08e-01 2.18e+01  -1.0 5.62e-01    -  1.00e+00 1.00e+00f  1
   4  8.6476680e+00 1.72e-02 8.45e-01  -1.0 2.10e-01    -  1.00e+00 1.00e+00f  1
   5  8.6474398e+00 1.15e-05 8.07e-04  -1.7 3.39e-03    -  1.00e+00 1.00e+00h  1
   6  8.6474398e+00 8.42e-12 7.91e-10  -5.7 2.03e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 6

                                   (scaled)                 (unscaled)
Objective...............:   5.4592422674820063e-01    8.6474397516914987e+00
Dual infeasibility......:   7.9051456536755353e-10    1.2521750715422049e-08
Constraint violation....:   8.4190432403374871e-12    8.4190432403374871e-12
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   7.9051456536755353e-10    1.2521750715422049e-08


Number of objective function evaluations             = 7
Number of objective gradient evaluations             = 7
Number of equality constraint evaluations            = 7
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 7
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 6
Total seconds in IPOPT                               = 0.003

EXIT: Optimal Solution Found.
Modified penalty objective: 8.647439751691499

````

Try a different offset parameter:

````julia
set_parameter!(c_param, θ, [200.0, 0.5])  # Change the offset in the objective
result3 = ipopt(m_param)
println("Modified offset objective: $(result3.objective)")
````

````
This is Ipopt version 3.14.17, running with linear solver MUMPS 5.8.0.

Number of nonzeros in equality constraint Jacobian...:       24
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:       75

Total number of variables............................:       10
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        8
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  4.0810500e+03 2.48e+01 2.69e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.1767809e+03 1.49e+01 8.26e+01  -1.0 2.20e+00    -  1.00e+00 1.00e+00f  1
   2  6.5050886e+02 4.27e+00 1.36e+02  -1.0 1.43e+00    -  1.00e+00 1.00e+00f  1
   3  2.4276149e+01 3.07e-01 2.18e+01  -1.0 5.61e-01    -  1.00e+00 1.00e+00f  1
   4  8.8465512e+00 1.72e-02 8.43e-01  -1.0 2.09e-01    -  1.00e+00 1.00e+00f  1
   5  8.8451636e+00 1.15e-05 8.04e-04  -1.7 3.40e-03    -  1.00e+00 1.00e+00h  1
   6  8.8451630e+00 8.47e-12 7.88e-10  -5.7 2.05e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 6

                                   (scaled)                 (unscaled)
Objective...............:   5.5805444714793528e-01    8.8451629872947741e+00
Dual infeasibility......:   7.8812124187921384e-10    1.2491721683785540e-08
Constraint violation....:   8.4678930534209940e-12    8.4678930534209940e-12
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   7.8812124187921384e-10    1.2491721683785540e-08


Number of objective function evaluations             = 7
Number of objective gradient evaluations             = 7
Number of equality constraint evaluations            = 7
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 7
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 6
Total seconds in IPOPT                               = 0.003

EXIT: Optimal Solution Found.
Modified offset objective: 8.845162987294774

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

