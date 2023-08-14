```@meta
EditURL = "guide.jl"
```

# Getting Started
ExaModels can create nonlinear prgogramming models and allows solving the created models using NLP solvers (in particular, those that are interfaced with `NLPModels`, such as [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) and [MadNLP](https://github.com/MadNLP/MadNLP.jl). This documentation page will describe how to use `ExaModels` to model and solve nonlinear optimization problems.

We will first consider the following simple nonlinear program [lukvsan1998indefinitely](@cite):
```math
\begin{aligned}
\min_{\{x_i\}_{i=0}^N} &\sum_{i=2}^N  100(x_{i-1}^2-x_i)^2+(x_{i-1}-1)^2\\
\text{s.t.} &  3x_{i+1}^3+2x_{i+2}-5+\sin(x_{i+1}-x_{i+2})\sin(x_{i+1}+x_{i+2})+4x_{i+1}-x_i e^{x_i-x_{i+1}}-3 = 0
\end{aligned}
```
We will follow the following Steps to create the model/solve this optimization problem.
- Step 0: import ExaModels.jl
- Step 1: create a [`ExaCore`](@ref ExaCore) object, wherein we can progressively build an optimization model.
- Step 2: create optimization variables with [`variable`]((@ref variable)), while attaching it to previously created `ExaCore`.
- Step 3 (interchangable with Step 3): create objective function with [`objective`](@ref objective), while attaching it to previously created `ExaCore`.
- Step 4 (interchangable with Step 2): create constraints with [`constraint`](@ref constraint), while attaching it to previously created `ExaCore`.
- Step 5: create an [`ExaModel`](@ref ExaModel) based on the `ExaCore`.

Now, let's jump right in. We import ExaModels via (Step 0):

````julia
using ExaModels
````

Now, all the functions that are necessary for creating model are imported to into `Main`.

An `ExaCore` object can be created simply by (Step 1):

````julia
c = ExaCore()
````

````
An ExaCore

  Float type: ...................... Float64
  Array type: ...................... Vector{Float64}
  Backend: ......................... Nothing

  number of objective patterns: .... 0
  number of constraint patterns: ... 0

````

This is where our optimziation model information will be progressively stored. This object is not yet an `NLPModel`, but it will essentially store all the necessary information.

Now, let's create the optimziation variables. From the problem definition, we can see that we will need $N$ scalar variables. We will choose $N=10$, and create the variable $x\in\mathbb{R}^{N}$ with the follwoing command:

````julia
N = 10
x = variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
````

````
Variable

  x ∈ R^{10}
````

This creates the variable `x`, which we will be able to refer to when we create constraints/objective constraionts. Also, this modifies the information in the `ExaCore` object properly so that later an optimization model can be properly created with the necessary information. Observe that we have used the keyword argument `start` to specify the initial guess for the solution. The variable upper and lower bounds can be specified in a similar manner.

The objective can be set as follows:

````julia
objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
````

````
Objective

  min (...) + ∑_{p ∈ P} f(x,p)

  where |P| = 9
````

The constraints can be set as follows:

````julia
constraint(
    c,
    3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
    x[i]exp(x[i] - x[i+1]) - 3 for i = 1:N-2
)
````

````
Constraint

  s.t. (...)
       g♭ ≤ [g(x,p)]_{p ∈ P} ≤ g♯

  where |P| = 8
````

Finally, we are ready to create an `ExaModel` from the data we have collected in `ExaCore`. Since `ExaCore` includes all the necessary information, we can do this simply by:

````julia
m = ExaModel(c)
````

````
An ExaModel

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

Now, we got an optimization model ready to be solved. This problem can be solved with for example, with the Ipopt solver, as follows.

````julia
using NLPModelsIpopt
result = ipopt(m)
````

````
"Execution stats: first-order stationary"
````

Here, `result` is an `AbstractExecutionStats`, which typically contains the solution information. We can check several information as follows.

````julia
println("Status: $(result.status)")
println("Number of iterations: $(result.iter)")
````

````
Status: first_order
Number of iterations: 6

````

The solution values for variable `x` can be inquired by:

````julia
sol = solution(result, x)
````

````
10-element view(::Vector{Float64}, 1:10) with eltype Float64:
 -0.9505563573613093
  0.9139008176388945
  0.9890905176644905
  0.9985592422681151
  0.9998087408802769
  0.9999745932450963
  0.9999966246997642
  0.9999995512524277
  0.999999944919307
  0.999999930070643
````

ExaModels provide several APIs similar to this:
- [`solution`](@ref solution) inquires the primal solution.
- [`multiplier`](@ref multiplier) inquires the dual solution.
- [`multiplier_L`](@ref multiplier_L) inquires the lower bound dual solution.
- [`multiplier_U`](@ref multiplier_U) inquires the upper bound dual solution.

This concludes a short tutorial on how to use ExaModels to model and solve optimization problems. Want to learn more? Take a look at the following examples, which provide further tutorial on how to use ExaModels.jl. Each of the examples are designed to instruct a few additional techniques.
- [Example: Quadrotor](): modeling multiple types of objective values and constraints.
- [Example: Distillation Column](): using two-dimensional index sets for variables.
- [Example: Optimal Power Flow](): handling complex data and using constraint augmentation.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

