```@meta
EditURL = "<unknown>/src/guide.jl"
```

# Getting Started
## Automatic Differentiation
`MadDiff` provides a flexible user-interface for evaluating first/second-order derivatives of nonlinear expressions. In the following example, using `MadDiff`, we will create a function, gradient, and Hessian evaluator of the following function:
```math
f(x) = x_1^2 + e^{(x_2^{p_1})/2} + \log(x_2x_3+p_2),
```
where ``x`` is the variable vector, and ``p`` is the parameter vector.

We first import `MadDiff`.

````julia
using MadDiff
````

First, we create a `Source` of `Variable`'s.

````julia
x = Variable()
````

````
x
````

The `Base.getindex!` function is extended so that `x[i]` for any `i` creates an expression for ``x_i``. For example,

````julia
x[2]
````

````
x[2]
````

We can do a similar thing for `Parameter`'s.

````julia
p = Parameter()
p[1]
````

````
p[1]
````

Now, we create the nonlienar expression expression.

````julia
expr = x[1]^2 + exp(x[2]^p[1])/2 + log(x[2]*x[3]+p[2])
````

````
x[1]^2 + exp(x[2]^p[1])/2 + log(x[2]*x[3] + p[2])
````

The function evaluator of the above expression can be created by using `MadDiff.function_evaluator` as follows:

````julia
f = Evaluator(expr)
````

````
Evaluator:
x[1]^2 + exp(x[2]^p[1])/2 + log(x[2]*x[3] + p[2])
````

Now for a given variable and parameter values, the function can be evaluated as follows.

````julia
x0 = [0.,0.5,1.5]
p0 = [2,0.5]
f(x0,p0)
````

````
0.8651562596580804
````

The gradient evaluator can be created as follows:

````julia
y0 = similar(x0)
g = GradientEvaluator(expr)
g(y0,x0,p0)
y0
````

````
3-element Vector{Float64}:
 0.0
 1.8420127083438709
 0.4
````

The Hessian evaluator can be created as follows:

````julia
z0 = zeros(3,3)
h = HessianEvaluator(expr)
h(z0,x0,p0)
z0
````

````
3×3 Matrix{Float64}:
 2.0  0.0        0.0
 0.0  0.486038   0.0
 0.0  0.32      -0.16
````

Note that only lower-triangular entries are evaluated.

The evaluator can be constructed in a sparse format:

````julia
sh = SparseHessianEvaluator(expr);
````

The sparse coordinates are:

````julia
sh.sparsity
````

````
4-element Vector{Tuple{Int64, Int64}}:
 (1, 1)
 (2, 2)
 (3, 2)
 (3, 3)
````

The sparse Hessian can be evaluated as follows:

````julia
z1 = zeros(length(sh.sparsity))
sh(z1,x0,p0)
z1
````

````
4-element Vector{Float64}:
  2.0
  0.4860381250316117
  0.31999999999999995
 -0.16000000000000003
````

## Nonlinear Programming
### Built-in API
MadDiff provides a built-in API for creating nonlinear prgogramming models and allows solving the created models using NLP solvers (in particular, those that are interfaced with `NLPModels`, such as [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) and [MadNLP](https://github.com/MadNLP/MadNLP.jl)). We now use `MadDiff`'s bulit-in API to model the following nonlinear program:
```math
\begin{aligned}
\min_{\{x_i\}_{i=0}^N} &\sum_{i=2}^N  100(x_{i-1}^2-x_i)^2+(x_{i-1}-1)^2\\
\text{s.t.} &  3x_{i+1}^3+2x_{i+2}-5+\sin(x_{i+1}-x_{i+2})\sin(x_{i+1}+x_{i+2})+4x_{i+1}-x_i e^{x_i-x_{i+1}}-3 = 0
\end{aligned}
```
We model the problem with:

````julia
N = 10000
````

````
10000
````

First, we create a `MadDiffModel`.

````julia
m = MadDiffModel()
````

````
MadDiffModel{Float64} (not instantiated).

````

The variables can be created as follows:

````julia
x = [variable(m; start = mod(i,2)==1 ? -1.2 : 1.) for i=1:N];
````

The objective can be set as follows:

````julia
objective(m, sum(100(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i=2:N));
````

The constraints can be set as follows:

````julia
for i=1:N-2
    constraint(m, 3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3 == 0);
end
````

The important last step is instantiating the model. This step must be taken before calling optimizers.

````julia
instantiate!(m)
````

````
MadDiffModel{Float64} (instantiated).
  Problem name: Generic
   All variables: ████████████████████ 10000  All constraints: ████████████████████ 9998  
            free: ████████████████████ 10000             free: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                lower: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                upper: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              low/upp: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           fixed: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                fixed: ████████████████████ 9998  
          infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               infeas: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
            nnzh: ( 99.96% sparsity)   19999           linear: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
                                                    nonlinear: ████████████████████ 9998  
                                                         nnzj: ( 99.97% sparsity)   29994 

  Counters:
             obj: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 grad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 cons: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
        cons_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0             cons_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 jcon: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jgrad: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                  jac: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0              jac_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
         jac_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                jprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0            jprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
       jprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jtprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0           jtprod_lin: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
      jtprod_nln: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                 hess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0                hprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     
           jhess: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0               jhprod: ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ 0     

````

To solve the problem with `Ipopt`,

````julia
using NLPModelsIpopt
ipopt(m);
````

````
This is Ipopt version 3.13.4, running with linear solver mumps.
NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

Number of nonzeros in equality constraint Jacobian...:    29994
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:    19999

Total number of variables............................:    10000
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:     9998
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.5405160e+06 2.48e+01 2.73e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.3512419e+06 1.49e+01 8.27e+01  -1.0 2.20e+00    -  1.00e+00 1.00e+00f  1
   2  1.5156131e+05 4.28e+00 1.36e+02  -1.0 1.43e+00    -  1.00e+00 1.00e+00f  1
   3  6.6755024e+01 3.09e-01 2.18e+01  -1.0 5.63e-01    -  1.00e+00 1.00e+00f  1
   4  6.2338933e+00 1.73e-02 8.47e-01  -1.0 2.10e-01    -  1.00e+00 1.00e+00h  1
   5  6.2324586e+00 1.15e-05 8.16e-04  -1.7 3.35e-03    -  1.00e+00 1.00e+00h  1
   6  6.2324586e+00 8.36e-12 7.97e-10  -5.7 2.00e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 6

                                   (scaled)                 (unscaled)
Objective...............:   7.8692659500479645e-01    6.2324586324379885e+00
Dual infeasibility......:   7.9743417331632266e-10    6.3156786526652763e-09
Constraint violation....:   8.3555384833289281e-12    8.3555384833289281e-12
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   7.9743417331632266e-10    6.3156786526652763e-09


Number of objective function evaluations             = 7
Number of objective gradient evaluations             = 7
Number of equality constraint evaluations            = 7
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 7
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 6
Total CPU secs in IPOPT (w/o function evaluations)   =      0.158
Total CPU secs in NLP function evaluations           =      0.026

EXIT: Optimal Solution Found.

````

### MadDiff as a AD backend of JuMP
MadDiff can be used as an automatic differentiation backend of JuMP. The problem above can be modeled in `JuMP` and solved with `Ipopt` along with `MadDiff`

````julia
using JuMP, Ipopt

m = JuMP.Model(Ipopt.Optimizer)

@variable(m, x[i=1:N], start=mod(i,2)==1 ? -1.2 : 1.)
@NLobjective(m, Min, sum(100(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i=2:N))
@NLconstraint(m, [i=1:N-2], 3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3 == 0)

optimize!(m; differentiation_backend = MadDiffAD())
````

````
This is Ipopt version 3.13.4, running with linear solver mumps.
NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

Number of nonzeros in equality constraint Jacobian...:    29994
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:    19999

Total number of variables............................:    10000
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:     9998
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  2.5405160e+06 2.48e+01 2.73e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.3512419e+06 1.49e+01 8.27e+01  -1.0 2.20e+00    -  1.00e+00 1.00e+00f  1
   2  1.5156131e+05 4.28e+00 1.36e+02  -1.0 1.43e+00    -  1.00e+00 1.00e+00f  1
   3  6.6755024e+01 3.09e-01 2.18e+01  -1.0 5.63e-01    -  1.00e+00 1.00e+00f  1
   4  6.2338933e+00 1.73e-02 8.47e-01  -1.0 2.10e-01    -  1.00e+00 1.00e+00h  1
   5  6.2324586e+00 1.15e-05 8.16e-04  -1.7 3.35e-03    -  1.00e+00 1.00e+00h  1
   6  6.2324586e+00 8.36e-12 7.97e-10  -5.7 2.00e-06    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 6

                                   (scaled)                 (unscaled)
Objective...............:   7.8692659500479645e-01    6.2324586324379885e+00
Dual infeasibility......:   7.9743417331632266e-10    6.3156786526652763e-09
Constraint violation....:   8.3555384833289281e-12    8.3555384833289281e-12
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   7.9743417331632266e-10    6.3156786526652763e-09


Number of objective function evaluations             = 7
Number of objective gradient evaluations             = 7
Number of equality constraint evaluations            = 7
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 7
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 6
Total CPU secs in IPOPT (w/o function evaluations)   =      0.157
Total CPU secs in NLP function evaluations           =      0.028

EXIT: Optimal Solution Found.

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

