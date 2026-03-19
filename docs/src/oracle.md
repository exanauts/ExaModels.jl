```@meta
EditURL = "oracle.jl"
```

# [Oracle Constraints](@id oracle)

ExaModels can combine its SIMD symbolic constraints with user-supplied
"oracle" constraint blocks via [`VectorNonlinearOracle`](@ref).  An oracle
encapsulates an opaque nonlinear function together with derivative
callbacks — either sparse coordinate matrices (`jac!`/`hess!`) or
matrix-free products (`jvp!`/`vjp!`/`hvp!`).

This is useful when part of the model comes from an external source that
ExaModels cannot trace symbolically: a neural network, a PDE solver,
a GPU simulation, etc.

## Low-level API: `VectorNonlinearOracle`

At the lowest level, you construct a `VectorNonlinearOracle` with callbacks
that operate on the **full** problem variable vector, then register it with
`constraint(core, oracle)`.  This gives full control over sparsity patterns
and variable indexing.

## High-level API: `embed_oracle`

For the common pattern of embedding an opaque function `f: ℝⁿ → ℝᵐ` into
a model, `embed_oracle` handles all the plumbing automatically:

1. Creates auxiliary output variables `z ∈ ℝᵐ`
2. Registers an oracle constraint `z − f(x) = 0`
3. Returns `z` as a regular `Variable` for use in any SIMD expression

The callbacks operate on **local** vectors — no offset management needed:

- `f!(y, x)`:          `y[1:m] = f(x[1:n])`
- `jvp!(Jv, x, v)`:    `Jv[1:m] = J_f(x) * v[1:n]`
- `vjp!(Jtv, x, w)`:   `Jtv[1:n] = J_f(x)' * w[1:m]`
- `hvp!(Hv, x, w, v)`:  `Hv[1:n] = (Σ wᵢ ∇²fᵢ(x)) * v[1:n]`  (optional)

When `gpu=true`, all callbacks receive device arrays (e.g. `CuArray`) and
must use broadcast operations — no scalar indexing.  The oracle's index
arrays and work buffers are automatically placed on the correct device.

## Example

We embed an opaque element-wise squaring function `f(x)ᵢ = xᵢ²` and mix
its output `z` with symbolic ExaModels expressions in the objective and
constraints.  Using broadcast in the callbacks makes them work on both
CPU and GPU without changes.

```math
\min_{x} \sum_i \bigl(z_i + \sin(x_i)\bigr) \quad
\text{s.t.} \quad z = f(x),\; x_i \ge 1 \;\forall i
```

````julia
using ExaModels
using CUDA
using MadNLP
using MadNLPGPU

N = 100

core = ExaCore(Float64; backend = CUDABackend())
x = variable(core, N; start = (1.0 for _ in 1:N))

z, _ = embed_oracle(
    core, x, N;
    f! = (y, xv) -> (y .= xv .^ 2; nothing),
    jvp! = (Jv, xv, v) -> (Jv .= 2 .* xv .* v; nothing),
    vjp! = (Jtv, xv, w) -> (Jtv .= 2 .* xv .* w; nothing),
    hvp! = (Hv, xv, w, v) -> (Hv .= 2 .* w .* v; nothing),
    gpu = true,
)
````

````
(Variable

  x ∈ R^{100}
, VectorNonlinearOracle

  ncon: 100   nnzj: 20000   nnzh: 20100   gpu: true
  jac!: false   jvp!: true   vjp!: true   hvp!: true
)
````

The oracle output `z` can now be used in SIMD generator expressions,
freely mixed with symbolic operations on `x`:

````julia
objective(core, z[i] + sin(x[i]) for i in 1:N)
constraint(core, x[i] for i in 1:N; lcon = 1.0, ucon = Inf)

model = ExaModel(core)
result = madnlp(model; print_level = MadNLP.INFO, linear_solver = LapackCUDASolver)

println("\nStatus:    ", result.status)
println("Objective: ", result.objective)
println("x[1:5]:    ", round.(result.solution[1:5]; digits = 6))
println("z[1:5]:    ", round.(result.solution[(N + 1):(N + 5)]; digits = 6))
````

````
This is [34mMad[31mN[32mL[35mP[0m version v0.9.1, running with cuSOLVER v12.1.0 -- (BUNCHKAUFMAN)

Number of nonzeros in constraint Jacobian............:    20100
Number of nonzeros in Lagrangian Hessian.............:    20200

Total number of variables............................:      200
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:      100
Total number of inequality constraints...............:      100
        inequality constraints with only lower bounds:      100
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du inf_compl lg(mu) lg(rg) alpha_pr ir ls
   0  8.4147098e+01 1.00e+00 1.00e+00 1.00e-02  -1.0     -   0.00e+00  0  0 
   1  8.6209054e+01 9.80e-01 2.49e+00 4.00e-02  -1.7     -   2.00e-02  1  1h
   2  1.8609911e+02 5.78e-05 1.49e-02 2.01e-02  -1.7     -   1.00e+00  1  1h
   3  1.8410049e+02 6.17e-05 1.65e-05 2.55e-04  -3.8     -   1.00e+00  1  1f
   4  1.8408435e+02 1.42e-09 1.88e-05 7.15e-05  -5.0     -   1.00e+00  1  1h

Number of Iterations....: 4

                                   (scaled)                 (unscaled)
Objective...............:   1.8408435170850049e+02    1.8408435170850049e+02
Dual infeasibility......:   1.8821851682560009e-05    1.8821851682560009e-05
Constraint violation....:   1.4161841727523133e-09    1.4161841727523133e-09
Complementarity.........:   7.1480903731064105e-05    7.1480903731064105e-05
Overall NLP error.......:   7.1480903731064105e-05    7.1480903731064105e-05

Number of objective function evaluations              = 5
Number of objective gradient evaluations              = 5
Number of constraint evaluations                      = 5
Number of constraint Jacobian evaluations             = 5
Number of Lagrangian Hessian evaluations              = 4
Number of KKT factorizations                          = 4
Number of KKT backsolves                              = 8

Total wall secs in initialization                     =  3.772 s
Total wall secs in linear solver                      =  1.487 s
Total wall secs in NLP function evaluations           =  2.934 s
Total wall secs in solver (w/o init./fun./lin. alg.)  =  8.645 s
Total wall secs                                       = 16.839 s

[32mEXIT: Optimal Solution Found (tol = 1.0e-04).[0m

Status:    SOLVE_SUCCEEDED
Objective: 184.0843517085005
x[1:5]:    [0.999804, 0.999804, 0.999804, 0.999804, 0.999804]
z[1:5]:    [0.999479, 0.999479, 0.999479, 0.999479, 0.999479]

````

The optimal solution is xᵢ = 1, zᵢ = 1 for all i, with objective
N × (1 + sin(1)) ≈ 184.15.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

