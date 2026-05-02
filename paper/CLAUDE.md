# ExaModels.jl Paper

## Target Venue
Mathematical Programming Computation (MPC)

## Title
"ExaModels.jl: an Algebraic Modeling System for Nonlinear Programming on GPUs"

## Paper Outline
1. **Introduction** (DRAFTED)
   - Motivation: repetitive pattern structure in large-scale NLPs
   - Motivating example: Goddard rocket (COPS)
   - Related works: AMPL, JuMP, CasADi, Pyomo, Gravity
   - Our approach: SIMD abstraction, parameterized expression tree, coloring-free AD
   - Julia/AOT compilation angle
   - Contributions list, notation, outline

2. **Modeling Abstraction** (DRAFTED)
   - 2.1 NLP and AMS (callbacks aligned with NLPModels.jl: obj, grad!, cons!, jac_coord!, hess_coord!)
   - 2.2 SIMD Abstraction of NLPs (math formulation, SIMDFunction, COPS/OPF examples)

3. **Core Data Structures** (DRAFTED)
   - 3.1 ExaCore: incremental model construction (fields: var, par, obj, cons, counters, arrays)
   - 3.2 ExaModel: compiled NLP model (NLPModelMeta, AbstractNLPModel, solver interface)
   - 3.3 User-Facing Syntax (@add_var, @add_par, @add_obj, @add_con, @add_con!, @add_expr; end-to-end example)

4. **Parameterized Expression Tree** (DRAFTED)
   - 4.1 Node Types (Var, DataIndexed, ParameterNode, Constant{T}, Node1, Node2; VarSource/DataSource sentinels)
   - 4.2 Expression Type and Structure (nested type parameterization)
   - 4.3 Tree Construction via Multiple Dispatch (step-by-step dispatch, getindex/getproperty/iterate for tuples/named tuples)
   - 4.4 Compile-Time Constants and Algebraic Simplification
   - 4.5 Function Registration (@register_univariate, @register_bivariate)

5. **Sparse Reverse-Mode AD Without Coloring**
   - 5.1 First-Order Derivatives (adjoint tree, drpass, grpass)
   - 5.2 Second-Order Derivatives (second-adjoint tree, hrpass, hdrpass, hrpass0)
   - 5.3 Sparsity Detection Without Coloring (probe evaluation, Compressor, SIMDFunction)

6. **GPU Acceleration and Implementation** (DRAFTED)
   - 6.1 SIMDFunction: the compiled pattern (struct, Rosenbrock example)
   - 6.2 CPU vs GPU Evaluation (CPU @simd loop, GPU kernels via KA.jl, code snippets)
   - 6.3 Implementation Details (COO not CSC, obj buffer+reduce, grad sparse+compress, data-level parallelism only)

7. **Type Stability and Ahead-of-Time Compilation** (DRAFTED)
   - 7.1 Immutable ExaCore and Reconstruction (type stability by design, contrast with JuMP mutable Model)
   - 7.2 Ahead-of-Time Compilation (juliac --trim=safe, NLPModelsIpoptLite.jl, standalone binary example)
   - 7.3 Generic Numeric Precision (Float32/64, DoubleFloats.jl, replace_T, Constant{T})

8. **Solver Interface** (DRAFTED)
   - NLPModels.jl callback API, contrast with MOI
   - Solver ecosystem: Ipopt, KNITRO, MadNLP, Uno, Percival, DCISolver, FletcherPenaltySolver, etc.

9. **Extensions** (DRAFTED)
   - 9.1 Parameterized Problems (eq:param, set_value! for efficient resolves)
   - 9.2 Two-Stage Stochastic Programs (eq:twostage, TwoStageExaCore, EachScenario())
   - 9.3 Batch Models (eq:batch, BatchExaCore, matrix storage, FlatNLPModel)

10. **Numerical Experiments**
    - 10.1 Benchmark Problems (LuksanVlcek, AC OPF/PGLIB, COPS)
    - 10.2 AD Performance Comparison (ExaModels CPU vs JuMP vs AMPL — AD times only)
    - 10.3 GPU Acceleration (CPU vs GPU scaling)

11. **Conclusions** (DRAFTED)
    - Summary of 3 key contributions, living document, limitations/future work

## Benchmark Repos
- `~/git/LuksanVlcekBenchmark.jl` — 18 scalable sparse problems, ExaModels vs JuMP
- `~/git/COPSBenchmark.jl` — 23 COPS problems, ExaModels vs JuMP
- `~/git/ExaModelsPower.jl` — AC/DC OPF, multi-period, SCOPF; PGLIB instances

## Bibliography
Using `~/MIT Dropbox/Sungho Shin/main.bib` (copied to paper/main.bib).

## Key Cite Keys
- ExaModels prior paper: `shinAcceleratingOptimalPower2024`
- JuMP: `lubinJuMP10Recent2023`
- AMPL: `fourerModelingLanguageMathematical1990`
- CasADi: `anderssonCasADiSoftwareFramework2019`
- Pyomo: `hartPyomoModelingSolving2011`
- Gravity: `hijaziGravityMathematicalModeling2018`
- Julia: `bezansonJuliaFreshApproach2017`
- KernelAbstractions: `churavyKernelAbstractionsjl2025`
- COPS: `dolanBenchmarkingOptimizationSoftware2001`
- LuksanVlcek: `lukVsanSparsePartiallySeparable`
- Plasmo.jl: `jalvingGraphbasedModelingAbstraction2022`
- MadNLP: `shinMadNLPjl2025`
- Condensed-space: `pacaudCondensedspaceMethodsNonlinear2024`
- ExaModelsPower: `shinExaModelsPowerjl2024`

## Writing Style Notes (from reading Shin's prior papers)
- Problem-first framing: open with concrete computational challenges
- Direct, no-hedge language ("we address" not "we propose to explore")
- Concrete quantitative claims early
- Honest about tradeoffs
- Implementation-grounded: connect theory to software
- Short, structured paragraphs — one idea per paragraph
- Name specific tools rather than vague "existing methods"
