# ExaModels.jl Paper

## Target Venue
Mathematical Programming Computation (MPC)

## Title
"ExaModels.jl: an Algebraic Modeling System for Nonlinear Programming on GPUs"

## Build Instructions
- **Compile**: `pdflatex -interaction=nonstopmode main.tex` (from `paper/` directory)
- **Bibliography**: `biber main` (NOT bibtex — we use biblatex with biber backend)
- **Full rebuild**: `pdflatex main && biber main && pdflatex main && pdflatex main`
- **Upload**: `scp main.pdf 000:~/public_html/main.pdf`
- Always compile then upload after every edit.

## LaTeX Conventions
- **Bibliography**: `biblatex` with `authoryear` style and `biber` backend. Use `\citep{}` for parenthetical citations, `\citet{}` for textual (author as subject). All citations currently use `\citep`.
- **Cross-references**: Always use `\cref{}` (lowercase, produces "(1)" for equations) and `\Cref{}` (capitalized, produces "Equation (1)", "Section 2", "Table 1", "Figure 1"). NEVER use `\ref{}`, `\eqref{}`, `Section~\ref{}`, `Table~\ref{}`, or `Figure~\ref{}`.
- **Floats**: Always use `[t]` placement for figures and tables (top of page). Never `[h]`, `[!htbp]`, etc.
- **Tables**: Use plain `\begin{tabular}{...}`, NOT `\tabular*{\textwidth}`.

## Paper Structure (current state)

### Section 1: Introduction
- **Motivation** (no subsection heading): Pattern structure in large-scale NLPs, Goddard rocket as motivating example with iterator-representability walkthrough, generalization to other problem classes
- **1.1 Related work**
  - 1.1.1 Algebraic modeling systems (AMPL, JuMP, CasADi, Pyomo, CVXPY, Gravity + comparison table)
  - 1.1.2 Monolithic vs. structured representations
- **1.2 Our approach**: ExaModels design philosophy, parameterized expression tree, coloring-free AD, KA.jl GPU portability, extensions
- **Contributions** (M1, M2, I1-I4, E1):
  - M = methodological (SIMD abstraction, coloring-free AD)
  - I = implementational (GPU portability, AOT, arbitrary precision, extensions)
  - E = experimental (benchmark implementations + comparison)
- Outline, Notation

### Section 2: Modeling Abstraction (DRAFTED)
- 2.1 NLP and AMS (callbacks: obj, grad!, cons!, jac_coord!, hess_coord!)
- 2.2 SIMD Abstraction (math formulation, SIMDFunction, COPS/OPF examples)
- 2.3 Core Data Structures (ExaCore, ExaModel, SIMDFunction, Objective, Constraint, ConstraintAugmentation)
- 2.4 User-Facing Syntax (@add_var, @add_par, @add_obj, @add_con, @add_con!, @add_expr; Goddard rocket code example)

### Section 3: Parameterized Expression Tree (DRAFTED)
- 3.1 Node Types (Var, DataIndexed, ParameterNode, Constant{T}, Node1, Node2)
- 3.2 Expression Type and Structure (nested type parameterization)
- 3.3 Tree Construction via Multiple Dispatch
- 3.4 Compile-Time Constants and Algebraic Simplification
- 3.5 Function Registration (@register_univariate, @register_bivariate)

### Section 4: Sparse Reverse-Mode AD Without Coloring
- 4.1 First-Order Derivatives (adjoint tree, drpass, grpass)
- 4.2 Second-Order Derivatives (second-adjoint tree, hrpass, hdrpass, hrpass0)
- 4.3 Sparsity Detection Without Coloring (probe evaluation, Compressor)

### Section 5: GPU Acceleration and Implementation (DRAFTED)
- 5.1 SIMDFunction: the compiled pattern
- 5.2 CPU vs GPU Evaluation (CPU @simd loop, GPU kernels via KA.jl)
- 5.3 Implementation Details (COO not CSC, obj buffer+reduce, grad sparse+compress)

### Section 6: Type Stability and AOT Compilation (DRAFTED)
- 6.1 Immutable ExaCore and Reconstruction (contrast with JuMP mutable Model)
- 6.2 Ahead-of-Time Compilation (juliac --trim=safe, NLPModelsIpoptLite.jl)
- 6.3 Generic Numeric Precision (Float32/64, DoubleFloats.jl, replace_T)

### Section 7: Solver Interface (DRAFTED)
- NLPModels.jl callback API, contrast with MOI
- Solver ecosystem: Ipopt, KNITRO, MadNLP, Uno, Percival, DCISolver, FletcherPenaltySolver

### Section 8: Extensions (DRAFTED)
- 8.1 Parameterized Problems (set_value!)
- 8.2 Two-Stage Stochastic Programs (TwoStageExaCore, EachScenario())
- 8.3 Batch Models (BatchExaCore, matrix storage, FlatNLPModel)

### Section 9: Numerical Experiments
- 9.1 Benchmark Problems (LuksanVlcek, COPS, PGLIB-OPF)
- 9.2 AD Performance Comparison (ExaModels CPU vs JuMP vs AMPL)
- 9.3 GPU Acceleration (CPU vs GPU scaling) — **NOT YET WRITTEN**

### Section 10: Conclusions (DRAFTED)
- Summary of contributions, limitations/future work

## Abstract (finalized)
The abstract establishes: (1) SIMD abstraction as main contribution, (2) coloring-free AD as secondary, (3) GPU compatibility as consequence of compile-time structure recognition. Implementation features (precision, GPU portability, AOT, extensions) are mentioned as "additionally". Benchmarks against JuMP and AMPL on CPU and GPU backends.

## Key Terminology
- Use "algebraic pattern" (not "computational pattern") when we can afford a few words
- Use "nonlinear optimization solution procedure" (not "solvers" which may be confused with just the linear algebra)
- "pattern--data separation" is the core concept
- Use "SIMD abstraction" for the high-level idea

## Benchmark Infrastructure
- **Benchmark repos**:
  - `~/git/LuksanVlcekBenchmark.jl` — 18 scalable sparse problems
  - `~/git/COPSBenchmark.jl` — 23 COPS problems
  - `~/git/ExaModelsPower.jl` — AC/DC OPF, PGLIB instances
- **Results**: `paper/benchmark/results/combined.csv` — ExaModels CPU data (108 LV+COPS rows + 54 OPF rows). JuMP/AMPL reference data NOT YET COLLECTED.
- **Table generator**: `paper/benchmark/table.jl` — reads combined.csv, produces .tex files in results/tables/
- **SGM summary table**: Currently shows "---" for JuMP/speedup columns (no reference data yet)
- **Appendix tables**: Per-suite tables with 3 sizes per problem (20, 2000, 200000 for LV; variable for COPS/OPF)

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
- Goddard rocket: `goddardMethodReachingExtreme1919`
- MadNLP: `shinMadNLPjl2025`
- Condensed-space: `pacaudCondensedspaceMethodsNonlinear2024`
- ExaModelsPower: `shinExaModelsPowerjl2024`
- CUTEst: `gouldCUTEstConstrainedUnconstrained2015`
- PGLIB-OPF: `babaeinejadsarookolaeePowerGridLibrary2021`

## Writing Style
- Problem-first framing: open with concrete computational challenges, explain lazily after
- Direct, no-hedge language ("we address" not "we propose to explore")
- Concrete quantitative claims early
- Honest about tradeoffs
- Implementation-grounded: connect theory to software
- Short, structured paragraphs — one idea per paragraph
- Name specific tools rather than vague "existing methods"
- Hit readers with the problem/formulation first, then explain what it does and define symbols lazily afterward
- Avoid "decompose" — people will think of decomposition algorithms
- Avoid overemphasizing solver compatibility; focus on the modeling/AD layer
