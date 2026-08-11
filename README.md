![Logo](full-logo.svg)

*An [algebraic modeling](https://en.wikipedia.org/wiki/Algebraic_modeling_language) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) tool in [Julia Language](https://julialang.org/), specialized for [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming).*

---

| **License** | **Documentation** | **Build Status** | **Coverage** | **Citation** |
|:-----------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/exanauts/ExaModels.jl/blob/main/LICENSE) | [![doc](https://img.shields.io/badge/docs-stable-blue.svg)](https://exanauts.github.io/ExaModels.jl/stable) [![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://exanauts.github.io/ExaModels.jl/dev)  | [![build](https://github.com/exanauts/ExaModels.jl/actions/workflows/test.yml/badge.svg)](https://github.com/exanauts/ExaModels.jl/actions/workflows/test.yml) | [![codecov](https://codecov.io/gh/exanauts/ExaModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/exanauts/ExaModels.jl) | [![arXiv](https://img.shields.io/badge/arXiv-2307.16830-b31b1b.svg)](https://arxiv.org/abs/2307.16830) |

## Overview

Large-scale [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming) (NLPs) arising in engineering, statistics, and scientific applications almost always have **partially separable, repetitive structure**: the objective and constraints are sums or stacks of scalar terms, and the same algebraic pattern repeats across many data points. An NLP solver re-evaluates the model functions (the objective, the constraints, the gradient, the constraint Jacobian, and the Lagrangian Hessian) at every iteration, so the cost of these evaluations can take up a substantial part of the total solution time. Exploiting the repetitive structure is the key to making them fast.

ExaModels.jl is built around exactly that. It employs a **[SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of nonlinear programs**: a model is expressed as a small number of algebraic patterns, each paired with an iterator over the data points at which the pattern is evaluated. The velocity dynamics of the Goddard rocket problem, for instance, is a single pattern applied at every time step:

```julia
@add_con(core, vel,
    -v[i] + v[i-1] + 0.5 * dt[1] * (
        (tau[i] - D_c * v[i]^2 * exp(-h_c * (h[i] - h_0) / h_0)
            - m[i] * g_0 * (h_0 / h[i])^2) / m[i]
        + (tau[i-1] - D_c * v[i-1]^2 * exp(-h_c * (h[i-1] - h_0) / h_0)
            - m[i-1] * g_0 * (h_0 / h[i-1])^2) / m[i-1])
    for i = 1:nh)
```

Because the pattern and its data travel separately, the repetitive structure is visible to the system without any structure detection, and the whole pipeline exploits it:

- **Pattern-specialized derivative kernels.** Each algebraic pattern is encoded as a parameterized expression tree in Julia's type system, so the Julia compiler generates a dedicated evaluation and derivative kernel per pattern. Refining a discretization grows the number of data points, never the number of kernels. Even on a single CPU thread, these specialized kernels make derivative evaluation (the sparse Hessian in particular, which is the typical bottleneck) substantially faster than in general-purpose modeling and AD frameworks in our cross-system benchmarks.
- **Coloring-free sparse automatic differentiation.** Sparsity is analyzed once, on the pattern itself rather than on the assembled problem, and first- and second-order derivatives are assembled directly into partially compressed sparse COO storage, with no graph coloring and no runtime sparsity detection.
- **Native GPU execution.** Every NLP function evaluation reduces to embarrassingly parallel loops over data points, implemented portably with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). The same model runs on NVIDIA (CUDA), AMD (ROCm), Intel (oneAPI), and OpenCL devices, on Apple silicon via Metal (in Float32, as Metal provides no double precision), and on multi-threaded CPUs.
- **Near-constant-time evaluation on GPUs.** Once the device supplies enough parallel threads, evaluation cost is set by the number of patterns rather than the number of data points. In our benchmarks, GPU execution evaluates sparse Hessians 76 times faster than single-threaded CPU evaluation on the largest Lukšan–Vlček instances, 30 times on COPS, and 7.3 times on PGLIB-OPF.

This is a deliberate trade relative to general algebraic modeling tools such as [JuMP](https://github.com/jump-dev/JuMP.jl) or [AMPL](https://ampl.com/): ExaModels.jl asks for the model equations in the structured iterator form above, and in exchange preserves the parallelizable structure end to end. Paired with a GPU-capable solver such as [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl), the entire solution pipeline (model evaluation, derivatives, and the optimization itself) runs on the GPU.

## Citation

If you use ExaModels.jl in your research, please cite:

```bibtex
@article{shin2024accelerating,
  title   = {Accelerating optimal power flow with {GPUs}: {SIMD} abstraction of nonlinear programs and condensed-space interior-point methods},
  author  = {Shin, Sungho and Anitescu, Mihai and Pacaud, Fran\c{c}ois},
  journal = {Electric Power Systems Research},
  volume  = {236},
  pages   = {110651},
  year    = {2024},
  doi     = {10.1016/j.epsr.2024.110651}
}
```

## Supporting ExaModels.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/exanauts/ExaModels.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/exanauts/ExaModels.jl/discussions).
