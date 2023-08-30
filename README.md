![Logo](full-logo.svg) 

*An [algebraic modeling](https://en.wikipedia.org/wiki/Algebraic_modeling_language) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) tool in [Julia Language](https://julialang.org/), specialized for [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming).*

---

| **License** | **Documentation** | **Build Status** | **Coverage** | **Citation** |
|:-----------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sshin23/ExaModels.jl/blob/main/LICENSE) | [![doc](https://img.shields.io/badge/docs-stable-blue.svg)](https://sshin23.github.io/ExaModels.jl/stable) [![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://sshin23.github.io/ExaModels.jl/dev)  | [![build](https://github.com/sshin23/ExaModels.jl/actions/workflows/test.yml/badge.svg)](https://github.com/sshin23/ExaModels.jl/actions/workflows/test.yml) | [![codecov](https://codecov.io/gh/sshin23/ExaModels.jl/branch/main/graph/badge.svg?token=8ViJWBWnZt)](https://codecov.io/gh/sshin23/ExaModels.jl) | [![arXiv](https://img.shields.io/badge/arXiv-2307.16830-b31b1b.svg)](https://arxiv.org/abs/2307.16830) |

## Overview
ExaModels.jl employs what we call **[SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction for [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming)** (NLPs), which allows for the **preservation of the parallelizable structure** within the model equations, facilitating **efficient, parallel [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)** on the **[GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) accelerators**.

ExaModels.jl is different from other algebraic modeling tools, such as [JuMP](https://github.com/jump-dev/JuMP.jl) or [AMPL](https://ampl.com/), in the following ways:
- **Modeling Interface**: ExaModels.jl requires users to specify the model equations always in the form of `Generator`s. This restrictive structure allows ExaModels.jl to preserve the SIMD-compatible structure in the model equations. This unique feature distinguishes ExaModels.jl from other algebraic modeling tools.
- **Performance**: ExaModels.jl compiles (via Julia's compiler) derivative evaluation codes tailored to each computation pattern. Through reverse-mode automatic differentiation using these tailored codes, ExaModels.jl achieves significantly faster derivative evaluation speeds, even when using CPU.
- **Portability**: ExaModels.jl goes beyond traditional boundaries of
algebraic modeling systems by **enabling derivative evaluation on GPU
accelerators**. Implementation of GPU kernels is accomplished using
the portable programming paradigm offered by
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
With ExaModels.jl, you can run your code on various devices, including
multi-threaded CPUs, NVIDIA GPUs, AMD GPUs, and Intel GPUs. Note that
Apple's Metal is currently not supported due to its lack of support
for double-precision arithmetic.


## Highlight
The performance comparison of ExaModels with other algebraic modeling systems for evaluating different NLP functions (obj, con, grad, jac, and hess) are shown below. Note that Hessian computations are the typical bottlenecks.
![benchmark](https://raw.githubusercontent.com/sshin23/ExaModels.jl/main/docs/src/assets/benchmark.svg)
## Supporting ExaModels.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/sshin/ExaModels.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/sshin23/ExaModels.jl/discussions).
