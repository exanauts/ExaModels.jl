# ExaModels.jl
*An implementation of SIMD abstraction for nonlinear programs and automatic differentiation.*

| **License** | **Documentation** | **Build Status** | **Coverage** | **Citation** |
|:-----------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://sshin23.github.io/ExaModels.jl/)  | [![build](https://github.com/sshin23/ExaModels.jl/actions/workflows/test.yml/badge.svg)](https://github.com/sshin23/ExaModels.jl/actions/workflows/test.yml) | [![codecov](https://codecov.io/gh/sshin23/ExaModels.jl/branch/main/graph/badge.svg?token=8ViJWBWnZt)](https://codecov.io/gh/sshin23/ExaModels.jl) |

## Introduction
ExaModels.jl employs what we call **SIMD abstraction for nonlinear programs** (NLPs), which allows for the **preservation of the parallelizable structure** within the model equations, facilitating **efficient, parallel derivative evaluations** on the **GPU**.

ExaModels.jl is different from other algebraic modeling tools, such as JuMP or AMPL, in the following ways:
- **Modeling Interface**: ExaModels.jl enforces users to specify the model equations always in the form of `Iterable`s. This allows ExaModels.jl to preserve the SIMD-compatible structure in the model equations.
- **Performance**: ExaModels.jl compiles (via Julia's compiler) derivative evaluation codes that are specific to each computation pattern, based on reverse-mode automatic differentiation. This makes the speed of derivative evaluation (even on the CPU) significantly faster than other existing tools.
- **Portability**: ExaModels.jl can evaluate derivatives on GPU accelerators. The code is currently only tested for NVIDIA GPUs, but GPU code is implemented mostly based on the portable programming paradigm, KernelAbstractions.jl. In the future, we are interested in supporting Intel, AMD, and Apple GPUs.

## Supporting ExaModels.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/sshin/ExaModels.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/sshin23/ExaModels.jl/discussions).
