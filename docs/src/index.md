# Introduction

Welcome to the documentation of [ExaModels.jl](https://github.com/sshin23/ExaModels.jl)

!!! note
    This documentation is also available in PDF format: [ExaModels.pdf](ExaModels.pdf).

## What is ExaModels.jl?
ExaModels.jl is an [algebraic modeling](https://en.wikipedia.org/wiki/Algebraic_modeling_language) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) tool in [Julia Language](https://julialang.org/), specialized for [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming). ExaModels.jl employs what we call [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction for [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming) (NLPs), which allows for the preservation of the parallelizable structure within the model equations, facilitating efficient, parallel [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) on the [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) accelerators. More details about SIMD abstraction can be found [here](/simd).

## Differences from other tools
ExaModels.jl is different from other algebraic modeling tools, such as [JuMP](https://github.com/jump-dev/JuMP.jl) or [AMPL](https://ampl.com/), in the following ways:
- **Modeling Interface**: ExaModels.jl enforces users to specify the model equations always in the form of `Generator`. This allows ExaModels.jl to preserve the SIMD-compatible structure in the model equations.
- **Performance**: ExaModels.jl compiles (via Julia's compiler) derivative evaluation codes that are specific to each computation pattern, based on reverse-mode automatic differentiation. This makes the speed of derivative evaluation (even on the CPU) significantly faster than other existing tools.
- **Portability**: ExaModels.jl can evaluate derivatives on GPU accelerators. The code is currently only tested for NVIDIA GPUs, but GPU code is implemented mostly based on the portable programming paradigm, [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). In the future, we are interested in supporting Intel, AMD, and Apple GPUs.

# Supporting ExaModels.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/sshin/ExaModels.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/sshin23/ExaModels.jl/discussions).
