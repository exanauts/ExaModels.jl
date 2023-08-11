# Introduction

Welcome to the documentation of [ExaModels.jl](https://github.com/sshin23/ExaModels.jl)

!!! note
    This documentation is also available in PDF format: [ExaModels.pdf](ExaModels.pdf).

## What is ExaModels.jl?
ExaModels.jl is an [algebraic modeling](https://en.wikipedia.org/wiki/Algebraic_modeling_language) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) tool in [Julia Language](https://julialang.org/), specialized for [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming). ExaModels.jl employs what we call [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction for [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming) (NLPs), which allows for the preservation of the parallelizable structure within the model equations, facilitating efficient, parallel [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) on the [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) accelerators. More details about SIMD abstraction can be found [here](/simd).

## Key differences from other tools
ExaModels.jl is different from other algebraic modeling tools, such as [JuMP](https://github.com/jump-dev/JuMP.jl) or [AMPL](https://ampl.com/), in the following ways:
- **Modeling Interface**: ExaModels.jl enforces users to specify the model equations always in the form of `Generator`. This allows ExaModels.jl to preserve the SIMD-compatible structure in the model equations.
- **Performance**: ExaModels.jl compiles (via Julia's compiler) derivative evaluation codes that are specific to each computation pattern, based on reverse-mode automatic differentiation. This makes the speed of derivative evaluation (even on the CPU) significantly faster than other existing tools.
- **Portability**: ExaModels.jl can evaluate derivatives on GPU accelerators. The code is currently only tested for NVIDIA GPUs, but GPU code is implemented mostly based on the portable programming paradigm, [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). In the future, we are interested in supporting Intel, AMD, and Apple GPUs.

## Highlights
For the nonlinear optimization problems that are suitable for SIMD abstraction, ExaModels.jl greatly accelerate the performance of derivative evaluations. The following is a recent benchmark result. Remarkably, for the AC OPF problem for a 9241 bus system, derivative evalution using ExaModels.jl on GPUs can be up to 2 orders of magnitudes faster than JuMP or AMPL.
```
===============================================================================================================================
 Case |          ExaModels (single)           |          ExaModels (multi)            |           ExaModels (gpu)             |
      |  obj     con    grad     jac    hess  |  obj     con    grad     jac    hess  |  obj     con    grad     jac    hess  |
===============================================================================================================================
  LV1 |2.8e-06 2.5e-06 2.3e-06 4.2e-06 3.9e-05|5.8e-06 4.4e-06 6.2e-06 6.9e-06 4.4e-05|3.7e-05 2.2e-05 4.1e-05 2.6e-05 5.2e-05|
  LV2 |1.7e-05 1.6e-05 1.4e-05 3.1e-05 1.7e-04|2.0e-05 1.8e-05 2.2e-05 3.3e-05 4.2e-04|4.7e-05 3.1e-05 4.2e-05 5.1e-05 7.9e-05|
  LV3 |1.6e-04 1.6e-04 1.4e-04 3.1e-04 6.3e-04|1.1e-04 7.4e-05 7.9e-05 1.1e-04 3.8e-04|4.7e-05 4.7e-05 4.3e-05 5.0e-05 8.4e-05|
  QR1 |1.1e-05 9.4e-06 7.9e-06 1.2e-05 4.5e-05|1.5e-05 2.6e-05 1.5e-05 3.1e-05 7.1e-05|6.2e-05 1.0e-04 6.2e-05 1.2e-04 1.8e-04|
  QR2 |3.9e-05 3.4e-05 2.6e-05 5.7e-05 4.2e-04|5.4e-05 4.9e-05 6.3e-05 7.6e-05 1.6e-04|7.1e-05 1.3e-04 6.2e-05 2.9e-04 5.4e-04|
  QR3 |3.5e-04 2.9e-04 2.3e-04 5.5e-04 4.3e-03|2.9e-04 2.3e-04 2.6e-04 4.0e-04 5.8e-04|8.0e-05 3.8e-04 8.4e-05 3.9e-04 5.3e-04|
  DC1 |4.0e-06 8.0e-06 4.2e-06 6.7e-06 3.2e-05|8.6e-06 2.6e-05 9.7e-06 2.6e-05 7.5e-05|4.9e-05 9.8e-05 5.0e-05 1.1e-04 1.4e-04|
  DC2 |3.2e-06 1.5e-05 3.3e-06 2.2e-05 3.6e-05|8.2e-06 5.6e-05 9.6e-06 4.9e-05 6.6e-05|4.8e-05 1.0e-04 4.8e-05 1.1e-04 1.5e-04|
  DC3 |4.4e-05 1.4e-04 2.3e-05 2.2e-04 2.8e-03|3.9e-05 1.2e-04 2.4e-05 2.0e-04 1.2e-03|6.5e-05 1.2e-04 5.7e-05 1.2e-04 1.7e-04|
  PF1 |1.3e-05 2.7e-05 1.1e-05 2.8e-05 9.0e-05|1.2e-05 5.5e-05 9.9e-06 5.5e-05 3.7e-04|4.6e-05 1.9e-04 4.2e-05 1.8e-04 2.2e-04|
  PF2 |1.5e-05 1.9e-04 1.5e-05 2.1e-04 2.6e-03|1.2e-05 2.2e-04 1.5e-05 3.7e-04 2.8e-04|5.3e-05 2.5e-04 4.3e-05 2.3e-04 5.4e-04|
  PF3 |9.3e-06 1.4e-03 6.9e-05 1.7e-03 1.1e-02|5.6e-05 5.9e-04 4.8e-05 6.7e-04 5.1e-03|6.2e-05 2.7e-04 4.5e-05 5.5e-04 6.8e-04|
===============================================================================================================================
 Case |                 JuMP                  |                 AMPL                  |
      |  obj     con    grad     jac    hess  |  obj     con    grad     jac    hess  |
=======================================================================================
  LV1 |6.9e-07 1.3e-06 2.2e-06 6.4e-06 8.3e-05|2.1e-06 7.5e-06 3.9e-06 1.1e-05 1.6e-04|
  LV2 |8.6e-07 4.3e-06 1.6e-05 6.1e-05 8.0e-04|2.5e-05 9.5e-05 5.2e-05 1.6e-04 7.5e-04|
  LV3 |5.8e-06 1.3e-04 1.8e-04 1.8e-03 1.2e-02|5.3e-04 2.2e-03 1.4e-03 3.5e-03 2.1e-02|
  QR1 |1.2e-06 2.5e-06 1.0e-06 1.6e-05 1.7e-04|6.7e-06 2.5e-05 6.0e-06 3.6e-05 1.9e-04|
  QR2 |7.2e-06 2.7e-05 6.6e-06 2.1e-04 2.4e-03|5.6e-05 1.9e-04 6.1e-05 3.3e-04 1.9e-03|
  QR3 |7.0e-05 5.9e-04 7.2e-05 4.6e-03 2.9e-02|1.0e-03 4.5e-03 1.0e-03 7.9e-03 2.8e-02|
  DC1 |6.7e-07 4.8e-06 4.8e-07 3.6e-06 2.3e-05|5.4e-06 1.7e-05 2.5e-06 2.9e-05 2.2e-04|
  DC2 |7.0e-07 2.5e-05 9.3e-07 2.5e-05 1.3e-05|1.2e-05 1.4e-04 3.3e-05 2.8e-04 2.3e-03|
  DC3 |2.8e-06 2.5e-04 6.2e-06 2.5e-04 2.1e-03|3.3e-04 3.9e-03 1.3e-03 7.9e-03 2.8e-02|
  PF1 |4.7e-07 1.2e-04 4.1e-07 6.7e-05 7.9e-04|2.7e-06 6.0e-05 2.6e-06 1.8e-04 1.3e-03|
  PF2 |8.7e-07 3.1e-03 2.1e-06 1.8e-03 1.4e-02|2.1e-05 1.2e-03 2.9e-05 2.5e-03 1.1e-02|
  PF3 |2.2e-06 2.5e-02 1.3e-05 1.7e-02 1.1e-01|3.8e-04 1.5e-02 6.0e-04 2.8e-02 9.6e-02|
=======================================================================================
  * commit : 7f83708eed982ad7d9b4d098707024dbe7e514e2
  * CPU    : 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz (nthreads = 4)
  * GPU    : NVIDIA GeForce RTX 3060 Laptop GPU
```

## Citing ExaModels.jl
If you use ExaModels.jl in your research, we would greatly appreciate your citing this [preprint](https://arxiv.org/abs/2307.16830).
```bibtex
@misc{shin2023accelerating,
      title={Accelerating Optimal Power Flow with {GPU}s: {SIMD} Abstraction of Nonlinear Programs and Condensed-Space Interior-Point Methods}, 
      author={Sungho Shin and Fran{\c{c}}ois Pacaud and Mihai Anitescu},
      year={2023},
      eprint={2307.16830},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

## Supporting ExaModels.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/sshin/ExaModels.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/sshin23/ExaModels.jl/discussions).
