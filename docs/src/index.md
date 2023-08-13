# Introduction

Welcome to the documentation of [ExaModels.jl](https://github.com/sshin23/ExaModels.jl)

!!! note
    This documentation is also available in PDF format: [ExaModels.pdf](ExaModels.pdf).

!!! warning
	**Please help us improve ExaModels.jl and this documentation!** ExaModels.jl is in the early stage of development, and you may encounter unintended behaviors or missing documentations. If you find anything is not working as intended or documentation is missing, please [open issues](https://github.com/sshin/ExaModels.jl/issues) or [pull requests](https://github.com/sshin/ExaModels.jl/pulls) or start [discussions](https://github.com/sshin/ExaModels.jl/discussions). 

## What is ExaModels.jl?
ExaModels.jl is an [algebraic modeling](https://en.wikipedia.org/wiki/Algebraic_modeling_language) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) tool in [Julia Language](https://julialang.org/), specialized for [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming). ExaModels.jl employs what we call [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction for [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming) (NLPs), which allows for the preservation of the parallelizable structure within the model equations, facilitating efficient [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) either on the single-thread CPUs, multi-threaded CPUs, as well as [GPU accelerators](https://en.wikipedia.org/wiki/Graphics_processing_unit). More details about SIMD abstraction can be found [here](/simd).

## Documentation Structure
This documentation is structured in the following way.
- The remainder of [this page](.) highlights several key aspects of ExaModels.jl.
- The mathematical abstraction---SIMD abstraction of nonlinear programming---of ExaModels.jl is discussed in [Mathematical Abstraction page](./simd).
- The step-by-step tutorial of using ExaModels.jl can be found in [Tutorial page](./guide).
- This documentation does not intend to discuss the engineering behind the implementation of ExaModels.jl. Some high-level idea is discussed in [a recent publication](https://arxiv.org/abs/2307.16830), but the full details of the engineering behind it will be discussed in the future publications.

## Key differences from other algebraic modeling tools
ExaModels.jl is different from other algebraic modeling tools, such as [JuMP](https://github.com/jump-dev/JuMP.jl) or [AMPL](https://ampl.com/), in the following ways:
- **Modeling Interface**: ExaModels.jl enforces users to specify the model equations always in the form of `Generator`. This allows ExaModels.jl to preserve the SIMD-compatible structure in the model equations.
- **Performance**: ExaModels.jl compiles (via Julia's compiler) derivative evaluation codes that are specific to each computation pattern, based on reverse-mode automatic differentiation. This makes the speed of derivative evaluation (even on the CPU) significantly faster than other existing tools.
- **Portability**: ExaModels.jl can evaluate derivatives on GPU accelerators. The code is currently only tested for NVIDIA GPUs, but GPU code is implemented mostly based on the portable programming paradigm, [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). In the future, we are interested in supporting Intel, AMD, and Apple GPUs.

## When I should use ExaModels.jl?
ExaModels.jl shines when your model has
- nonlinear objective and constraints;
- a large number of variables and constraints;
- highly repetitive structure;
- sparse Hessian and Jacobian.
These features are often exhibited in optimization problems associated with first-principle physics-based models. Primary examples include optimal control problems formulated with direct subscription method [biegler2010nonlinear](@cite) and network system optimization problems, such as optimal power flow [coffrin2018powermodels](@cite) and gas network control/estimation problems.


## Performance
For the nonlinear optimization problems that are suitable for SIMD abstraction, ExaModels.jl greatly accelerate the performance of derivative evaluations. The following is a recent benchmark result. Remarkably, for the AC OPF problem for a 9241 bus system, derivative evalution using ExaModels.jl on GPUs can be up to 2 orders of magnitudes faster than JuMP or AMPL.
```
==============================================================
|                 Evaluation Wall Time (sec)                 |
==============================================================
|      |           |           ExaModels (single)            |
| case | nvar ncon |   obj     con    grad     jac    hess   |
==============================================================
|  LV1 | 100   98  | 6.3e-06 7.6e-06 6.3e-06 1.2e-05 9.3e-05 |
|  LV2 |   1k 998  | 2.8e-05 3.0e-05 2.4e-05 5.5e-05 6.7e-04 |
|  LV3 |  10k  10k | 2.8e-04 2.9e-04 2.3e-04 5.4e-04 2.5e-03 |
|  QR1 | 659  459  | 8.9e-06 1.0e-05 6.2e-06 1.9e-05 3.1e-05 |
|  QR2 |   7k   5k | 5.7e-05 5.1e-05 4.1e-05 1.0e-04 2.1e-04 |
|  QR3 |  65k  45k | 5.9e-04 5.4e-04 4.6e-04 1.2e-03 6.4e-03 |
|  DC1 | 402  396  | 1.4e-06 4.9e-06 2.6e-06 6.4e-06 3.5e-05 |
|  DC2 |   3k   3k | 2.0e-06 2.6e-05 6.2e-06 5.4e-05 9.7e-05 |
|  DC3 |  34k  33k | 1.1e-05 2.4e-04 6.3e-05 5.3e-04 2.0e-03 |
|  PF1 |   1k   2k | 2.0e-06 3.5e-05 2.3e-06 3.7e-05 3.4e-04 |
|  PF2 |  11k  17k | 5.3e-06 3.1e-04 9.7e-06 3.2e-04 3.6e-03 |
|  PF3 |  86k 131k | 1.9e-05 2.8e-03 1.2e-04 2.7e-03 2.0e-02 |
==============================================================
|      |           |           ExaModels (multli)            |
| case | nvar ncon |   obj     con    grad     jac    hess   |
==============================================================
|  LV1 | 100   98  | 1.3e-05 1.5e-05 1.7e-05 1.6e-05 1.0e-04 |
|  LV2 |   1k 998  | 4.3e-05 4.1e-05 4.8e-05 7.5e-05 2.4e-04 |
|  LV3 |  10k  10k | 2.5e-04 2.2e-04 3.6e-04 5.1e-04 2.7e-03 |
|  QR1 | 659  459  | 2.2e-05 4.8e-05 2.4e-05 5.8e-05 8.4e-05 |
|  QR2 |   7k   5k | 2.5e-04 9.7e-05 2.6e-04 1.4e-04 6.5e-04 |
|  QR3 |  65k  45k | 5.0e-04 1.2e-03 7.7e-04 2.0e-03 6.9e-03 |
|  DC1 | 402  396  | 1.0e-05 3.7e-05 1.3e-05 3.9e-05 1.2e-04 |
|  DC2 |   3k   3k | 9.0e-06 1.8e-04 2.0e-05 2.0e-04 2.7e-04 |
|  DC3 |  34k  33k | 2.3e-05 4.3e-04 8.6e-05 8.2e-04 2.6e-03 |
|  PF1 |   1k   2k | 7.1e-06 7.8e-05 9.7e-06 8.7e-05 4.0e-04 |
|  PF2 |  11k  17k | 8.7e-06 1.5e-03 1.8e-05 1.4e-03 7.6e-03 |
|  PF3 |  86k 131k | 1.3e-04 1.9e-03 2.2e-04 2.3e-03 9.2e-03 |
==============================================================
|      |           |           ExaModels (gpu)               |
| case | nvar ncon |   obj     con    grad     jac    hess   |
==============================================================
|  LV1 | 100   98  | 8.3e-05 4.8e-05 8.9e-05 5.3e-05 1.1e-04 |
|  LV2 |   1k 998  | 5.1e-05 2.6e-05 5.4e-05 3.7e-05 6.6e-05 |
|  LV3 |  10k  10k | 6.4e-05 2.8e-05 5.7e-05 3.5e-05 6.8e-05 |
|  QR1 | 659  459  | 1.1e-04 2.2e-04 1.0e-04 2.3e-04 3.2e-04 |
|  QR2 |   7k   5k | 9.1e-05 1.5e-04 8.3e-05 1.6e-04 2.3e-04 |
|  QR3 |  65k  45k | 1.0e-04 1.7e-04 9.0e-05 1.8e-04 2.6e-04 |
|  DC1 | 402  396  | 8.3e-05 2.0e-04 7.9e-05 2.0e-04 2.6e-04 |
|  DC2 |   3k   3k | 6.6e-05 1.6e-04 6.7e-05 1.6e-04 2.1e-04 |
|  DC3 |  34k  33k | 7.3e-05 1.6e-04 7.0e-05 1.7e-04 2.5e-04 |
|  PF1 |   1k   2k | 5.7e-05 3.4e-04 5.2e-05 2.8e-04 3.5e-04 |
|  PF2 |  11k  17k | 5.6e-05 3.2e-04 5.6e-05 3.1e-04 3.1e-04 |
|  PF3 |  86k 131k | 9.9e-05 3.6e-04 5.0e-05 2.9e-04 3.8e-04 |
==============================================================
|      |           |                  JuMP                   |
| case | nvar ncon |   obj     con    grad     jac    hess   |
==============================================================
|  LV1 | 100   98  | 5.5e-06 2.8e-05 8.6e-06 3.1e-05 3.5e-04 |
|  LV2 |   1k 998  | 4.6e-05 2.9e-04 1.1e-04 4.9e-04 2.9e-03 |
|  LV3 |  10k  10k | 9.1e-04 4.8e-03 2.1e-03 7.0e-03 2.5e-02 |
|  QR1 | 659  459  | 9.2e-06 3.1e-05 8.3e-06 5.1e-05 1.0e-04 |
|  QR2 |   7k   5k | 1.2e-04 3.7e-04 1.1e-04 7.0e-04 3.8e-03 |
|  QR3 |  65k  45k | 2.1e-03 8.2e-03 2.1e-03 1.6e-02 4.0e-02 |
|  DC1 | 402  396  | 2.5e-06 2.0e-05 4.1e-06 3.9e-05 3.5e-04 |
|  DC2 |   3k   3k | 2.7e-05 2.9e-04 6.0e-05 6.0e-04 1.2e-03 |
|  DC3 |  34k  33k | 4.8e-04 7.7e-03 1.9e-03 1.5e-02 4.2e-02 |
|  PF1 |   1k   2k | 3.5e-06 1.1e-04 4.0e-06 2.3e-04 1.9e-03 |
|  PF2 |  11k  17k | 3.8e-05 2.1e-03 5.0e-05 4.4e-03 1.7e-02 |
|  PF3 |  86k 131k | 6.9e-04 3.5e-02 1.1e-03 7.2e-02 1.1e-01 | 
==============================================================
|      |           |                  AMPL                   |
| case | nvar ncon |   obj     con    grad     jac    hess   |
==============================================================
|  LV1 | 100   98  | 1.2e-06 1.7e-06 9.1e-06 1.3e-05 2.1e-04 |
|  LV2 |   1k 998  | 1.5e-06 8.5e-06 2.6e-05 1.7e-04 1.6e-03 |
|  LV3 |  10k  10k | 8.9e-06 2.3e-04 3.1e-04 3.4e-03 2.2e-02 |
|  QR1 | 659  459  | 1.5e-06 3.4e-06 1.4e-06 2.7e-05 2.9e-04 |
|  QR2 |   7k   5k | 1.2e-05 4.5e-05 1.2e-05 3.5e-04 4.4e-03 |
|  QR3 |  65k  45k | 1.4e-04 1.1e-03 1.2e-04 7.9e-03 5.4e-02 |
|  DC1 | 402  396  | 5.3e-07 4.8e-06 1.2e-06 5.2e-06 3.5e-05 |
|  DC2 |   3k   3k | 8.6e-07 3.9e-05 5.5e-06 6.2e-05 2.6e-05 |
|  DC3 |  34k  33k | 4.8e-06 4.1e-04 2.5e-05 4.1e-04 3.3e-03 |
|  PF1 |   1k   2k | 9.5e-07 2.4e-04 2.8e-06 1.3e-04 1.6e-03 |
|  PF2 |  11k  17k | 1.1e-06 5.1e-03 7.7e-06 3.5e-03 2.6e-02 |
|  PF3 |  86k 131k | 3.5e-06 4.1e-02 5.4e-05 3.6e-02 2.3e-01 |
==============================================================
 * commit : 8a396718b7f7632d239e9edb18f6177fedf4e2a0
 * CPU    : Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz (nthreads = 20)
 * GPU    : Quadro GV100
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

## References

```@bibliography
```
