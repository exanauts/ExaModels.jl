# Highlights

## Key differences from other algebraic modeling tools
ExaModels.jl is different from other algebraic modeling tools, such as [JuMP](https://github.com/jump-dev/JuMP.jl) or [AMPL](https://ampl.com/), in the following ways:
- **Modeling Interface**: ExaModels.jl enforces users to specify the model equations always in the form of `Generator`. This allows ExaModels.jl to preserve the SIMD-compatible structure in the model equations.
- **Performance**: ExaModels.jl compiles (via Julia's compiler) derivative evaluation codes that are specific to each computation pattern, based on reverse-mode automatic differentiation. This makes the speed of derivative evaluation (even on the CPU) significantly faster than other existing tools.
- **Portability**: ExaModels.jl can evaluate derivatives on GPU accelerators. The code is currently only tested for NVIDIA GPUs, but GPU code is implemented mostly based on the portable programming paradigm, [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). In the future, we are interested in supporting Intel, AMD, and Apple GPUs.

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

## References

```@bibliography
```
