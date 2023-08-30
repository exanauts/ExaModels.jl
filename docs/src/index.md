# Overview

Welcome to the documentation of [ExaModels.jl](https://github.com/sshin23/ExaModels.jl)	
!!! note
    ExaModels runs on julia `VERSION ≥ v"1.9"`

!!! warning
	**Please help us improve ExaModels and this documentation!** ExaModels is in the early stage of development, and you may encounter unintended behaviors or missing documentations. If you find anything is not working as intended or documentation is missing, please [open issues](https://github.com/sshin/ExaModels.jl/issues) or [pull requests](https://github.com/sshin/ExaModels.jl/pulls) or start [discussions](https://github.com/sshin/ExaModels.jl/discussions). 

## What is ExaModels.jl?
ExaModels.jl is an [algebraic modeling](https://en.wikipedia.org/wiki/Algebraic_modeling_language) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) tool in [Julia Language](https://julialang.org/), specialized for [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming). ExaModels.jl employs what we call [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction for [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming) (NLPs), which allows for the preservation of the parallelizable structure within the model equations, facilitating efficient [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) either on the single-thread CPUs, multi-threaded CPUs, as well as [GPU accelerators](https://en.wikipedia.org/wiki/Graphics_processing_unit). More details about SIMD abstraction can be found [here](/simd).

## When should I use ExaModels.jl?
ExaModels.jl shines when your model has
- nonlinear objective and constraints;
- a large number of variables and constraints;
- highly repetitive structure;
- sparse Hessian and Jacobian.
These features are often exhibited in optimization problems associated with first-principle physics-based models. Primary examples include optimal control problems formulated with direct subscription method [biegler2010nonlinear](@cite) and network system optimization problems, such as optimal power flow [coffrin2018powermodels](@cite) and gas network control/estimation problems.

## New to Julia?
Welcome to Julia community! Below is the helpful link to start using Julia.
- [juliaup](https://docs.julialang.org/en/v1/manual/getting-started/): julia installer and version multiplexer.
- Julia tutorial in [Julia's official documentation](https://docs.julialang.org/en/v1/manual/getting-started/ep).

## Supported Solvers
ExaModels can be used with any solver that can handle `NLPModel` data type, but several callbacks are not currently implemented, and cause some errors. Currently, it is tested with the following solvers:
- [Ipopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) (via [NLPModelsIpopt.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl))
- [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl)

## Documentation Structure
This documentation is structured in the following way.
- The remainder of [this page](.) highlights several key aspects of ExaModels.jl.
- The mathematical abstraction---SIMD abstraction of nonlinear programming---of ExaModels.jl is discussed in [Mathematical Abstraction page](./simd).
- The step-by-step tutorial of using ExaModels.jl can be found in [Tutorial page](./guide).
- This documentation does not intend to discuss the engineering behind the implementation of ExaModels.jl. Some high-level idea is discussed in [a recent publication](https://arxiv.org/abs/2307.16830), but the full details of the engineering behind it will be discussed in the future publications.


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
