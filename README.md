# ExaModels.jl
*An [algebraic modeling](https://en.wikipedia.org/wiki/Algebraic_modeling_language) and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) tool in [Julia Language](https://julialang.org/), specialized for [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction of [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming).*

| **License** | **Documentation** | **Build Status** | **Coverage** | **Citation** |
|:-----------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://sshin23.github.io/ExaModels.jl/)  | [![build](https://github.com/sshin23/ExaModels.jl/actions/workflows/test.yml/badge.svg)](https://github.com/sshin23/ExaModels.jl/actions/workflows/test.yml) | [![codecov](https://codecov.io/gh/sshin23/ExaModels.jl/branch/main/graph/badge.svg?token=8ViJWBWnZt)](https://codecov.io/gh/sshin23/ExaModels.jl) |

## Introduction
ExaModels.jl employs what we call **[SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) abstraction for [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming)** (NLPs), which allows for the **preservation of the parallelizable structure** within the model equations, facilitating **efficient, parallel [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)** on the **[GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) accelerators**.

ExaModels.jl is different from other algebraic modeling tools, such as [JuMP](https://github.com/jump-dev/JuMP.jl) or [AMPL](https://ampl.com/), in the following ways:
- **Modeling Interface**: ExaModels.jl enforces users to specify the model equations always in the form of `Generator`. This allows ExaModels.jl to preserve the SIMD-compatible structure in the model equations.
- **Performance**: ExaModels.jl compiles (via Julia's compiler) derivative evaluation codes that are specific to each computation pattern, based on reverse-mode automatic differentiation. This makes the speed of derivative evaluation (even on the CPU) significantly faster than other existing tools.
- **Portability**: ExaModels.jl can evaluate derivatives on GPU accelerators. The code is currently only tested for NVIDIA GPUs, but GPU code is implemented mostly based on the portable programming paradigm, [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). In the future, we are interested in supporting Intel, AMD, and Apple GPUs.

## Quick Start
### Installation
```juliaimport Pkg
Pkg.add("ExaModels.jl")
```
### Usage
The following `JuMP.Model` 
```julia
using JuMP
N = 100

m=JuMP.Model()

JuMP.@variable(
    m,
    x[i=1:N],
    start= mod(i,2)==1 ? -1.2 : 1.
)
JuMP.@NLconstraint(
    m,
    [i=1:N-2],
    3x[i+1]^3+2x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3==0.
)
JuMP.@NLobjective(
    m,
    Min,
    sum(100(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i=2:N)
)
```
can be translated into the following `ExaModels.Model`
```julia
using ExaModels
N= 100

c = ExaModels.Core()
x = ExaModels.variable(
    c, N;
    start = (mod(i,2)==1 ? -1.2 : 1. for i=1:N)
)
ExaModels.constraint(
    c,
    3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3
    for i in 1:N-2
)
ExaModels.objective(c,
    100*(x[i-1]^2-x[i])^2+(x[i-1]-1)^2
    for i in 2:N
)
m = ExaModels.Model(c)
```
Any nonlinear optimization solver in the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) ecosystem can solve the problem. For example, 
```julia
using NLPModelsIpopt
result = ipopt(m)
```
## Supporting ExaModels.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/sshin/ExaModels.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/sshin23/ExaModels.jl/discussions).
