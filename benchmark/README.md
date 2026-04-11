# ExaModels Benchmarks

This directory contains the benchmark suite used to track the performance of
ExaModels across commits and hardware backends.  Results are reported as
wall-clock minimums over a fixed number of samples for each of the five NLP
callbacks: objective (`obj`), constraints (`cons`), gradient (`grad`),
Jacobian (`jac`), and Hessian (`hess`).

## Problem suite

| Group       | Sizes             | Description                                      |
|:------------|:------------------|:-------------------------------------------------|
| `rosenrock` | 1k / 10k / 100k   | Extended Rosenbrock (COPS test problem)          |
| `OPF`       | case14 / case1354 / case30000 | AC Optimal Power Flow (pglib-opf cases) |
| `chain`     | 10 / 100 / 1000   | Hanging chain (COPS)                             |
| `elec`      | 10 / 100 / 1000   | Electrons on a sphere (COPS)                     |

## Supported backends

Each benchmark can be run on one or more KernelAbstractions backends:

| Backend   | Keyword    | 
|:----------|:-----------|
| CPU       | `nothing`  |
| NVIDIA    | `cuda`     |
| AMD       | `amdgpu`   |
| Intel GPU | `oneapi`   |
| Apple GPU | `metal`    |

## Running locally

```sh
# CPU only (default)
make

# Single backend
make cuda
make amdgpu
make oneapi
make metal

# All available backends
make full

# Re-run comparison without re-benchmarking
make compare
```

Each target runs `runbenchmark.jl` for both `main` and `current`, then calls
`compare.jl` to print the ratio table.

## Running manually

```sh
# Build results for a specific revision + backend
julia runbenchmark.jl main    cuda
julia runbenchmark.jl current cuda

# Compare
julia --project=. compare.jl
```

## Output format

`runbenchmark.jl` prints a table while running:

```
==========================================================================================
  name                        nvar   ncon |      obj     cons     grad      jac     hess
==========================================================================================
  rosenrock-1000-CUDA         2994   2996 | 1.23e-05 4.56e-06 2.34e-05 3.45e-06 6.78e-06
  ...
```

`compare.jl` prints ratios (`current / main`); values below 1.0 are
improvements:

```
Relative timing: current / main  (values < 1.0 are improvements)

==============================================================
  name                        |      obj     cons     grad      jac     hess
==============================================================
  rosenrock-1000-CUDA         |    0.982    1.003    0.971    0.995    0.988
  ...
```

## CI integration

Benchmarks run automatically on every pull request via the `benchmark` GitHub
Actions workflow (`.github/workflows/benchmark.yml`).  Trigger options from a
PR comment:

| Comment                  | Backends run          |
|:-------------------------|:----------------------|
| `run benchmark`          | all (CPU only)        |
| `run benchmark cuda`     | CUDA only             |
| `run benchmark amdgpu`   | AMDGPU only           |
| `run benchmark oneapi`   | oneAPI only           |
| `run benchmark metal`    | Metal only            |
| `run benchmark nothing`  | CPU only              |
| `run benchmark full`     | all                   |

Results are posted as a comment on the PR and updated on re-runs.
