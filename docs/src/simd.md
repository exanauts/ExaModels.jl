# [SIMD Abstraction](@id simd)

In this page, we explain what SIMD abstraction of nonlinear program is, and why it can be beneficial for scalable optimization of large-scale optimization problems. More discussion can be found in our [paper](https://arxiv.org/abs/2307.16830).

## What is SIMD abstraction?
The mathematical statement of the problem formulation is as follows.
```math
\begin{aligned}
  \min_{x^\flat\leq x \leq x^\sharp}
  & \sum_{l\in[L]}\sum_{i\in [I_l]} f^{(l)}(x; p^{(l)}_i)\\
  \text{s.t.}\; &g^\flat \leq \left[g^{(m)}(x; q_j)\right]_{j\in [J_m]} +\sum_{n\in [N_m]}\sum_{k\in [K_n]}h^{(n)}(x; s^{(n)}_{k}) \leq g^\sharp,\quad \forall m\in[M]
\end{aligned}
```
where $f^{(\ell)}(\cdot,\cdot)$, $g^{(m)}(\cdot,\cdot)$, and
$h^{(n)}(\cdot,\cdot)$ are twice differentiable functions with respect
to the first argument, whereas $\{\{p^{(k)}_i\}_{i\in [N_k]}\}_{k\in[K]}$,
$\{\{q^{(k)}_{i}\}_{i\in [M_l]}\}_{m\in[M]}$, and
$\{\{\{s^{(n)}_{k}\}_{k\in[K_n]}\}_{n\in[N_m]}\}_{m\in[M]}$ are
problem data, which can either be discrete or continuous.
It is also assumed
that our functions $f^{(l)}(\cdot,\cdot)$, $g^{(m)}(\cdot,\cdot)$, and
$h^{(n)}(\cdot,\cdot)$ can be expressed with computational
graphs of moderate length. 

## Why SIMD abstraction?
Many physics-based models, such as AC OPF, have a highly repetitive
structure. One of the manifestations of it is that the mathematical
statement of the model is concise, even if the practical model may contain
millions of variables and constraints. This is possible due to the use of
repetition over a certain index and data sets. For example,
it suffices to use 15 computational patterns to fully specify the
AC OPF model. These patterns arise from (1) generation cost, (2) reference
bus voltage angle constraint, (3-6) active and reactive power flow (from and to),
(7) voltage angle difference constraint, (8-9) apparent
power flow limits (from and to), (10-11) power balance equations,
(12-13) generators' contributions to the power balance equations, and
(14-15) in/out flows contributions to the power balance
equations. However, such repetitive structure is not well exploited in
the standard NLP modeling paradigms. In fact, without the SIMD
abstraction, it is difficult for the AD package to detect the
parallelizable structure within the model, as it will require the full
inspection of the computational graph over all expressions.  By
preserving the repetitive structures in the model, the repetitive
structure can be directly available in AD implementation.

Using the multiple dispatch feature of Julia, ExaModels.jl generates
highly efficient derivative computation code, specifically compiled
for each computational pattern in the model. These derivative evaluation codes can be run over the data in various GPU array formats,
and implemented via array and kernel programming in Julia Language. In
turn, ExaModels.jl has the capability to efficiently evaluate first and
second-order derivatives using GPU accelerators.
