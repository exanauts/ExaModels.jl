# [Constraint Augmentation with `add_con!`](@id constraint-augmentation)

!!! warning "Key Concept"
    Constraint augmentation via `add_con!` is perhaps the **single most important
    concept** that distinguishes ExaModels from other algebraic modeling tools like
    JuMP.  Understanding when and why to use it is essential for writing efficient
    ExaModels code --- especially on GPUs.

## Background: Expression Patterns and Compilation

ExaModels achieves its performance by compiling specialized derivative
evaluation code for each unique **expression pattern** in your model.  Each call
to `@add_obj` or `@add_con` introduces one pattern.  The derivative code for
that pattern is then applied over all data points in the iterator via SIMD-style
parallelism.

This design has two important consequences:

1. **Compilation cost scales with the number of unique patterns**, not the
   number of constraints.  A model with 3 patterns over 1,000,000 data points
   compiles almost as fast as the same 3 patterns over 100 data points.

2. **GPU performance depends on few, large kernels.**  Each expression pattern
   becomes a GPU kernel.  Many small kernels mean many kernel launches, which
   is slow.  Fewer patterns with more data points per pattern is ideal.

In JuMP, you can freely write arbitrarily complex per-constraint expressions
because the AD system operates on a scalar expression graph.  In ExaModels,
**keeping the number of expression patterns small** is critical for both
compilation time and GPU scalability.

## What `add_con!` Does

`add_con!` **augments** an existing constraint by adding terms to it.  Instead
of writing one monolithic expression per constraint, you build constraints
incrementally --- each `add_con!` call contributes terms that share a single
expression pattern:

```julia
c = ExaCore(concrete = Val(true))
@add_var(c, x, n)

# Create the base constraint (e.g., a balance equation)
@add_con(c, base, rhs[i] for i in 1:n_buses)

# Augment with contributions from different sources
@add_con!(c, base, a.bus => flow[a.i] for a in arcs)
@add_con!(c, base, g.bus => -gen[g.i] for g in generators)
```

### Syntax

`add_con!` supports two equivalent calling conventions:

**Three-argument form** --- the constraint and index-expression pair are separate arguments:

```julia
@add_con!(c, g, a.bus => p[a.i] for a in arc_data)
# or at the function level:
c, _ = add_con!(c, g, a.bus => p[a.i] for a in arc_data)
```

The syntax `a.bus => p[a.i]` means: "add the value `p[a.i]` to the
constraint row identified by index `a.bus`."

**Two-argument `+=` form** --- the constraint and index are embedded directly:

```julia
@add_con!(c, g[a.bus] += p[a.i] for a in arc_data)
# or at the function level:
c, _ = add_con!(c, g[a.bus] += p[a.i] for a in arc_data)
```

The `+=` form reads more naturally --- "add `p[a.i]` into `g[a.bus]`" ---
and is fully equivalent.  Both forms compile to the same code.

Multiple augmentations on the same base constraint are summed at evaluation time.

## Example: Power Balance in Optimal Power Flow

The canonical use case is power balance constraints in AC Optimal Power Flow.
Each bus must balance power from loads, generators, and line flows:

```math
\sum_{\text{gen}} P_g - \sum_{\text{arc}} P_f - P_d - G_s \cdot V_m^2 = 0 \quad \forall \text{bus}
```

Without `add_con!`, you would need to construct a single expression that
references all generators and arcs connected to each bus --- a complex,
bus-specific expression that varies in structure per bus.  This would produce
as many patterns as there are distinct bus topologies.

With `add_con!`, the same model decomposes into simple, uniform patterns:

```julia
# Base: load and shunt (one pattern, iterated over buses)
@add_con(c, p_balance, b.pd + b.gs * vm[b.i]^2 for b in bus_data)

# Arc flows (one pattern, iterated over arcs)
@add_con!(c, p_balance[a.bus] += p[a.i] for a in arc_data)

# Generation (one pattern, iterated over generators)
@add_con!(c, p_balance[g.bus] += -pg[g.i] for g in gen_data)
```

Three simple patterns instead of one complex per-bus pattern.  Each pattern
compiles quickly and maps to a single efficient GPU kernel.

## Example: Data-Driven Dynamics (Reducing Compilation Blowup)

Consider a quantum optimal control problem with 20 state variables, where each
state equation has 50+ terms involving products of controls and trigonometric
functions.  A naive approach defines a separate dynamics function for each state:

```julia
# Naive: 20 different expression patterns (one per state equation)
function dyn_1(x, u, v, t, n)
    return -2.0 * x[11,t,n] +
           1.414 * u[1,t] * u[2,t] * x[12,t,n] * cos(v[6]) + ...
end

function dyn_2(x, u, v, t, n)
    # ... different structure
end
# ... up to dyn_20

for i in 1:20
    @add_con(c, x[i,j+1,k] - x[i,j,k] - dyns[i](x,u,v,j+1,k) * dt
             for (j,k) in grid)
end
```

This creates 20 distinct expression patterns.  Compilation time grows with each
new pattern, and on GPUs, 20 separate kernels are launched per evaluation.
As the number of states grows to hundreds or thousands, this approach becomes
impractical.

The solution is to **restructure the dynamics as data** and use `add_con!`:

```julia
# Build a single array encoding ALL terms across ALL state equations
terms = [
    (con = state_idx, coeff = coefficient, xi = x_index,
     ui = u_index1, uj = u_index2, vi = v_index)
    for (state_idx, coefficient, x_index, ...) in all_dynamics_terms
]

# One base constraint for all state equations
@add_con(c, dynamics, x[i,j+1,k] - x[i,j,k] for (i,j,k) in state_grid)

# One augmentation pattern for ALL terms across ALL equations
@add_con!(c, dynamics[t.con] += t.coeff * x[t.xi] * u[t.ui] * u[t.uj] * cos(v[t.vi])
    for t in terms
)
```

Now there are only **2 expression patterns** regardless of how many state
variables or terms exist.  The complexity is moved from compiled code into data
arrays, which is exactly what GPUs excel at processing.

!!! info "When to use `add_con!`"
    Use `add_con!` whenever a constraint naturally decomposes into
    contributions from **different data sources** (e.g., generators, arcs,
    loads contributing to bus balance), or when you have **many structurally
    similar terms** that can be parameterized by data.  The goal is always to
    **minimize the number of unique expression patterns** in your model.

## Summary

| Approach | Patterns | Compilation | GPU Performance |
|---|---|---|---|
| One complex expression per constraint | Many (grows with model complexity) | Slow | Poor (many kernel launches) |
| `add_con` + `add_con!` augmentation | Few (fixed by problem structure) | Fast | Excellent (few large kernels) |

See the [Optimal Power Flow example](@ref opf) for a complete working
example using `@add_con!`.
