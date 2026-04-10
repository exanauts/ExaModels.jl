# Upgrading from v0.9 to v0.10

This page documents the breaking API changes between ExaModels v0.9 and v0.10.
It is written to be directly usable as a prompt for a code-migration agent.

---

## Summary for an automated migration agent

The following is a complete, machine-readable description of every rename.
Apply each substitution globally to all `.jl` files that import ExaModels.

### 1. Model-building function and macro renames

Every old name has a direct one-to-one replacement.
The old mutating style (`x = variable(c, ...)`) is now deprecated; use the
functional style (`c, x = add_var(c, ...)`) or the macro style
(`@add_var(c, x, ...)`).

| v0.9 function | v0.10 function | v0.10 macro |
|---|---|---|
| `variable(c, ...)` | `add_var(c, ...)` → returns `(c, var)` | `@add_var(c, x, ...)` |
| `parameter(c, ...)` | `add_par(c, ...)` → returns `(c, par)` | `@add_par(c, θ, ...)` |
| `constraint(c, ...)` | `add_con(c, ...)` → returns `(c, con)` | `@add_con(c, g, ...)` |
| `constraint!(c, c1, ...)` | `add_con!(c, c1, ...)` → returns `(c, aug)` | `@add_con!(c, g, ...)` |
| `objective(c, ...)` | `add_obj(c, ...)` → returns `(c, obj)` | `@add_obj(c, f, ...)` |
| `subexpr(c, ...)` | `add_expr(c, ...)` → returns `(c, expr)` | `@add_expr(c, s, ...)` |

### 2. `ExaCore` mutability and `ConcreteExaCore`

In v0.9, `ExaCore` was a mutable struct.
In v0.10, `ExaCore(; concrete = Val(false))` (the default) returns a
`Ref{ExaCore}` that supports both the new functional API and the deprecated
mutating API.  Pass `concrete = Val(true)` (or use `ConcreteExaCore(...)`) to
get the bare immutable struct, which is required for AOT compilation with
`juliac`.

```julia
# v0.9
c = ExaCore()
x = variable(c, 10; lvar = 0.0)
objective(c, x[i]^2 for i in 1:10)
m = ExaModel(c)

# v0.10 — functional style (recommended)
c, x = add_var(ExaCore(), 10; lvar = 0.0)
c, _  = add_obj(c, x[i]^2 for i in 1:10)
m = ExaModel(c)

# v0.10 — macro style (most concise)
c = ExaCore()
@add_var(c, x, 10; lvar = 0.0)
@add_obj(c, x[i]^2 for i in 1:10)
m = ExaModel(c)

# v0.10 — concrete (AOT / juliac)
c = ConcreteExaCore()          # equivalent to ExaCore(concrete = Val(true))
c, x = add_var(c, 10; lvar = 0.0)
c, _  = add_obj(c, x[i]^2 for i in 1:10)
m = ExaModel(c)
```

### 3. Mechanical substitution rules

Apply these regex/string substitutions in order to any v0.9 file.
Each rule is written as `OLD → NEW`.

```
variable(     →  add_var(          # only when first arg is an ExaCore/Ref
parameter(    →  add_par(
objective(    →  add_obj(
constraint!(  →  add_con!(         # must come before constraint( rule
constraint(   →  add_con(
subexpr(      →  add_expr(
```

After renaming the call sites, update the call pattern:

```
# Old: result assigned directly
x = add_var(c, ...)

# New: functional pair destructuring
c, x = add_var(c, ...)
```

If the caller uses `ExaCore(; concrete = Val(false))` (default) or the
deprecated wrappers via a `Ref`, the `c` on the left-hand side is the same
`Ref` object updated in-place, so no further changes are needed there.

### 4. `ExaModelsLinearAlgebra` renamed to `ExaModelsOptimalControl`

The extension that provides LinearAlgebra operations (`dot`, `cross`, `det`,
`norm`, `tr`, `diagm`, `diag`, matrix–vector products) on ExaModels expression
nodes was called `ExaModelsLinearAlgebra` in v0.9.  It is now
`ExaModelsOptimalControl` and is triggered by loading `LinearAlgebra`:

```julia
using ExaModels, LinearAlgebra   # triggers ExaModelsOptimalControl automatically
```

No import of a separate package is needed.

### 5. Complete before/after example

```julia
# ── v0.9 ────────────────────────────────────────────────────────────────────
using ExaModels

n = 100
c = ExaCore()
x = variable(c, n; lvar = -1.0, uvar = 1.0, start = 0.0)
θ = parameter(c, ones(n))
s = subexpr(c, θ[i] * x[i]^2 for i in 1:n)
g = constraint(c, x[i] + x[i+1] for i in 1:n-1; lcon = -1.0, ucon = 1.0)
constraint!(c, g, i => sin(x[i+1]) for i in 1:n-1)
objective(c, s[i] for i in 1:n)
m = ExaModel(c)

# ── v0.10 (functional) ───────────────────────────────────────────────────────
using ExaModels

n = 100
c = ExaCore()                                        # returns Ref{ExaCore} by default
c, x = add_var(c, n; lvar = -1.0, uvar = 1.0, start = 0.0)
c, θ = add_par(c, ones(n))
c, s = add_expr(c, θ[i] * x[i]^2 for i in 1:n)
c, g = add_con(c, x[i] + x[i+1] for i in 1:n-1; lcon = -1.0, ucon = 1.0)
c, _ = add_con!(c, g, i => sin(x[i+1]) for i in 1:n-1)
c, _ = add_obj(c, s[i] for i in 1:n)
m = ExaModel(c)

# ── v0.10 (macro) ────────────────────────────────────────────────────────────
using ExaModels

n = 100
c = ExaCore()
@add_var(c, x, n; lvar = -1.0, uvar = 1.0, start = 0.0)
@add_par(c, θ, ones(n))
@add_expr(c, s, θ[i] * x[i]^2 for i in 1:n)
@add_con(c, g, x[i] + x[i+1] for i in 1:n-1; lcon = -1.0, ucon = 1.0)
@add_con!(c, g, i => sin(x[i+1]) for i in 1:n-1)
@add_obj(c, s[i] for i in 1:n)
m = ExaModel(c)
```
