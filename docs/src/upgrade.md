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

### 2. `ExaCore` and `LegacyExaCore`

In v0.9, `ExaCore` was a mutable struct that was modified in-place by each
model-building call.

In v0.10, `ExaCore` is an immutable struct.  For backward compatibility,
`ExaCore()` (i.e. `concrete = Val(false)`, the default) returns a
`LegacyExaCore` — a thin mutable wrapper that supports the deprecated mutating
wrappers (`variable`, `parameter`, `objective`, `constraint`, `constraint!`,
`subexpr`).  A deprecation warning is emitted at construction time to signal
that this path will be removed in a future release.

Note that `LegacyExaCore` does **not** support the new functional `add_*` API.
Migrate to `ExaCore(concrete = Val(true))` to use `add_var`, `add_obj`, etc.

To obtain the bare immutable `ExaCore` — required for type-stable code and AOT
compilation with `juliac` — pass `concrete = Val(true)`:

```julia
# v0.9
c = ExaCore()
x = variable(c, 10; lvar = 0.0)
objective(c, x[i]^2 for i in 1:10)
m = ExaModel(c)

# v0.10 — functional style (recommended)
c = ExaCore(concrete = Val(true))
c, x = add_var(c, 10; lvar = 0.0)
c, _  = add_obj(c, x[i]^2 for i in 1:10)
m = ExaModel(c)

# v0.10 — macro style (most concise)
c = ExaCore(concrete = Val(true))
@add_var(c, x, 10; lvar = 0.0)
@add_obj(c, x[i]^2 for i in 1:10)
m = ExaModel(c)
```

### 3. Mechanical substitution rules

Apply these regex/string substitutions in order to any v0.9 file.
Each rule is written as `OLD → NEW`.

```
variable(     →  add_var(          # only when first arg is an ExaCore
parameter(    →  add_par(
objective(    →  add_obj(
constraint!(  →  add_con!(         # must come before constraint( rule
constraint(   →  add_con(
subexpr(      →  add_expr(
```

After renaming the call sites, update the call pattern and switch to the
immutable `ExaCore`:

```
# Old: result assigned directly, c mutated in-place
c = ExaCore()
x = variable(c, ...)

# New: functional pair destructuring, c rebound to updated immutable
c = ExaCore(concrete = Val(true))
c, x = add_var(c, ...)
```

### 4. Complete before/after example

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
c = ExaCore(concrete = Val(true))
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
c = ExaCore(concrete = Val(true))
@add_var(c, x, n; lvar = -1.0, uvar = 1.0, start = 0.0)
@add_par(c, θ, ones(n))
@add_expr(c, s, θ[i] * x[i]^2 for i in 1:n)
@add_con(c, g, x[i] + x[i+1] for i in 1:n-1; lcon = -1.0, ucon = 1.0)
@add_con!(c, g, i => sin(x[i+1]) for i in 1:n-1)
@add_obj(c, s[i] for i in 1:n)
m = ExaModel(c)
```
