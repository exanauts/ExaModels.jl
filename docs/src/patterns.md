# [Macro Behavior: Returning `core` from Functions](@id macro-behavior)

The `@add_var`, `@add_par`, `@add_obj`, `@add_con`, `@add_con!`, and `@add_expr`
macros all share an important behavior that users should understand,
especially when writing model-building functions.

## How the Macros Work

In ExaModels v0.10, `ExaCore` (created with `concrete = Val(true)`) is an
**immutable** struct. Every model-building call --- `add_var`, `add_obj`,
`add_con`, etc. --- returns a **new** core rather than modifying the old one:

```julia
c = ExaCore(concrete = Val(true))
c, x = add_var(c, 10)       # c is rebound to a new ExaCore
c, _ = add_obj(c, x[i]^2 for i in 1:10)  # c is rebound again
```

The `@add_var`-family macros are thin wrappers that do the same thing
behind the scenes --- they **rebind** the core variable in the calling scope:

```julia
c = ExaCore(concrete = Val(true))
@add_var(c, x, 10)          # expands to: c, x = add_var(c, 10)
@add_obj(c, x[i]^2 for i in 1:10)  # expands to: c, _ = add_obj(c, ...)
```

This rebinding is the key behavior to keep in mind: after every macro call,
`c` points to a new `ExaCore` that contains all previously accumulated
information **plus** the newly added component.

## The Pitfall: Functions That Forget to Return `core`

This works fine at the top level. But when you build a model inside a
**function**, the rebinding of `c` only affects the function's local scope.
If you forget to return the final `core`, the caller never sees the
accumulated model information:

```julia
# WRONG --- the caller gets nothing useful
function build_model_wrong()
    c = ExaCore(concrete = Val(true))
    @add_var(c, x, 10)
    @add_obj(c, x[i]^2 for i in 1:10)
    @add_con(c, x[i] + x[i+1] for i in 1:9)
    # c is local --- it is lost when the function returns!
end
```

The fix is straightforward: **return `core`** (or an `ExaModel` built from it)
at the end of the function:

```julia
# CORRECT --- return the ExaModel (or core) to the caller
function build_model()
    c = ExaCore(concrete = Val(true))
    @add_var(c, x, 10)
    @add_obj(c, x[i]^2 for i in 1:10)
    @add_con(c, x[i] + x[i+1] for i in 1:9)
    return ExaModel(c)   # <-- pass the final core to ExaModel
end
```

This applies to **all** `@add_*` macros: `@add_var`, `@add_par`, `@add_obj`,
`@add_con`, `@add_con!`, and `@add_expr`. Each one rebinds the core variable,
so the last `c` in your function is the only one that holds the complete model
definition. Either return it directly or pass it to `ExaModel(c)`.

## Returning `core` for Further Composition

When you want to build a model in stages across multiple functions, return
the core itself so the caller can continue adding components:

```julia
function add_dynamics!(c, x, u, data)
    @add_con(c, x[i+1] - x[i] - u[i] * data[i].dt for i in 1:length(data))
    return c   # <-- return updated core for further building
end

function full_model()
    c = ExaCore(concrete = Val(true))
    @add_var(c, x, 11)
    @add_var(c, u, 10)
    c = add_dynamics!(c, x, u, data)   # rebind c with the returned core
    @add_obj(c, x[i]^2 for i in 1:11)
    return ExaModel(c)
end
```

!!! tip
    A good rule of thumb: any function that calls `@add_*` macros should
    **return `c`** as its last expression.
