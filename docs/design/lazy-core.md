# Exploration: make `ExaCore` lazy (breaking release)

The framing is not "remove ExaCore" — it is **`ExaCore` becomes lazy**.
Construction *records*; `ExaModel(core, args)` *materializes*. Concretely,
the release renames today's `ExaTape` to `ExaCore` and internalizes the
eager accumulator (today's `ExaCore`) as the materialization fold's
internal state. Explored on branch `ss/tape-only`.

## Why the migration is nearly free

`ExaModel(tape) ≡ ExaModel(core)` is a tested identity (RecorderTest,
args-shapes testset). So the code every user already has —

```julia
c = ExaCore()
@add_var(c, x, N)
...
m = ExaModel(c)
```

— keeps working **verbatim** under the lazy core, by that identity: values
are inline, nothing references the data tracer, and materialization builds
exactly the model eager construction did. What users *gain*, on the same
object, is everything recording enables: symbolic data fields
(`DataTracer`), instantiation at any `args`/precision/backend, tape
serialization, and the AOT route.

- The MOI and JuMP conversions build the eager accumulator *internally*
  and hand it to `ExaModel`; the rename does not touch them. Same for the
  backend extensions.
- The repo has done this before: `LegacyExaCore` in `src/deprecated.jl` is
  the worked precedent for changing what `ExaCore` means across a release.

## What needs real work

| Surface | Today | Under the lazy core | Size |
|---|---|---|---|
| `add_*` + macros | dispatch on both objects | done | — |
| `T`/`backend` in `ExaCore(T; backend)` | construction-time | become `ExaModel`/`instantiate` keywords (already are); the lazy `ExaCore(; minimize)` keeps only what recording needs | small |
| Oracle registration | `objective(c, o)`, `constraint(c, o)` on the eager core | record an `OracleEntry`; materialization calls through. Oracle cores join the closure class (callbacks are opaque — not serializable), consistent with closure generators | ~50 lines + tests |
| Two-stage | `TwoStageExaCore(ns)` — an eager-core variant | needs a scenario-aware recording design (backlogged); keep `TwoStageExaCore` on the eager path for one cycle | separate PR |
| Benchmark repos' `_model` | eager-core constructors | become `ExaModel(<name>_tape(), n)`; the companion-PR tape builders are the survivors | mechanical |

## Semantic changes to document (the honest "breaking" part)

1. **Laziness is observable**: generators are iterated at materialization,
   not construction — mutating a captured array between building and
   `ExaModel(c)` diverges from eager semantics.
2. Mid-construction introspection of concrete offsets (reading counts off a
   half-built core) no longer exists; structure exists after
   materialization.
3. Errors in user expressions surface at `ExaModel(c)`, not at the `add_*`
   line that recorded them (stack traces still point into the stored
   closure).

## Compile-time scaling (statement count S; the risk item)

Distinct `add_con` statements grow the entry tuple types — the compile-time
axis. Measured cold (fresh process per cell), n = 100 fixed; `construct` =
build + `ExaModel` + evaluation-structure compile.

<!-- SCALING-TABLE: fill from ~/rune-work/logs/scaling.log -->

Flag asymmetry to keep in mind: default eager `ExaCore()` is
`concrete = Val(false)` while materialization always builds
`concrete = Val(true)`; part of any delta is that difference, not fold
overhead.

## Not explored yet

Whether the materialization fold hits a recursion/inference limit beyond
S ≈ several hundred (it recurses per entry; eager construction is
user-driven and has no single recursion). If S = 200 degrades, the fold can
be `@generated`-unrolled.
