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

| S | eager core | lazy (vararg fold) | lazy (@generated unroll) |
|---|---|---|---|
| 10 | 14.5 s | 7.0 s | — |
| 50 | 28.8 s | 26.3 s | — |
| 100 | 49.8 s | 67.8 s | 61.3 s |
| 200 | 106.5 s | 322.8 s | 305.9 s |

Eager grows linearly (~0.5 s/statement). The lazy path is at parity or
faster through S ≈ 50 — the realistic hand-written range (LuksanVlcek
models are 4–10 statements, OPF is 16) — and turns quadratic past
S ≈ 70–100, the machine-generated-model regime.

Flag asymmetry to keep in mind: default eager `ExaCore()` is
`concrete = Val(false)` while materialization always builds
`concrete = Val(true)`; part of the small-S advantage is that difference.

## The quadratic tail: first fix disproven, next candidates

Replacing the vararg recursion with a `@generated` flat unroll (so
inference sees one flat body, like user-written code) was the obvious fix
and it is **not** the answer: 322.8 s → 305.9 s at S = 200, noise-level.
The unroll is kept on this branch (marginally better, simpler to reason
about), but the dominant cost is elsewhere: the *types* threaded through
the one inferred materialization body — the core (whose constraint tuple
grows per entry) and the handles tuple — give inference O(S)-sized types in
an S-statement body, an O(S²) minimum that eager construction avoids
because its `add_*` calls are dispatched one at a time from dynamic scope,
never inferred as one unit.

Next candidates, unexplored:

1. **Chunked materialization with function barriers** — split the entries
   into fixed-size blocks, each its own inferred unit; caps the
   per-unit statement count, though the core type still grows across
   chunks.
2. **Type-erased handle threading** — handles as `Vector{Any}` with a
   type-assert at the (rare) `ConAugEntry` lookup; removes one of the two
   growing types from every signature.
3. **Dynamic materialization mode** — for very large tapes, run the fold in
   deliberately `@nospecialize`d/dynamic style (what eager top-level
   construction effectively is) and pay per-call dispatch instead of
   whole-chain inference; a `materialize(tape; dynamic = true)` escape
   hatch.

None of this blocks the release for the realistic range; it gates the
claim that machine-generated many-statement models can migrate.
