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
Two further probes localized the cost precisely:

- **Handle threading eliminated**: materializing with handle threading
  removed entirely (valid for the synthetic model — it has no
  augmentations) measured 300.0 s, i.e. no change.
- **Record/materialize split at S = 200**: recording 189.3 s,
  materializing 118.0 s.

That split is the finding. *Materialization is already near-eager cost*
(118 s vs the eager path's 106 s total — consistent with "one function
with S statements and a growing type is fine"). **The quadratic lives in
recording**: `_append(tape, entry) = ExaTape((entries..., entry), config)`
rebuilds the whole entries tuple per statement — S splats of a growing
tuple inside one inferred build body — and recording alone costs 1.8× the
*entire* eager construction.

So the fix target is the tape's accumulation representation, not the fold.
Candidates, unexplored:

1. **Cons-pair accumulation** — record entries as nested pairs
   `(entry, rest)` (O(1) append, no splat); materialization already walks
   entries one at a time and does not care about the shape. Type depth
   still grows, but each append's signature stops carrying a flat O(S)
   splat.
2. **Push-based recording with a type-erased spine** — `Vector{Any}` of
   entries at record time (recording is dynamic anyway — user code is
   type-unstable by design), with the concretely-typed tuple built *once*
   at the end or at materialization. Loses `@inferred` on the tape object
   itself, keeps it on materialization, which is the property that
   matters for trim.
3. **Chunked accumulation** — flush every ~32 entries into a frozen block;
   appends splat at most a chunk.

None of this blocks the release for the realistic range (parity or better
through S ≈ 50); it gates machine-generated many-statement models. Note
candidate 2 is likely the right one on principle: recording is the
*dynamic* phase by design, so paying dynamic-container costs there — and
keeping every static guarantee on the materialization side — matches the
system's own division of labor.
