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

Two control measurements completed the picture:

- **Eager with `concrete = Val(true)`** (the typed-tuple core; the default
  `ExaCore()` turns out to return the type-erased `LegacyExaCore` wrapper,
  so the original eager column was measuring that): S = 100 → 43.2 s,
  S = 200 → 113.1 s. **Linear.** The growing-typed-tuple pattern is *not*
  quadratic in the core's `add_*`.
- **Materialization ≈ eager, exactly**: 118.0 s vs 113.1 s at S = 200.
  Materialization *is* eager construction, and measures like it.

So the entire excess is recording's 189 s — for a phase that stores
closures and computes nothing. The quadratic is specific to the tape's
`_append` path (entry types carry the raw generator + kwargs types, and
each append splats the growing entries tuple in the recorded body's one
inference unit); the core's own accumulation demonstrably avoids this
cost. Exact micro-cause not yet isolated — it does not need to be, because
recording has no static obligations at all:

**Fix implemented and measured — record dynamic, freeze once.**
Recording at S = 200 drops **189.3 s → 3.5 s** with the erased spine
(materialization unchanged at 117.9 s ≈ eager's 113.1 s), so lazy total ≈
eager total at the far tail while keeping the small-S advantage. A toy
isolated the mechanism first: accumulating 200 *narrow* marker types costs
0.04 s, 200 closure-carrying entries cost 188.7 s — entry width, not kind
mixing, drives the quadratic (which also answers "separate var/con/obj
spines": separation alone cannot fix it, though it remains the right
*structure* for materialization order in the release). One aliasing bug
surfaced and is worth remembering: `TapeCon{length(entries) + 1}` was
evaluated after the append in the same expression — correct under the old
immutable append, off by one under the mutating push — caught by the
augmented-model parity probe, fixed by reading the position before
appending.

**Original design sketch.** Recording is the by-design
dynamic phase (user code may be arbitrarily type-unstable), so the tape
records entries onto a type-erased spine (`Vector{Any}`; O(1) pushes, no
type growth), and a `freeze` step builds the concretely-typed entries
tuple *once* — one O(S) construction instead of S of them — at the
record/materialize boundary (end of construction, or entry to
`ExaModel`/`instantiate`). Everything downstream keeps today's guarantees:
`@inferred` materialization, trim-safety, serialization — those attach to
the frozen tape, which is exactly today's `ExaTape`. Expected cost:
recording ~0, freeze ~one splat, materialization 118 s ≈ eager 113 s —
i.e. lazy total ≈ eager total at every S, with the small-S advantage
kept.

None of this blocks the release for the realistic range (parity or better
through S ≈ 50); it removes the machine-generated-model caveat once
implemented.

## Future: pattern merging at freeze (type shortening)

The concrete tape stays first-class, and `freeze` grows into the compile
step: beyond concretizing the spine, it can *shorten the number of distinct
con/obj types*. Creation cost (eager and materialization alike) grows
superlinearly in the number of distinct entry types, not data size — and
constants in expression trees are field values, not type parameters, so
structurally-identical trees already share types. What keeps entries
distinct is closure identity (one type per source location). Freeze can
trace closures to trees (sentinel tracing, as materialization already
does), group same-typed trees, and merge each group into one entry with
concatenated iterators/parameter tables: the one-type-per-pattern SIMD
philosophy applied across statements. The S = 200 benchmark — 200
statements of one shape differing by a literal — would collapse to an
effective S of 1. Not part of the initial release; the seam (freeze) ships
with it.
