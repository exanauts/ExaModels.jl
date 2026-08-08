# ExaTape: record/replay model construction for AOT compilation

## Problem

The current AOT path (`test/JuliaCTest` compiling `LuksanVlcekApp.jl` /
`COPSApp.jl`) puts the *user's* model-building function inside the
`juliac --trim=safe` call graph. If the user writes type-unstable or
trim-hostile code, compilation fails (or miscompiles) with errors that point
deep into the compiler, not at the user's model. The compilable surface must
become ExaModels' code, not the user's.

## Idea

JAX-style tracing, adapted to ExaModels' generator-based syntax:

- **`DataTracer`** — a record-time stand-in for the data `NamedTuple`.
  `data.N` returns a *typed symbolic node* (`DataField{Int, :N}`), and a small
  closed set of operations on tracer values (`data.N - 2`, `1:data.N`,
  `length(data.bus)`) return further typed nodes (`TracerExpr`). At replay
  time the same accesses hit a real `NamedTuple`.
- **`ExaTape`** — a record-time stand-in for `ExaCore`. `add_var` / `add_con`
  / `add_obj` dispatch on it and *record* their arguments instead of building
  anything, threading the tape through `(c, x) = add_var(c, ...)` exactly like
  the real API (so `@add_var` etc. work unchanged). Entries accumulate in a
  concretely-typed nested tuple — the same pattern `ExaCore` itself uses for
  `var`/`cons`/`obj`.
- **`record(build, template)`** — runs the user's `build(tape, tracer)` once,
  in ordinary dynamic Julia. User code may be arbitrarily type-unstable;
  nothing here is ever AOT-compiled.
- **`replay(tape, data; T, backend)`** — a type-stable fold over the tape
  entries that makes the *real* `add_var`/`add_con`/`add_obj` calls against a
  real `ExaCore`. Backend and precision are chosen at replay, not record —
  record once, replay on CPU or GPU.

## Key mechanism: replay-time re-tracing via `Ref` binding

ExaModels traces a constraint/objective generator by calling `gen.f(DataSource())`
— the closure runs once on a sentinel, and variable references become `Var`
nodes whose offsets are plain `Int` *fields* (`Var(Node2(+, i, offset))`),
computed from the `Variable` handle the closure captured.

The tape therefore stores the user's closures **uncalled**, and the recorded
`add_var` returns a `TapeVar` — a thin handle holding an *empty*
`Ref{Variable{S,O,T}}` whose concrete type is computable at record time from
the recorded dims (`S = Tuple{replay_type.(dims)...}`, `O = Int`). At replay:

1. `VarEntry` replays through the real `add_var`, which computes correct
   offsets for the *actual* sizes, and binds `tapevar.ref[] = v`.
2. `ConEntry`/`ObjEntry` rebuild `Base.Generator(f, resolve(iter, data))` and
   pass it to the real `add_con`/`add_obj`. Tracing happens inside the real
   API; `getindex(::TapeVar, i...)` delegates to the bound real `Variable`.

Consequences:

- Offsets are correct by construction — never recorded, never rewritten.
- The replayed `ExaCore`/`ExaModel` contains **zero recorder types**; the hot
  evaluation path is byte-identical to a directly-built model.
- A tape is replayable many times with different data/sizes/backends, but the
  `Ref` binding makes concurrent replays of the *same tape object* racy —
  replay a tape from one thread at a time (documented; enforcement later).

## Semantics: what a tape can and cannot capture

A tape freezes *structure*; values and sizes flow through.

- Fine: `add_var(c, data.N)`, `1:data.N - 2` as a generator iterable, a
  `start =` generator over a traced range, replaying at sizes different from
  the template.
- Frozen (must error at record time, not silently specialize): branching on
  data (`if data.has_storage`, `data.N > 5`). Comparisons and iteration on
  tracer values throw `RecorderStructureError` with an explanatory message.
  A `static(...)` escape hatch (deliberately freeze a value from the
  template) is future work.
- Fundamental limit: `sum(... for k in range)` *inside* an expression unrolls
  into `SumNode`'s type parameters at trace time, so an inner-sum range can
  never be data-dependent. (Outer iterables can.) Record-time error.

## Type stability

- Record phase: no stability requirement at all (runs once, dynamically).
- The tape: concrete nested tuples; every entry field concretely typed.
- Replay: recursive vararg fold (`_replay(c, data, e, rest...)`), the same
  compile-time-unrolled pattern ExaModels uses everywhere; `resolve` is typed
  by the `TracerValue{T}` parameter. Gate: `@inferred replay(tape, data)` in
  the test suite.

## Why this fixes AOT

A generated app does

```julia
const TAPE = record(build, template)          # runs at precompile time
main(data) = solve(ExaModel(replay(TAPE, data)))
```

`record` executes during precompilation, where dynamic Julia is fine. The
runtime call graph that `juliac --trim=safe` must compile is `replay` + the
standard ExaModels kernels — all ours. The user's `build` is never called at
runtime.

## PoC scope (this branch, LuksanVlcek example)

- In: `add_var` (dims, `start`/`lvar`/`uvar`, names via `@add_var`),
  `add_con`, `add_obj` (generator forms + macros), tracer ops
  (`getproperty`, `+`, `-`, `*`, unary `-`, `:`, `length`), structure-error
  guardrails, `record`/`replay`, `@inferred` gate, correctness tests against
  a directly-built model.
- Later (mechanical): `add_par`, `add_expr`, `add_con!` augmentation entries,
  richer tracer vocabulary (`eachindex`, `zip`, `enumerate`, nested
  fields/arrays-of-namedtuples), binding `Constraint` handles for
  `multipliers`. 
- Later (designed, not mechanical): closures capturing tracer *scalars*
  (eager unwrap through a bound `Ref` at trace time), `static()`,
  tape → generated-source dump for debuggability, the app-package generator
  around `replay`, thread-safety of replay.

## Naming

`DataTracer`, `ExaTape`, `record`, `replay`, `TapeVar`,
`RecorderStructureError`. File: `src/recorder.jl`, tests:
`test/RecorderTest/RecorderTest.jl`.
