# ExaTape: record/replay model construction for AOT compilation

## Problem

The pre-tape AOT path (compiling per-model app packages) put the *user's*
model-building function inside the `juliac --trim=safe` call graph. If the user writes type-unstable or
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
- **`ExaModel(tape, data; T, backend)`** — the public one-call form: a
  type-stable fold over the tape entries makes the *real*
  `add_var`/`add_con`/`add_obj` calls against a real `ExaCore`, and the model
  is built from it. Backend and precision are chosen at replay, not record —
  record once, replay on CPU or GPU. Construction is explicit — the user
  builds against `ExaTape()` and `DataTracer(template)` directly, in ordinary
  dynamic Julia; construction code may be arbitrarily type-unstable, since
  nothing there is ever AOT-compiled. (`ExaModels.replay(tape, data) ->
  ExaCore` remains as the unexported two-step form.)

## Key mechanism: replay-time re-tracing via `Ref` binding

ExaModels traces a constraint/objective generator by calling `gen.f(DataSource())`
— the closure runs once on a sentinel, and variable references become `Var`
nodes whose offsets are plain `Int` *fields* (`Var(Node2(+, i, offset))`),
computed from the `Variable` handle the closure captured.

The tape therefore stores the user's closures **uncalled**, and the recorded
`add_var` returns a `TapeVar` — a thin handle holding an *empty*
`Ref{Variable{S,O,T}}` whose concrete type is computable at record time from
the recorded dims (`S = Tuple{replay_type.(dims)...}`, `O = Int`). Indexing a
tape handle **always** yields a sentinel node (`TapeVarIndexed`) — never a
branch on binding state, which would make traced tree types a `Union` and
destroy replay inferability. At replay:

1. `VarEntry` replays through the real `add_var`, which computes correct
   offsets for the *actual* sizes, and binds `tapevar.ref[] = v`.
2. `ConEntry`/`ObjEntry` trace their stored closure themselves
   (`f(DataSource())`), rewrite the sentinels into offset-correct references
   with the `_rebind` tree walk, and feed the result to the *low-level*
   `(expr, pars)` forms. Augmentation pairs rebind the same way and travel
   through the named `FixedExpr` functor. Expression trees recorded directly
   (`add_con(tape, expr::AbstractNode, itr)` — the Python path) skip the
   trace and go straight to `_rebind`.

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
const TAPE = build(ExaTape(), DataTracer(template))   # runs at precompile time
main(data) = solve(ExaModel(TAPE, data))
```

`record` executes during precompilation, where dynamic Julia is fine. The
runtime call graph that `juliac --trim=safe` must compile is `replay` + the
standard ExaModels kernels — all ours. The user's `build` is never called at
runtime.

## Current scope (this branch)

- Recording: `add_var` / `add_par` / `add_con` / `add_con!` / `add_obj`
  (generator forms + macros, names, tags, bounds/start kwargs).
- Constraint handles for `add_con!` bind *positionally*: `add_con` on the tape
  returns `TapeCon{K}` (K = entry index) and replay threads a tuple of
  realized handles, because a `Ref`'s concrete type would need the traced
  `SIMDFunction` type, unknowable at record time. Variables and parameters
  bind through concretely-typed `Ref`s as described above.
- Tracer vocabulary: `getproperty`, `+`, `-`, `*`, `/`, `div`, `rem`, `mod`,
  unary `-`, `floor`/`ceil(T, ·)`, 2- and 3-arg `:`, `length`, `fill`,
  comprehensions over traced ranges (`collect` of a generator whose iterable
  is traced; the body must not capture tracers), `Iterators.product` with
  traced components (multi-dimensional generators).
- Guardrails: comparisons/iteration on tracers throw `RecorderStructureError`.
- Tests: recorded-vs-direct comparisons (values, derivatives, sparsity)
  against independently-implemented builders — the docs LuksanVlcek, the full
  18-model LuksanVlcekBenchmark set (tape builders in that repo's ExaModels
  extension), COPS chain/camshape (likewise), the repo's 2-D Luksan (product
  generators + `add_con!`), and the AC power flow model (one tape replayed
  across different pglib grids) — plus `@inferred replay` on every tape and
  the AOT leg: `compile_library` per model, consumed through CNLPModels.jl
  and solved host-side against in-process references.

### GPU compatibility

Backend and precision are replay-time choices, so one tape serves CPU and
GPU: `ExaModel(tape, data; T = Float32, backend = CUDABackend())` (verified:
CUDA replay of a recorded tape matches the CPU model's obj/grad/cons; a
Float32 device replay builds). Nothing in the tape, its serialization, or
the C ABI encodes an array type. For AOT, the generated library currently
replays on the default (CPU) backend; the seam for device libraries is
confined to two places — the backend argument of the generated
`ExaModel(TAPE, data)` call, and the C boundary (host pointers, with a
device library copying at the boundary until device-pointer entry points
are added). When juliac can trim device code, `compile_library` gains a
backend option and the rest of the pipeline lifts unchanged.

### Transcription idioms (writing a model against the recorder)

- A single-expression constraint/objective (`add_con(c, expr)`) evaluates the
  expression eagerly, which an unbound `TapeVar` cannot do; write it as a
  1-element generator (`expr for _ in 1:1`) — semantically identical to the
  real API's own `pars = 1:1` handling.
- A structural scalar used *inside* an expression (`constraint(x, N)` at a
  boundary) is injected through the iterable: `for n in data.N:data.N`. The
  value then flows through the iteration element at evaluation time.
- An instance scalar computed from data (`h = 1/(N+1)`) becomes a parameter:
  `c, h = add_par(c, 1; value = 1/(data.N + 1))`, then `h[1]` in expressions.
  (The direct model bakes `h` as a `Constant`, so trees — and possibly
  sparsity orderings — differ while assembled operators match.)

## Shared-library output (`compile_library`, ExaModelsJuliaC extension)

`compile_library` generates a throwaway app package with the C-ABI
`@ccallable` surface (prefix-interpolated, handle-based: `<prefix>_new(n) →
id`, id-first entry points, any number of coexisting instances), pins it to
the running ExaModels checkout, and drives JuliaC's
compile/link/bundle(`privatize = true`) pipeline to a self-contained `.so`.
Two input forms: a *model file* defining `build(c, data)`/`make_data(n)`
(recorded at the generated package's precompile time), or a *tape object*
(`compile_library(tape::ExaTape; template)`) — tree-built tapes contain only
named types, so the tape is serialized into the app and deserialized at its
precompile time; this is how models recorded from Python compile without any
Julia source being written.
ABI: `<prefix>_init(n)`, meta/structure queries, and
obj/grad/cons/jac/hess evaluations — 1-based indices, lower-triangle
Lagrangian Hessian with `obj_weight`, `Cint` statuses. The Julia-side
consumer (including the libblastrampoline snapshot/restore required when
hosting a bundled runtime inside a Julia process) is CNLPModels.jl.

## Later

- `add_expr` entries; binding `Constraint` handles for post-solve
  `multipliers`; closures capturing tracer scalars (eager unwrap through a
  bound `Ref` at trace time); `static()` escape hatch (needed for e.g. COPS
  models whose *inner* `sum` ranges are data-dependent — the SumNode
  fundamental limit above); tape → generated-source dump; a richer data ABI
  for `compile_library` (pointer-based array marshalling instead of
  `new(n::Cint)`; the tape-input form is currently limited to
  single-integer-field templates); thread-safety of replay.

## Naming

`DataTracer`, `ExaTape`, `record`, `replay`, `TapeVar`,
`RecorderStructureError`. File: `src/recorder.jl`, tests:
`test/RecorderTest/RecorderTest.jl`.
