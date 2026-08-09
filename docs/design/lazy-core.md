# Make `ExaCore` lazy — design (v3: the concrete lazy core)

The release makes `ExaCore` lazy without the user seeing a difference:
construction stays today's eager, concretely-typed construction — expression
trees are traced at each `add_*` call exactly as now — and *only values*
are deferred. An **args sentinel** (the `DataTracer` mechanism) may flow
into any value position: dimensions, iterator ranges, bounds, starts,
parameter values. Whatever depends on a sentinel — offsets above all — is
computed when `ExaModel(core, args)` materializes the model. The core is
concretely typed at construction; its values need not be.

This supersedes the tape-shaped design explored first (recording closures,
materializing later). The measurements below justify the inversion: the
tape's costs came from storing *untraced closures* (wide types); tracing
eagerly stores the compiled, compressed, narrow entry types, and every
pathology disappears by construction.

## What it wins (each point measured or verified; populations below)

1. **No width pathology, ever.** The toy isolates it: accumulating 200
   narrow entries costs 0.04 s; 200 closure-carrying entries cost 188.7 s
   (matching the tape's recorded 189.3 s). Eagerly-traced entries are the
   narrow kind — the eager core's own accumulation, measured linear
   (43.2 / 113.1 s at S = 100/200).
2. **Re-instantiation is a value pass.** All types exist after
   construction, so `ExaModel(core, args₂)` computes offsets, resolves
   sentinel expressions, and builds arrays — no type compilation. Under
   the tape, every instantiation re-ran typed construction (117.9 s at
   S = 200). One core, many cheap instantiations: a capability neither
   today's eager core nor the tape has.
3. **Construction-time error locality is restored.** Tracing runs at the
   `add_con` line, so a broken user expression fails there — one of the
   tape design's documented regressions disappears.
4. **Universal serialization.** Post-trace entries are closure-free:
   `SIMDFunction(T, gen, ...)` calls `gen.f(DataSource())` once and stores
   only the traced tree (`src/simdfunction.jl:41-43`; confirmed by the
   author). So *every* lazy core serializes — closure-built models
   included — where the tape could only serialize tree-built ones.
   `compile_library` extends to any model.
5. **Smaller AOT surface.** The trimmed runtime is the value pass, simpler
   than tape replay.
6. **User-invisible.** Today's API verbatim; sentinels are additive. A
   core built with no sentinels materializes to exactly today's model
   (the identity property, already a gated test on `ss/recorder`).

## Mechanisms required (each already exists in some form)

- **Deferred offsets as field values.** Offsets are `Int` *fields* inside
  node types (`Node2{+, I, Int64}`), not type parameters — trace with
  placeholder offsets, fill at materialization with a type-preserving,
  value-only tree walk (the `_rebind` machinery from `ss/recorder`,
  relocated). `@inferred` materialization holds because no types change.
- **Sentinel-typed tracing.** Tracing needs element *types*, not values;
  `DataTracer{NT}` carries the template's field types, including record
  eltypes for `for b in data.branch`-style iteration. Ranges, `fill`s,
  and deferred comprehensions stay value-level exactly as on
  `ss/recorder`.
- **Deferred-value slots inside concrete containers.** A slot holds either
  a plain value or a sentinel expression; its type is fixed per model at
  construction (concrete, possibly sentinel-typed), so entry types stay
  concrete without a `Union`.

## Semantics to document

- Structure still cannot depend on sentinel *values* (comparisons and
  iteration on sentinels throw at construction — unchanged guardrail).
- A sentinel-free core has today's semantics exactly. With sentinels,
  value resolution happens at `ExaModel(core, args)`; args follow the
  shapes already shipped on `ss/recorder` (NamedTuple by name, bare value
  for single-field schemas, `nothing` default).

## Roadmap: pattern merging (type shortening)

Creation cost grows with the number of *distinct* con/obj types, not data
size. Since constants are field values, structurally-identical trees
already share types; merging same-typed entries into one entry with
concatenated iterators/parameter tables collapses repeated-pattern models
(the S = 200 benchmark is one pattern — effective S = 1). Under v3 the
merge point is construction itself (entries arrive traced), even more
natural than the tape's freeze step.

## Measurement appendix (evidence base; S = distinct add_con statements, cold, n = 100 fixed)

| S | eager `ExaCore()` (= LegacyExaCore wrapper) | eager typed `Val(true)` | tape record | tape materialize |
|---|---|---|---|---|
| 10 | 14.5 | — | 7.0 total | (with materialize) |
| 50 | 28.8 | — | 26.3 total | (with materialize) |
| 100 | 49.8 | 43.2 | 67.8 total | (with materialize) |
| 200 | 106.5 | 113.1 | 189.3 → **3.5** (erased spine) | 118.0 |

- Materialize ≈ eager typed (118.0 vs 113.1): tape materialization *is*
  eager construction.
- Disproven along the way: recursion shape (`@generated` unroll, 322.8 →
  305.9 s — noise) and handle threading (removed entirely, 300.0 s —
  noise).
- Toy discriminator: 200 narrow entries 0.04 s vs 200 closure-carrying
  entries 188.7 s — entry *width*, not kind mixing, drives the cost; this
  also answers "separate var/con/obj spines" (separation alone cannot
  help; per-kind structure remains good for materialization order).
- The erased spine + `freeze` (implemented on this branch) fixed the
  tape's recording cost and remains a fallback pattern; v3 makes it
  unnecessary by never storing closures at all.
- Both eager columns are mildly superlinear (~×2.6 per ×2 at the tail) — a
  pre-existing property of creation and the target of pattern merging.
- Aliasing lesson kept from the spine work: positions must be read before
  a mutating append (`TapeCon{length(entries)+1}` evaluated after the
  push was off by one; caught by the augmented-model parity probes).
