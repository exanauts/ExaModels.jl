# # Recording Models for AOT Compilation
#
# ExaModels supports ahead-of-time compilation with `juliac --trim=safe`, which
# produces a small static binary containing the model evaluation kernels and a
# solver. Trimmed compilation, however, requires every function in the runtime
# call graph to be statically resolvable — including, in the naive setup, *your*
# model-building code. One type-unstable line in a user model breaks the whole
# compile, with errors that point at the compiler rather than at the model.
#
# The **recorder** removes user code from the compiled call graph entirely.
# Model construction is *recorded* once, at precompile time, in ordinary
# dynamic Julia — and the resulting **tape** is *instantiated* against actual data
# inside the binary by ExaModels' own (type-stable, trim-safe) machinery.
#
# ## Recording a model
#
# Write the model against a tape and a data stand-in, exactly as you would
# write it against an `ExaCore` and a `NamedTuple` of data:

using ExaModels

function luksan_vlcek_x0(i)
    return mod(i, 2) == 1 ? -1.2 : 1.0
end

function luksan_vlcek_con(x, i)
    return 3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
           x[i]exp(x[i] - x[i+1]) - 3
end

function luksan_vlcek_obj(x, i)
    return 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2
end

data = DataTracer((; N = 4))
tape = ExaTape()
@add_var(tape, x, data.N; start = (luksan_vlcek_x0(i) for i = 1:data.N))
@add_con(tape, luksan_vlcek_con(x, i) for i = 1:data.N-2)
@add_obj(tape, luksan_vlcek_obj(x, i) for i = 2:data.N)

# [`DataTracer`](@ref) wraps a *template*: only its field names and types are
# used, never its values. Against an [`ExaTape`](@ref), `data.N`
# returns a typed symbolic value, and the `add_*` calls record their arguments
# instead of building anything. The construction code above runs exactly once,
# here, in dynamic Julia — it is never part of an AOT-compiled call graph.
#
# ## Instantiating it
#
# `ExaModel(tape, args)` instantiates the tape — folding over it and making the
# real `add_var` / `add_con` / `add_obj` calls with every symbolic value
# resolved against the `args` you pass, which can have *different sizes* than
# the template — and builds the model in one call. `args` binds by name as a
# `NamedTuple`; for a single-field schema a bare value works too, and a tape
# that never touched the data tracer needs no args at all (`ExaModel(tape)`
# builds exactly what the same calls against an `ExaCore` would):

model = ExaModel(tape, (; N = 1000))
model = ExaModel(tape, 1000)            # same schema, bare value

# Element type and backend are instantiate-time choices, so a single tape serves
# CPU and GPU at any precision:
#
# ```julia
# using CUDA
# model = ExaModel(tape, (; N = 1000); T = Float32, backend = CUDABackend())
# ```
#
# The instantiated model contains no recorder machinery at all — its evaluation
# path is byte-identical to one built directly against `ExaCore`.
#
# ## What a tape can and cannot capture
#
# A tape freezes the model's *structure*; values and sizes flow through.
# Sizes (`data.N`), ranges (`1:data.N-2`), data arrays used as iterables,
# bounds, and start values are all resolved at instantiate time. But **control flow
# is not recordable**: an `if data.has_storage` would silently bake the
# recording-time branch into the tape, so comparisons and iteration on traced
# values throw a `RecorderStructureError` at record time instead.
#
# A few idioms for common situations:
#
# - A single-expression constraint (`add_con(c, x[1] - a)`) evaluates its
#   expression eagerly, which the recorder cannot do; write it as a 1-element
#   generator: `@add_con(c, x[1] - a for _ in 1:1)`.
# - A structural scalar used inside an expression at a boundary
#   (`constraint(x, N)`) is injected through the iterable:
#   `@add_con(c, constraint(x, n) for n in data.N:data.N)`.
# - An instance scalar computed from data and used inside expressions
#   (`h = 1/(N+1)`) becomes a parameter:
#   `c, h = add_par(c, 1; value = 1/(data.N + 1))`, then `h[1]` in
#   expressions.
#
# ## AOT compilation
#
# ### One command: a model file to a shared library
#
# The **ExaModelC** package (this repository's `/ExaModelC` subpackage) owns
# the compile surface: `compile_library` turns a model file — one that
# defines `build(c, data)` and `make_data(n)::NamedTuple` — into a
# self-contained shared library exposing the NLP through a C interface:
#
# ```julia
# using ExaModelC
# r = compile_library("lv_model.jl"; prefix = "lv", out = "lv_out")
# # → lv_out/lib/liblv.so — the tape is recorded at the generated package's
# #   precompile time; the trimmed call graph contains no user model code.
# ```
#
# `compile_library` also accepts a tape *object* directly
# (`compile_library(tape; template = (; N = 4))`) — tree-built tapes are
# serializable, which is how models recorded from Python (examodels-py's
# `exa.Tape`) compile to shared libraries without a Julia source file.
#
# Any consumer — Julia (via CNLPModels.jl, which also handles the
# libblastrampoline restore needed when hosting a bundled runtime), C, or
# Python — can then instantiate and evaluate the model through
# `lv_new(n) → id` / `lv_obj` / `lv_grad` / `lv_cons` / `lv_jac` / `lv_hess`
# and solve it with any NLPModels-compatible solver.
#
# This is the underlying pattern in every case: the tape is recorded at a
# package's *precompile* time (`const TAPE = build(ExaTape(), DataTracer(template))`)
# and instantiated at runtime — so `juliac --trim=safe` only ever needs to
# compile `instantiate` and the evaluation kernels, never `build`. See
# `docs/design/recorder.md` in the repository for the design rationale.
