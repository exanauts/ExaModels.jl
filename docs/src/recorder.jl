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
# dynamic Julia — and the resulting **tape** is *replayed* against actual data
# inside the binary by ExaModels' own (type-stable, trim-safe) machinery.
#
# This is the tracing model familiar from JAX: the data stand-in is a tracer,
# the tape is the jaxpr, and `replay` is the compiled artifact.
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

tape = record((; N = 4)) do c, data
    @add_var(c, x, data.N; start = (luksan_vlcek_x0(i) for i = 1:data.N))
    @add_con(c, luksan_vlcek_con(x, i) for i = 1:data.N-2)
    @add_obj(c, luksan_vlcek_obj(x, i) for i = 2:data.N)
    c
end

# The first argument to [`record`](@ref) is a *template*: only its field names
# and types are used, never its values. Inside the build function, `c` is an
# [`ExaTape`](@ref) and `data` is a [`DataTracer`](@ref); `data.N` returns a
# typed symbolic value, and the `add_*` calls record their arguments instead of
# building anything. The user code above runs exactly once, here, in dynamic
# Julia — it is never part of an AOT-compiled call graph.
#
# ## Replaying it
#
# [`replay`](@ref) folds over the tape and makes the real `add_var` /
# `add_con` / `add_obj` calls with every symbolic value resolved against the
# data you pass — which can have *different sizes* than the template:

core = replay(tape, (; N = 1000))
model = ExaModel(core)

# Element type and backend are replay-time choices, so a single tape serves
# CPU and GPU at any precision:
#
# ```julia
# using CUDA
# core = replay(tape, (; N = 1000); T = Float32, backend = CUDABackend())
# ```
#
# The replayed model contains no recorder machinery at all — its evaluation
# path is byte-identical to one built directly against `ExaCore`.
#
# ## What a tape can and cannot capture
#
# A tape freezes the model's *structure*; values and sizes flow through.
# Sizes (`data.N`), ranges (`1:data.N-2`), data arrays used as iterables,
# bounds, and start values are all resolved at replay time. But **control flow
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
# In an app package, record the tape at precompile time and replay in `main`:
#
# ```julia
# module MyModelApp
# using ExaModels, NLPModelsIpoptLite
#
# const TAPE = record(build, (; N = 4))   # runs at precompile time
#
# function (@main)(ARGS)
#     N = parse(Int, ARGS[1])
#     m = ExaModel(replay(TAPE, (; N = N)))
#     result = ipopt(m; print_level = 3)
#     return result.status == 0 ? 0 : 1
# end
# end
# ```
#
# `juliac --trim=safe` then only needs to compile `replay`, the evaluation
# kernels, and the solver — never `build`. See `test/RecorderApp.jl` for a
# complete working example (compiled and run as part of the test suite), and
# `docs/design/recorder.md` in the repository for the design rationale.
