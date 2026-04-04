"""
    precompile_model(m::AbstractExaModel)

Triggers compilation of all GPU kernels for a specific `ExaModel` by performing
a complete dry-run evaluation of every NLP callback (objective, gradient,
constraints, Jacobian, Hessian). After calling this function, subsequent
evaluations will not incur JIT compilation overhead.

Returns `nothing`.

## Example
```julia
using ExaModels, CUDA, KernelAbstractions

c = ExaCore(Float64; backend = CUDABackend())
c, x = add_var(c, 100)
c, _ = add_obj(c, x[i]^2 for i in 1:100)
c, _ = add_con(c, x[i] - 1 for i in 1:100; lcon = 0.0, ucon = Inf)
m = ExaModel(c)

precompile_model(m)  # all kernels now compiled; subsequent calls are fast
```
"""
function precompile_model(m::AbstractExaModel{T,VT,E}) where {T,VT,E}
    nvar = m.meta.nvar
    ncon = m.meta.ncon
    nnzj = m.meta.nnzj
    nnzh = m.meta.nnzh

    x = m.meta.x0
    y = similar(x, ncon)
    fill!(y, one(T))

    # Objective
    obj(m, x)

    # Gradient
    g = similar(x)
    grad!(m, x, g)

    # Constraints
    if ncon > 0
        c = similar(x, ncon)
        cons_nln!(m, x, c)
    end

    # Jacobian
    if nnzj > 0
        jrows = similar(x, Int, nnzj)
        jcols = similar(x, Int, nnzj)
        jac_structure!(m, jrows, jcols)

        jvals = similar(x, nnzj)
        jac_coord!(m, x, jvals)
    end

    # Hessian
    if nnzh > 0
        hrows = similar(x, Int, nnzh)
        hcols = similar(x, Int, nnzh)
        hess_structure!(m, hrows, hcols)

        hvals = similar(x, nnzh)
        if ncon > 0
            hess_coord!(m, x, y, hvals; obj_weight = one(T))
        else
            hess_coord!(m, x, similar(x, 0), hvals; obj_weight = one(T))
        end
    end

    # Synchronize if backend supports it
    _sync_backend(m)

    return nothing
end

# Default no-op; overridden by KernelAbstractions extension
_sync_backend(m) = nothing

"""
    warmup(backend; T = Float64)

Pre-compiles all ExaModels GPU infrastructure kernels for the given `backend`
and element type `T` by building and evaluating a small representative model.

This is useful to pay the JIT compilation cost upfront (e.g., at application
startup) rather than on the first real solve. The function builds a tiny model
with objective, constraints, gradient, Jacobian, and Hessian terms, triggering
compilation of every kernel family used by ExaModels.

Returns `nothing`.

## Example
```julia
using ExaModels, CUDA, KernelAbstractions

warmup(CUDABackend())          # compiles Float64 kernels
warmup(CUDABackend(); T = Float32)  # also compile Float32 variants
```
"""
function warmup(backend; T::Type{<:AbstractFloat} = Float64)
    # Build a small but representative model that exercises all kernel paths.
    # We need: multiple variables, an objective, constraints (both simple and
    # augmented), so that obj/grad/cons/jac/hess kernels all get compiled.
    N = 4

    c = ExaCore(T; backend = backend)

    # Variables
    c, x = add_var(c, N; start = ones(T, N))

    # Objective: sum of x[i]^2  (exercises kerf, kerg, kerh)
    c, _ = add_obj(c, x[i]^2 for i in 1:N)

    # Constraints: x[i] - 1 >= 0  (exercises kerj, kerh2 and constraint kernels)
    c, con = add_con(c, x[i] - 1 for i in 1:N; lcon = zeros(T, N), ucon = fill(T(Inf), N))

    # Constraint augmentation (exercises kerf2, kers, compress_to_dense)
    c, _ = add_con!(c, con, x[i] * x[i] for i in 1:N)

    # Build model (triggers build_extension with sort/pointer/sparsity kernels)
    m = ExaModel(c)

    # Run all NLP callbacks to compile evaluation kernels
    precompile_model(m)

    return nothing
end
