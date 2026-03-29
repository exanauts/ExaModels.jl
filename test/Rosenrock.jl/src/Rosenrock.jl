module Rosenrock

using LinearAlgebra
using ExaModels
using Ipopt_jll: libipopt
using OpenBLAS32_jll: libopenblas_path

function __init__()
    # Julia's libblastrampoline defaults to ILP64 (dscal_64_, dcopy_64_, ...).
    # Ipopt and MUMPS call LP64 symbols (dscal_, dcopy_, ...).
    # Forward the LP64 OpenBLAS that ships with Ipopt_jll so those symbols resolve.
    LinearAlgebra.BLAS.lbt_forward(libopenblas_path)
end

@inline function rosenrock_model(N = 1000; T = Float64, kwargs...)
    c = ExaCore(T; kwargs...)
    @var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    @var(c, y, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    @con(
        c,
        3 * x[k+1]^3 + 2 * x[k+2] - 5 + sin(x[k+1] - y[k+2]) * sin(x[k+1] + x[k+2]) + 4x[k+1] -
        x[k] * exp(x[k] - y[k+1]) - 3 for k = 1:N-2
    )
    @obj(c, obj, 100 * (x[i]^2 - x[i+1])^2 + (y[i] - 1)^2 for i = 1:N-1)
    m = ExaModel(c)
end

# Compute the concrete ExaModel type produced by rosenrock_model.
# ipopt.jl uses this so --trim=safe can resolve all NLP dispatch statically.
# const _IpoptModelType = typeof(rosenrock_model(1))

include("ipopt.jl")

function @main(ARGS)
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
    libipopt = length(ARGS) >= 2 ? ARGS[2] : "libipopt"

    m = rosenrock_model(N)

    println(Core.stdout, "Solving Rosenrock N=$N with Ipopt ($(libipopt))...")

    result = solve_with_ipopt(m; print_level = 5)

    println(Core.stdout, "Ipopt status : ", result.status)

    return result.status == 0 ? 0 : 1
end

end # module Rosenrock

