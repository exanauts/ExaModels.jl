module OracleApp

using ExaModels, NLPModelsIpoptLite

# Hand-written oracle callbacks.  Named functions (not anonymous closures) are
# required for AOT compilation with juliac --trim=safe.

function _oracle_f!(cv::AbstractVector, xv::AbstractVector)
    cv[1] = xv[3] - xv[4]
    cv[2] = xv[3]^2 + xv[4]^2
    return nothing
end

function _oracle_jac!(vv::AbstractVector, xv::AbstractVector)
    vv[1] = 1.0          # d(x3 - x4)/dx3
    vv[2] = -1.0         # d(x3 - x4)/dx4
    vv[3] = 2.0 * xv[3] # d(x3^2 + x4^2)/dx3
    vv[4] = 2.0 * xv[4] # d(x3^2 + x4^2)/dx4
    return nothing
end

function _oracle_hess!(hv::AbstractVector, xv::AbstractVector, yv::AbstractVector)
    hv[1] = 2.0 * yv[2]  # d²(x3^2+x4^2)/dx3² * y2
    hv[2] = 2.0 * yv[2]  # d²(x3^2+x4^2)/dx4² * y2
    return nothing
end

# Build the test NLP:
#   min  x1^2 + x2^2 + x3^2 + x4^2
#   s.t. x1 + x2 = 1          (SIMD constraint)
#        x3 - x4 = 0          (oracle, linear)
#        x3^2 + x4^2 = 0.5   (oracle, nonlinear)
#
# Optimal solution: x1=x2=0.5, x3=x4=0.5, obj=1.0.
@inline function _build_oracle_model(::Type{T} = Float64) where {T}
    c0 = ExaCore(T; concrete = Val(true))
    c1, x = add_var(c0, 4; lvar = T(-10), uvar = T(10), start = T[1, 1, 1, 1])
    c2, _ = add_obj(c1, x[i]^2 for i in 1:4)
    c3, _ = add_con(c2, x[1] + x[2]; lcon = T(1), ucon = T(1))

    oracle = VectorNonlinearOracle(
        nvar    = 4, ncon = 2,
        nnzj    = 4, nnzh = 2,
        jac_rows = [1, 1, 2, 2], jac_cols = [3, 4, 3, 4],
        hess_rows = [3, 4], hess_cols = [3, 4],
        lcon    = T[0, T(1)/2], ucon = T[0, T(1)/2],
        adapt   = Val(true),
        f!      = _oracle_f!,
        jac!    = _oracle_jac!,
        hess!   = _oracle_hess!,
    )
    c4 = constraint(c3, oracle)

    return ExaModel(c4)
end

function (@main)(ARGS)
    println(Core.stdout, "Building oracle NLP with VectorNonlinearOracle (hand-written callbacks)...")
    m = _build_oracle_model(Float64)
    result = ipopt(m; print_level = 5)
    println(Core.stdout, "Ipopt status : ", result.status)
    return result.status == 0 ? 0 : 1
end

end # module OracleApp
