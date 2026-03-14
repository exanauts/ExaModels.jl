using ExaModels, NLPModelsIpopt

function luksan_vlcek_model(N = 3; T = Float64, backend = nothing, kwargs...)

    c = ExaModels.ExaCore(T; backend = backend)
    x = ExaModels.variable(c, N, 3; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N, j=1:3))
    c1 = ExaModels.constraint(
        c,
        3x[i+1,j]^3 + 2 * x[i+2,j] - 5 + 4x[i+1,j] -
        x[i,j]exp(x[i,j] - x[i+1,j]) - 3 for i = 1:N-2, j=1:3
            )
    ExaModels.constraint!(
        c, c1,
        (i,j) => sin(x[i+1,j] - x[i+2,j])sin(x[i+1,j] + x[i+2,j]) for i=1:N-2, j=1:3
    )
    ExaModels.objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N, j=1:3)
    return ExaModels.ExaModel(c; kwargs...)
end

m = luksan_vlcek_model(10)
ipopt(m; linear_solver="ma27")
