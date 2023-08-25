function luksan_vlcek_model(N = 3; T = Float64, backend=nothing)

    c = ExaModels.ExaCore(T, backend)
    x = ExaModels.variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    ExaModels.constraint(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
        x[i]exp(x[i] - x[i+1]) - 3 for i = 1:N-2
    )
    ExaModels.objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
    return ExaModels.ExaModel(c)
end

