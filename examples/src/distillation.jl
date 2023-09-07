function distillation_column_model(N = 3; T = Float64, backend = nothing, kwargs...)
    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D = 0.2
    F = 0.4
    ybar = 0.8958
    ubar = 2.0
    alpha = 1.6
    dt = 10 / N
    xAf = 0.5
    xA0s = ExaModels.convert_array([(i, 0.5) for i = 0:NT+1], backend)

    itr0 = ExaModels.convert_array(collect(Iterators.product(1:N, 1:FT-1)), backend)
    itr1 = ExaModels.convert_array(collect(Iterators.product(1:N, FT+1:NT)), backend)
    itr2 = ExaModels.convert_array(collect(Iterators.product(0:N, 0:NT+1)), backend)

    c = ExaModels.ExaCore(T, backend)

    xA = ExaModels.variable(c, 0:N, 0:NT+1; start = 0.5)
    yA = ExaModels.variable(c, 0:N, 0:NT+1; start = 0.5)
    u = ExaModels.variable(c, 0:N; start = 1.0)
    V = ExaModels.variable(c, 0:N; start = 1.0)
    L2 = ExaModels.variable(c, 0:N; start = 1.0)

    ExaModels.objective(c, (yA[t, 1] - ybar)^2 for t = 0:N)
    ExaModels.objective(c, (u[t] - ubar)^2 for t = 0:N)

    ExaModels.constraint(c, xA[0, i] - xA0 for (i, xA0) in xA0s)
    ExaModels.constraint(
        c,
        (xA[t, 0] - xA[t-1, 0]) / dt - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t = 1:N
    )
    ExaModels.constraint(
        c,
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (u[t] * D * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) for
        (t, i) in itr0
    )
    ExaModels.constraint(
        c,
        (xA[t, FT] - xA[t-1, FT]) / dt -
        (1 / At) * (
            F * xAf + u[t] * D * xA[t, FT-1] - L2[t] * xA[t, FT] -
            V[t] * (yA[t, FT] - yA[t, FT+1])
        ) for t = 1:N
    )
    ExaModels.constraint(
        c,
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (L2[t] * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) for
        (t, i) in itr1
    )
    ExaModels.constraint(
        c,
        (xA[t, NT+1] - xA[t-1, NT+1]) / dt -
        (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT+1] - V[t] * yA[t, NT+1]) for
        t = 1:N
    )
    ExaModels.constraint(c, V[t] - u[t] * D - D for t = 0:N)
    ExaModels.constraint(c, L2[t] - u[t] * D - F for t = 0:N)
    ExaModels.constraint(
        c,
        yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr2
    )

    return ExaModels.ExaModel(c; kwargs...)
end
