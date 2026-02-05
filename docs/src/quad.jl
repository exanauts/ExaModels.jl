# # [Example: Quadrotor](@id quad)

function quadrotor_model(N = 3; backend = nothing)

    n = 9
    p = 4
    d(i, j, N) =
        (j == 1 ? 1 * sin(2 * pi / N * i) : 0.0) +
        (j == 3 ? 2 * sin(4 * pi / N * i) : 0.0) +
        (j == 5 ? 2 * i / N : 0.0)
    dt = 1 / N
    R = fill(1 / 10, 4)
    Q = [1, 0, 1, 0, 1, 0, 1, 1, 1]
    Qf = [1, 0, 1, 0, 1, 0, 1, 1, 1] / dt

    x0s = [(i, 0.0) for i in 1:n]
    itr0 = [(i, j, R[j]) for (i, j) in Base.product(1:N, 1:p)]
    itr1 = [(i, j, Q[j], d(i, j, N)) for (i, j) in Base.product(1:N, 1:n)]
    itr2 = [(j, Qf[j], d(N + 1, j, N)) for j in 1:n]

    c = ExaCore(; backend = backend)

    x = variable(c, 1:(N + 1), 1:n)
    u = variable(c, 1:N, 1:p)

    constraint(c, x[1, i] - x0 for (i, x0) in x0s)
    constraint(c, -x[i + 1, 1] + x[i, 1] + (x[i, 2]) * dt for i in 1:N)
    constraint(
        c,
        -x[i + 1, 2] +
            x[i, 2] +
            (
                u[i, 1] * cos(x[i, 7]) * sin(x[i, 8]) * cos(x[i, 9]) +
                u[i, 1] * sin(x[i, 7]) * sin(x[i, 9])
            ) * dt for i in 1:N
    )
    constraint(c, -x[i + 1, 3] + x[i, 3] + (x[i, 4]) * dt for i in 1:N)
    constraint(
        c,
        -x[i + 1, 4] +
            x[i, 4] +
            (
                u[i, 1] * cos(x[i, 7]) * sin(x[i, 8]) * sin(x[i, 9]) -
                u[i, 1] * sin(x[i, 7]) * cos(x[i, 9])
            ) * dt for i in 1:N
    )
    constraint(c, -x[i + 1, 5] + x[i, 5] + (x[i, 6]) * dt for i in 1:N)
    constraint(
        c,
        -x[i + 1, 6] + x[i, 6] + (u[i, 1] * cos(x[i, 7]) * cos(x[i, 8]) - 9.8) * dt for
            i in 1:N
    )
    constraint(
        c,
        -x[i + 1, 7] +
            x[i, 7] +
            (u[i, 2] * cos(x[i, 7]) / cos(x[i, 8]) + u[i, 3] * sin(x[i, 7]) / cos(x[i, 8])) * dt
            for i in 1:N
    )
    constraint(
        c,
        -x[i + 1, 8] + x[i, 8] + (-u[i, 2] * sin(x[i, 7]) + u[i, 3] * cos(x[i, 7])) * dt for
            i in 1:N
    )
    constraint(
        c,
        -x[i + 1, 9] +
            x[i, 9] +
            (
                u[i, 2] * cos(x[i, 7]) * tan(x[i, 8]) +
                u[i, 3] * sin(x[i, 7]) * tan(x[i, 8]) +
                u[i, 4]
            ) * dt for i in 1:N
    )

    objective(c, 0.5 * R * (u[i, j]^2) for (i, j, R) in itr0)
    objective(c, 0.5 * Q * (x[i, j] - d)^2 for (i, j, Q, d) in itr1)
    objective(c, 0.5 * Qf * (x[N + 1, j] - d)^2 for (j, Qf, d) in itr2)

    return m = ExaModel(c)

end

#-

using ExaModels, NLPModelsIpopt

m = quadrotor_model(100)
result = ipopt(m)
