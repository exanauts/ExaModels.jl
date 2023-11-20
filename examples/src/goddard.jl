#=
    Goddard problem.

    Model the ascent of a rocket through the atmosphere.
    This example is taken from:
    https://ct.gitlabpages.inria.fr/gallery/goddard-j/goddard.html

=#

function goddard_model(
    N::Int;
    T=Float64,
    backend=nothing,
    kwargs...,
)
    # Parameters
    Cd = 310.0
    Tmax = 3.5
    β = 500.0
    b = 2.0
    t0 = 0.0
    r0 = 1.0
    v0 = 0.0
    vmax = 0.1
    m0 = 1.0
    x0 = [ r0, v0, m0 ]
    mf = 0.6
    n, p = 3, 1

    core = ExaModels.ExaCore(T, backend)
    # Intiial position
    x0s = ExaModels.convert_array([(i, x0[i]) for i in 1:3] , backend)

    xl = repeat([r0, 0.0, mf]', N+1)
    xu = repeat([Inf, vmax, m0]', N+1)

    dt = ExaModels.variable(core, 1; lvar=0.0)
    x = ExaModels.variable(core, 1:N+1, 1:n; lvar=xl, uvar=xu)
    u = ExaModels.variable(core, 1:N; lvar=0.0, uvar=1.0)

    # Initial constraint.
    ExaModels.constraint(core, x[1, i] - x0_ for (i, x0_) in x0s)
    # Dynamics
    ExaModels.constraint(
        core,
        -x[t+1, 1] + x[t, 1] + x[t, 2] * dt[1] for t in 1:N
    )
    ExaModels.constraint(
        core,
        -x[t+1, 2] + x[t, 2] + ((Tmax * u[t] - (Cd * x[t, 2]^2 * exp(-β * (x[t,1] - 1.0))))/ x[t, 3] - 1 / x[t, 1]^2) * dt[1] for t in 1:N
    )
    ExaModels.constraint(
        core,
        -x[t+1, 3] + x[t, 3] - (b * Tmax * u[t]) * dt[1] for t in 1:N
    )

    # Objective (minus sign as problem is to maximize final radius)
    o = ExaModels.objective(
        core,
        -x[t, 1] for t in (N+1,)
    )

    return ExaModels.ExaModel(core, kwargs...)
end

