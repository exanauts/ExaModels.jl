function distillation_column_model(T, S = Array, device = nothing)

    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D  = 0.2
    F  = 0.4
    ybar = .8958
    ubar = 2.0
    alpha= 1.6
    dt = 10/T
    xAf = 0.5
    xA0s = S([(i,0.5) for i in 0:NT+1])

    itr0 = S(collect(Iterators.product(1:T,1:FT-1)))
    itr1 = S(collect(Iterators.product(1:T,FT+1:NT)))
    itr2 = S(collect(Iterators.product(0:T,0:NT+1)))

    c = SIMDiff.Core(S)

    xA  = SIMDiff.variable(c, 0:T, 0:NT+1; start = .5)
    yA  = SIMDiff.variable(c, 0:T, 0:NT+1; start = .5)
    u   = SIMDiff.variable(c, 0:T; start = 1.)
    V   = SIMDiff.variable(c, 0:T; start = 1.)
    L2  = SIMDiff.variable(c, 0:T; start = 1.)

    SIMDiff.objective(c, (yA[t,1] - ybar)^2 for t=0:T)
    SIMDiff.objective(c, ( u[t] - ubar)^2 for t=0:T)

    SIMDiff.constraint(
        c,
        xA[0,i] - xA0 for (i,xA0) in xA0s
            )
    SIMDiff.constraint(
        c,
        (xA[t,0] - xA[t-1,0]) / dt - (1/Ac) * (yA[t,1] - xA[t,0]) for t=1:T)
    SIMDiff.constraint(
        c, (xA[t,i] - xA[t-1,i]) / dt - (1/At) * (u[t] * D * (yA[t,i-1] - xA[t,i]) - V[t] * (yA[t,i] - yA[t,i+1]))
        for (t,i) in itr0)
    SIMDiff.constraint(
        c, (xA[t,FT] - xA[t-1,FT]) / dt - (1/At) * (
            F * xAf + u[t] * D * xA[t,FT-1] - L2[t] * xA[t,FT]
            - V[t] * (yA[t,FT] - yA[t,FT+1])
        )
        for t=1:T)
    SIMDiff.constraint(
        c, (xA[t,i] - xA[t-1,i]) / dt - (1/At) * (
            L2[t] * (yA[t,i-1] - xA[t,i]) - V[t] * (yA[t,i] - yA[t,i+1])
        )
        for (t,i) in itr1)
    SIMDiff.constraint(
        c, (xA[t,NT+1] - xA[t-1,NT+1]) / dt - (1/Ar) * (
            L2[t] * xA[t,NT] - (F - D) * xA[t,NT+1] - V[t] * yA[t,NT+1]
        )
        for t=1:T
            )
    SIMDiff.constraint(c, V[t] - u[t] * D - D for t in 0:T)
    SIMDiff.constraint(c, L2[t]- u[t] * D - F for t in 0:T)
    SIMDiff.constraint(c, yA[t,i] * (1-xA[t,i]) - alpha * xA[t,i] * (1-yA[t,i]) for (t,i) in itr2)

    # Iterators.product
    return SIMDiff.Model(
        c; device = device
    )
end

function jump_distillation_column_model(T)

    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D  = 0.2
    F  = 0.4
    ybar = .8958
    ubar = 2.0
    alpha= 1.6
    dt = 10/T
    xAf = 0.5
    xA0s = Dict(i=>0.5 for i in 0:NT+1)

    itr0 = collect(Iterators.product(1:T,1:FT-1))
    itr1 = collect(Iterators.product(1:T,FT+1:NT))
    itr2 = collect(Iterators.product(0:T,0:NT+1))

    # c = SIMDiff.Core()

    # xA  = SIMDiff.variable(c, 0:T, 0:NT+1; start = .5)
    # yA  = SIMDiff.variable(c, 0:T, 0:NT+1; start = .5)
    # u   = SIMDiff.variable(c, 0:T; start = 1.)
    # V   = SIMDiff.variable(c, 0:T; start = 1.)
    # L2  = SIMDiff.variable(c, 0:T; start = 1.)

    m = JuMP.Model()

    JuMP.@variable(m, xA[0:T,0:NT+1], start = .5)
    JuMP.@variable(m, yA[0:T,0:NT+1], start = .5)
    JuMP.@variable(m, u[0:T], start = 1.)
    JuMP.@variable(m, V[0:T], start = 1.)
    JuMP.@variable(m, L2[0:T], start = 1.)

    JuMP.@objective(m, Min, sum((yA[t,1] - ybar)^2 for t=0:T) + sum( (u[t] - ubar)^2 for t=0:T))

    JuMP.@constraint(m, [i in 0:NT+1], xA[0,i] - xA0s[i] == 0)
    JuMP.@constraint(m, [t=1:T], (xA[t,0] - xA[t-1,0]) / dt - (1/Ac) * (yA[t,1] - xA[t,0]) == 0)
    JuMP.@constraint(
        m, [t=1:T, i=1:FT-1],
        (xA[t,i] - xA[t-1,i]) / dt
        - (1/At) * (u[t] * D * (yA[t,i-1] - xA[t,i]) - V[t] * (yA[t,i] - yA[t,i+1])) == 0
    )
    JuMP.@constraint(
        m, [t=1:T],
        (xA[t,FT] - xA[t-1,FT]) / dt - (1/At) * (
            F * xAf + u[t] * D * xA[t,FT-1] - L2[t] * xA[t,FT]
            - V[t] * (yA[t,FT] - yA[t,FT+1])
        ) == 0
    )
    JuMP.@constraint(
        m, [t=1:T, i=FT+1:NT],
        (xA[t,i] - xA[t-1,i]) / dt - (1/At) * (
            L2[t] * (yA[t,i-1] - xA[t,i]) - V[t] * (yA[t,i] - yA[t,i+1])
        ) == 0
    )
    JuMP.@constraint(
        m, [t=1:T],
        (xA[t,NT+1] - xA[t-1,NT+1]) / dt - (1/Ar) * (
            L2[t] * xA[t,NT] - (F - D) * xA[t,NT+1] - V[t] * yA[t,NT+1]
        ) == 0
    )
    JuMP.@constraint(m, [t in 0:T], V[t] - u[t] * D - D == 0)
    JuMP.@constraint(m, [t in 0:T], L2[t]- u[t] * D - F == 0)
    JuMP.@constraint(
        m, [t in 0:T, i in 0:NT+1],
        yA[t,i] * (1-xA[t,i]) - alpha * xA[t,i] * (1-yA[t,i]) == 0
    )

    return m
end
