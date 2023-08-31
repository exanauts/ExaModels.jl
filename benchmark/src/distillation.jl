function exa_distillation_column_model(N= 3; kwargs...)
    return ExaModelsExamples.distillation_column_model(N; kwargs...)
end

function jump_distillation_column_model(T = 3)

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
    dt = 10 / T
    xAf = 0.5
    xA0s = Dict(i => 0.5 for i = 0:NT+1)

    itr0 = collect(Iterators.product(1:T, 1:FT-1))
    itr1 = collect(Iterators.product(1:T, FT+1:NT))
    itr2 = collect(Iterators.product(0:T, 0:NT+1))

    m = JuMP.Model()

    JuMP.@variable(m, xA[0:T, 0:NT+1], start = 0.5)
    JuMP.@variable(m, yA[0:T, 0:NT+1], start = 0.5)
    JuMP.@variable(m, u[0:T], start = 1.0)
    JuMP.@variable(m, V[0:T], start = 1.0)
    JuMP.@variable(m, L2[0:T], start = 1.0)

    JuMP.@objective(
        m,
        Min,
        sum((yA[t, 1] - ybar)^2 for t = 0:T) + sum((u[t] - ubar)^2 for t = 0:T)
    )

    JuMP.@constraint(m, [i in 0:NT+1], xA[0, i] - xA0s[i] == 0)
    JuMP.@constraint(
        m,
        [t = 1:T],
        (xA[t, 0] - xA[t-1, 0]) / dt - (1 / Ac) * (yA[t, 1] - xA[t, 0]) == 0
    )
    JuMP.@constraint(
        m,
        [t = 1:T, i = 1:FT-1],
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (u[t] * D * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) ==
        0
    )
    JuMP.@constraint(
        m,
        [t = 1:T],
        (xA[t, FT] - xA[t-1, FT]) / dt -
        (1 / At) * (
            F * xAf + u[t] * D * xA[t, FT-1] - L2[t] * xA[t, FT] -
            V[t] * (yA[t, FT] - yA[t, FT+1])
        ) == 0
    )
    JuMP.@constraint(
        m,
        [t = 1:T, i = FT+1:NT],
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (L2[t] * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) == 0
    )
    JuMP.@constraint(
        m,
        [t = 1:T],
        (xA[t, NT+1] - xA[t-1, NT+1]) / dt -
        (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT+1] - V[t] * yA[t, NT+1]) == 0
    )
    JuMP.@constraint(m, [t in 0:T], V[t] - u[t] * D - D == 0)
    JuMP.@constraint(m, [t in 0:T], L2[t] - u[t] * D - F == 0)
    JuMP.@constraint(
        m,
        [t in 0:T, i in 0:NT+1],
        yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) == 0
    )

    return MathOptNLPModel(m)
end

function ampl_distillation_column_model(T = 3)
    nlfile = joinpath(TMPDIR, "distillation_column_model_$T.nl")

    if !isfile(nlfile)
        @info "Writing nlfile for N = $T"
        py"""
        from pyomo.environ import *

        # Create a ConcreteModel
        m = ConcreteModel()

        # Constants
        T = $T
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
        xA0s = {i: 0.5 for i in range(NT+2)}  

        # Define the decision variables
        m.xA = Var(range(T+1), range(NT+2), initialize=0.5)
        m.yA = Var(range(T+1), range(NT+2), initialize=0.5)
        m.u = Var(range(T+1), initialize=1.0)
        m.V = Var(range(T+1), initialize=1.0)
        m.L2 = Var(range(T+1), initialize=1.0)

        # Define the objective function
        m.obj = Objective(
            expr=sum((m.yA[t, 1] - ybar)**2 for t in range(T+1)) +
                 sum((m.u[t] - ubar)**2 for t in range(T+1)),
            sense=minimize
        )

        # Define the constraints
        m.constr1 = ConstraintList()
        for i in range(NT+2):
            m.constr1.add(expr=m.xA[0, i] - xA0s[i] == 0)

        m.constr2 = ConstraintList()
        for t in range(1, T+1):
            m.constr2.add(
                expr=(m.xA[t, 0] - m.xA[t-1, 0]) / dt - (1/Ac) * (m.yA[t, 1] - m.xA[t, 0]) == 0
            )

        m.constr3 = ConstraintList()
        for t in range(1, T+1):
            for i in range(1, FT):
                m.constr3.add(
                    expr=(
                        (m.xA[t, i] - m.xA[t-1, i]) / dt -
                        (1/At) * (
                            m.u[t] * D * (m.yA[t, i-1] - m.xA[t, i]) - m.V[t] * (m.yA[t, i] - m.yA[t, i+1])
                        ) == 0
                    )
                )
                
        m.constr4 = ConstraintList()
        for t in range(1, T+1):
            m.constr4.add(
                expr=(
                    (m.xA[t, FT] - m.xA[t-1, FT]) / dt -
                    (1/At) * (
                        F * xAf + m.u[t] * D * m.xA[t, FT-1] - m.L2[t] * m.xA[t, FT]
                        - m.V[t] * (m.yA[t, FT] - m.yA[t, FT+1])
                    ) == 0
                )
            )

        m.constr5 = ConstraintList()
        for t in range(1, T+1):
            for i in range(FT+1, NT+1):
                m.constr5.add(
                    expr=(
                        (m.xA[t, i] - m.xA[t-1, i]) / dt -
                        (1/At) * (
                            m.L2[t] * (m.yA[t, i-1] - m.xA[t, i]) - m.V[t] * (m.yA[t, i] - m.yA[t, i+1])
                        ) == 0
                    )
                )

        m.constr6 = ConstraintList()
        for t in range(1, T+1):
            m.constr6.add(
                expr=(
                    (m.xA[t, NT+1] - m.xA[t-1, NT+1]) / dt -
                    (1/Ar) * (
                        m.L2[t] * m.xA[t, NT] - (F - D) * m.xA[t, NT+1] - m.V[t] * m.yA[t, NT+1]
                    ) == 0
                )
            )

        m.constr7 = ConstraintList()
        for t in range(T+1):
            m.constr7.add(expr=m.V[t] - m.u[t] * D - D == 0)

        m.constr8 = ConstraintList()
        for t in range(T+1):
            m.constr8.add(expr=m.L2[t] - m.u[t] * D - F == 0)

        m.constr9 = ConstraintList()
        for t in range(T+1):
            for i in range(NT+2):
                m.constr9.add(expr=m.yA[t, i] * (1 - m.xA[t, i]) - alpha * m.xA[t, i] * (1 - m.yA[t, i]) == 0)
        m.write($nlfile)
        """
    end
    return AmplNLReader.AmplModel(nlfile)
end
