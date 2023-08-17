function exa_luksan_vlcek_model(N = 3; kwargs...)
    return ADBenchmarkModel(
        ExaModelsExamples.luksan_vlcek_model(N; kwargs...)
    )
end

function jump_luksan_vlcek_model(N = 3)
    jm = JuMP.Model()

    JuMP.@variable(jm, x[i = 1:N], start = mod(i, 2) == 1 ? -1.2 : 1.0)
    JuMP.@NLconstraint(
        jm,
        [i = 1:N-2],
        3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
        x[i]exp(x[i] - x[i+1]) - 3 == 0.0
    )
    JuMP.@NLobjective(jm, Min, sum(100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N))

    return ADBenchmarkModel(MathOptNLPModel(jm))
end


function ampl_luksan_vlcek_model(N = 3)
    nlfile = joinpath(TMPDIR, "luksan_vlcek_model_$N.nl")

    if !isfile(nlfile)
        @info "Writing nlfile for N = $N"
        py"""
        N = $N

        import pyomo.environ as pyo

        model = pyo.ConcreteModel()

        model.x = pyo.Var(range(1,N+1), initialize=lambda model, i: -1.2 if i % 2 == 1 else 1.0)

        def luksan_constraint_rule(model, i):
            return (3*model.x[i+1]**3 + 2*model.x[i+2] - 5 
                    + pyo.sin(model.x[i+1]-model.x[i+2])*pyo.sin(model.x[i+1]+model.x[i+2])
                    + 4*model.x[i+1] - model.x[i]*pyo.exp(model.x[i]-model.x[i+1]) - 3 == 0)
        def luksan_objective_rule(model):
            return sum(100*(model.x[i-1]**2-model.x[i])**2+(model.x[i-1]-1)**2 for i in range(2,N+1))


        model.constraint = pyo.Constraint(range(1,N-1), rule=luksan_constraint_rule)
        model.objective  = pyo.Objective(rule=luksan_objective_rule, sense=pyo.minimize)
        model.write($nlfile)
        """
    end

    return ADBenchmarkModel(AmplNLReader.AmplModel(nlfile))
end
