function luksan_vlcek_obj(x, i)
    return 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2
end

function luksan_vlcek_con(x, i)
    return 3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
           x[i]exp(x[i] - x[i+1]) - 3
end

function luksan_vlcek_x0(i)
    return mod(i, 2) == 1 ? -1.2 : 1.0
end

function exa_luksan_vlcek_model(backend, N)

    c = ExaCore(backend)
    x = variable(c, N; start = (luksan_vlcek_x0(i) for i = 1:N))
    constraint(c, luksan_vlcek_con(x, i) for i = 1:N-2)
    objective(c, luksan_vlcek_obj(x, i) for i = 2:N)

    return ExaModel(c; prod = true)
end

function jump_luksan_vlcek_model(backend, N)
    jm = JuMP.Model()

    JuMP.@variable(jm, x[i = 1:N], start = mod(i, 2) == 1 ? -1.2 : 1.0)
    JuMP.@NLconstraint(
        jm,
        [i = 1:N-2],
        3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
        x[i]exp(x[i] - x[i+1]) - 3 == 0.0
    )
    JuMP.@NLobjective(jm, Min, sum(100(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N))

    return MathOptNLPModel(jm)
end
