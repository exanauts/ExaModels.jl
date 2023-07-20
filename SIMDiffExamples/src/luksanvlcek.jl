function luksan_vlcek_model(N, backend = nothing)
    
    c = SIMDiff.Core(backend)
    x = SIMDiff.variable(
        c, N;
        start = (mod(i,2)==1 ? -1.2 : 1. for i=1:N)
    )
    SIMDiff.constraint(
        c,
        3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3
        for i in 1:N-2)
    SIMDiff.objective(c, 100*(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i in 2:N)
    return SIMDiff.Model(c)
end

function jump_luksan_vlcek_model(N)
    jm=JuMP.Model()
    JuMP.@variable(jm,x[i=1:N], start= mod(i,2)==1 ? -1.2 : 1.)
    JuMP.@NLconstraint(jm,[i=1:N-2], 3x[i+1]^3+2x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3==0.)
    JuMP.@NLobjective(jm,Min,sum(100(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i=2:N))
    return jm
end
