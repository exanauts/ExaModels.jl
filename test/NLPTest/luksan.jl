function luksan_vlcek_obj(x,i)
    return 100*(x[i-1]^2-x[i])^2+(x[i-1]-1)^2
end

function luksan_vlcek_con(x,i)
    return 3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3
end

function luksan_vlcek_x0(i)
    return mod(i,2)==1 ? -1.2 : 1.0
end

function luksan_vlcek_adnlp_model(backend, N)
    return ADNLPModel(
        x->sum(luksan_vlcek_obj(x,i) for i=2:N),
        [luksan_vlcek_x0(i) for i=1:N],
        fill(-Inf,N),
        fill(Inf,N),
        x->[luksan_vlcek_con(x,i) for i=1:N-2],
        zeros(N-2),
        zeros(N-2)
    ) 
end

function luksan_vlcek_simdiff_model(backend, N)
    
    c = SIMDiff.Core(backend)
    x = SIMDiff.variable(
        c, N;
        start = (luksan_vlcek_x0(i) for i=1:N)
    )
    SIMDiff.constraint(
        c,
        luksan_vlcek_con(x,i)
        for i in 1:N-2)
    SIMDiff.objective(c, luksan_vlcek_obj(x,i) for i in 2:N)
    return SIMDiff.Model(c)
end
