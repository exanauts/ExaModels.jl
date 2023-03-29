module TestNLP

function luksan_vlcek_model(
    N,
    S = Array,
    device = nothing,
    )
    c = SIMDiff.Core(S)
    x = SIMDiff.variable(
        c, N;
        start = S([mod(i,2)==1 ? -1.2 : 1. for i=1:N])
    )
    SIMDiff.constraint(
        c,
        3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3
        for i in 1:N-2)
    SIMDiff.objective(c, 100*(x[i-1]^2-x[i])^2+(x[i-1]-1)^2 for i in 2:N)
    return SIMDiff.Model(c; device=device)
end

function test_luksan_vlcek_model()
    m = luksan_vlcek_model(3)
    ipopt(m)
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

end # TestNLP

TestNLP.runtests()
