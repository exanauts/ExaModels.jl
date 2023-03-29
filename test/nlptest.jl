module TestNLP

using Test

function luksan_vlcek_objective(x,i)
    return 100*(x[i-1]^2-x[i])^2+(x[i-1]-1)^2
end
function luksan_vlcek_constraint(x,i)
    return 3x[i+1]^3+2*x[i+2]-5+sin(x[i+1]-x[i+2])sin(x[i+1]+x[i+2])+4x[i+1]-x[i]exp(x[i]-x[i+1])-3
end
function luksan_vlceck_starting(i)
    return mod(i,2)==1 ? -1.2 : 1.0
end

function luksan_vlcek_simdiff_model(N, device = nothing)
    c = SIMDiff.Core(device)
    x = SIMDiff.variable(c, N; start = (luksan_vleck_starting(i) for i=1:N))
    SIMDiff.constraint(c, luksan_vleck_constraint(x,i) for i in 1:N-2)
    SIMDiff.objective(c, luksan_vlcek_objective(x,i) for i in 2:N)
    return SIMDiff.Model(c; device=device)
end

function luksan_vlceck_adnlp_model
    
end

function test_luksan_vlcek_model()
    m = luksan_vlcek_simdiff_model(3)
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
