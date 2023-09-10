module ADTest

using ExaModels
using Random, Test, ForwardDiff, SpecialFunctions

Random.seed!(0)

const FUNCTIONS = [
    ("function-test-1-1",x->beta(erf(x[1]/x[2]/3.0)+3.0*x[2],erf(x[9])^2)), 
    ("function-test-1-2",x->0*x[1]), 
    ("function-test-1-3",x->beta(cos(log(abs2(inv(inv(x[1])))+1.)),erfc(tanh(0*x[1])))), 
    ("function-test-1-4",x->(0*x[1]^x[3]^1.0+x[1])/x[9]/x[10]), 
    ("function-test-1-5",x->exp(x[1]+1.)^x[2]*log(abs2(x[3])+3)/tanh(x[2])), 
    ("function-test-1-6",x->beta(2*logbeta(x[1],x[5]),beta(x[2],x[3]))), 
    ("function-test-1-7",x->besselj0(exp(erf(-x[1])))), 
    ("function-test-1-8",x->erfc(abs2(x[1]^2/x[2])^x[9]/x[10])), 
    ("function-test-1-9",x->erfc(x[1])^erf(2.5x[2])), 
    ("function-test-1-10",x->sin(1/x[1])), 
    ("function-test-1-11",x->exp(x[2])/cos(x[1])^2+sin(x[1]^2)), 
    ("function-test-1-12",x->sin(x[9]inv(x[1])-x[8]inv(x[2]))), 
    ("function-test-1-13",x->x[1]/log(x[2]^2+9.)), 
    ("function-test-1-14",x->beta(beta(tan(beta(x[1],1)+2.0),cos(sin(x[2]))),x[3])), 
    ("function-test-1-15",x->beta(cos(beta(beta(x[1]^9,x[2]),x[2]*x[3])),sin(x[2]*x[3]/2.0)/1.0)),
]

function gradient(f, x)
    T = eltype(x)
    y = fill!(similar(x), zero(T))
    ExaModels.gradient!(y, (p,x)->f(x), x, nothing, one(T))
    return y
end

function sgradient(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(nothing))
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    a1 = unique(y1)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Tuple{Int,Int}, n)

    ExaModels.sgradient!(buffer_I, ff, nothing, nothing, comp, 0, NaN)
    ExaModels.sgradient!(buffer, ff, nothing, x, comp, 0, one(T))
    
    y = zeros(length(x))
    y[collect(i for (i,j) in buffer_I)] += buffer

    return y
end

function sjacobian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(nothing))
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    a1 = unique(y1)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.sjacobian!(buffer_I, buffer_J, ff, nothing, nothing, comp, 0, 0, NaN)
    ExaModels.sjacobian!(buffer, nothing, ff, nothing, x, comp, 0, 0, one(T))
    
    y = zeros(length(x))
    y[buffer_J] += buffer

    return y
end

function shessian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(nothing))
    y2 = []
    ExaModels.hrpass0(t, nothing, y2, nothing, nothing, 0, NaN, NaN)

    a2 = unique(y2)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

    n = length(a2)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.shessian!(buffer_I, buffer_J, ff, nothing, nothing, comp, 0, NaN, NaN)
    ExaModels.shessian!(buffer, nothing, ff, nothing, x, comp, 0, one(T), zero(T))

    y = zeros(length(x),length(x))
    for (k,(i,j)) in enumerate(zip(buffer_I,buffer_J))
        if i== j
            y[i,j] += buffer[k]
        else
            y[i,j] += buffer[k]
            y[j,i] += buffer[k]
        end
    end
    return y
end

function runtests()
    @testset "NLP test" begin
        for (name, f) in FUNCTIONS
            x0 = randn(10)
            @testset "$name" begin
                g = ForwardDiff.gradient(f, x0)
                h = ForwardDiff.hessian(f, x0)
                @test gradient(f, x0) ≈ g atol=1e-6
                @test sgradient(f, x0) ≈ g atol=1e-6
                @test sjacobian(f, x0) ≈ g atol=1e-6
                @test shessian(f, x0) ≈ h atol=1e-6
            end
        end
    end
end

end #module
