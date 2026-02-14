# Issue https://github.com/MadNLP/MadNLP.jl/issues/518

function _exa_trivialmax_model(backend, n)
    c = ExaCore(; minimize=false, backend=backend)
    x = variable(c, n)
    s = constraint(c, x[1]; lcon=0, ucon=1)
    objective(c, x[1]^2)
    return ExaModel(c; prod=true), (x,), (s,)
end
exa_trivialmax_model(backend, n) = _exa_trivialmax_model(backend, n)[1]

function _jump_trivialmax_model(backend, n)
    model = Model()
    @variable(model, x[1:n])
    @constraint(model, s[i=1:n], 0 <= x[i] <= 1)
    @objective(model, Max, x[1]^2)
    return model, (x,), (s,)
end
jump_trivialmax_model(backend, n) = MathOptNLPModel(_jump_trivialmax_model(backend, n)[1])

