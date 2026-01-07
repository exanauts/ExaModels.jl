using ExaModels, NLPModels
function test()
    m=ExaCore()
    v=variable(m, 5)
    e1=expression(m, (4,), v[i] * v[i+1] for i in 1:4)
    e2=expression(m, (4,), e1[i] + v[i] for i in 1:4)
    c=constraint(m, e2[i] / i for i in 1:4; ucon=10.0)
    o=objective(m, e2[i] for i in 1:4)
    mod = ExaModel(m)
    jac_buffer = zeros(mod.meta.nnzj)
    x = [i for i in 1:mod.meta.nvar]
    jac_coord!(mod, x, jac_buffer)
    @info jac_buffer
    @info length(jac_buffer)
    rows = zeros(Int, mod.meta.nnzj)
    cols = zeros(Int, mod.meta.nnzj)
    jac_structure!(mod, rows, cols)
    @info rows
    @info cols
end
