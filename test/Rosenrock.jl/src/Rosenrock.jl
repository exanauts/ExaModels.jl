module Rosenrock

using ExaModels

function @main(ARGS)

    N = parse(Int, ARGS[1])
    core, x = ExaCore(
        varspec(N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    )
    
    m, c, o = ExaModel(
        core,
        conspec(
            3 * x[k+1]^3 + 2 * x[k+2] - 5 + sin(x[k+1] - x[k+2]) * sin(x[k+1] + x[k+2]) + 4x[k+1] -
                x[k] * exp(x[k] - x[k+1]) - 3 for k = 1:N-2),
        objspec(100 * (x[i]^2 - x[i+1])^2 + (x[i] - 1)^2 for i = 1:N-1)
    )
    
    xbuffer = randn(m.meta.nvar)
    ybuffer = randn(m.meta.ncon)
    jbuffer = randn(m.meta.nnzj)
    hbuffer = randn(m.meta.nnzh)

    ExaModels.obj(m, m.meta.x0)
    ExaModels.cons!(m, m.meta.x0, ybuffer)
    ExaModels.grad!(m, m.meta.x0, xbuffer)
    ExaModels.jac_coord!(m, m.meta.x0, jbuffer)
    ExaModels.hess_coord!(m, m.meta.x0, m.meta.y0, hbuffer; obj_weight=1.0)

    println(Core.stdout, "Model runs successfully! sum(hess) = ", sum(hbuffer))

    return 0
end

end # module Rosenrock

