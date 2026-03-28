module Rosenrock

using ExaModels

function rosenrock_model(N = 1000; T = Float64, backend = nothing, kwargs ...)
    c = ExaCore(T; backend = backend)
    c, x = add_variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    # c, y  = add_variable(c, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    # c, con = add_constraint(
    #     c,
        # 3 * x[k+1]^3 + 2 * x[k+2] - 5 + sin(x[k+1] - y[k+2]) * sin(x[k+1] + x[k+2]) + 4x[k+1] -
        # x[k] * exp(x[k] - y[k+1]) - 3 for k = 1:N-2
    # )
    # c, o = add_objective(c, 100 * (x[i]^2 - x[i+1])^2 + (y[i] - 1)^2 for i = 1:N-1)
    # @objective(c, 100 * (x[i]^2 - x[i+1])^2 + (y[i] - 1)^2 for i = 1:N-1)
    return ExaModel(c)
end

function @main(ARGS)

    # N = parse(Int, ARGS[1])

    m = rosenrock_model(10)
    
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

