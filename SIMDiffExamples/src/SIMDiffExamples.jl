module SIMDiffExamples

import SIMDiff: SIMDiff, NLPModels
import PowerModels: PowerModels, silence
using JuMP, NLPModelsJuMP

include("adbenchmarkmodel.jl")
include("opf.jl")
include("luksanvlcek.jl")
include("distillation.jl")
include("quadrotor.jl")


function compile_callbacks(m)
    nvar = m.meta.nvar
    ncon = m.meta.ncon
    nnzj = m.meta.nnzj
    nnzh = m.meta.nnzh
    
    x = similar(m.meta.x0, nvar)
    y = similar(m.meta.x0, ncon)
    c = similar(m.meta.x0, ncon)
    g = similar(m.meta.x0, nvar)
    jac = similar(m.meta.x0, nnzj)
    hess = similar(m.meta.x0, nnzh)
    jrows = similar(m.meta.x0, Int, nnzj)
    jcols = similar(m.meta.x0, Int, nnzj)
    hrows = similar(m.meta.x0, Int, nnzh)
    hcols = similar(m.meta.x0, Int, nnzh)

    println("Objective evaluation")
    @time NLPModels.obj(m,x)
    println("Constraints evaluation")
    @time NLPModels.cons!(m,x,c)
    println("Gradient evaluation")
    @time NLPModels.grad!(m,x,g)
    println("Jacobian evaluation")
    @time NLPModels.jac_coord!(m,x,jac)
    println("Hessian evaluation")
    @time NLPModels.hess_coord!(m,x,y,hess)
    println("Jacobina sparsity evaluation")
    @time NLPModels.jac_structure!(m,jrows,jcols)
    println("Hessian sparsity evaluation")
    @time NLPModels.hess_structure!(m,hrows,hcols)

    return
end


function parse_log(file)
    open(file) do f
        t1=nothing
        t2=nothing
        while !eof(f)
            s = readline(f)
            if occursin("Total CPU secs in NLP function evaluations", s)
                t1 = parse(Float64,split(s, "=")[2])
            elseif occursin("Total CPU secs in IPOPT (w/o function evaluations)", s)
                t2 = parse(Float64,split(s, "=")[2])
            end
        end
        return t1,t2
    end
end

end # module SIMDiffExamples
