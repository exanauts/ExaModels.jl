module ExaModelsExamples

import ExaModels: ExaModels, NLPModels
import JuMP, NLPModelsJuMP
import PowerModels: PowerModels, silence
import PyCall: @py_str
import MadNLP
import AmplNLReader
import CPUTime: @CPUtime

silence()

include("opf.jl")
include("luksanvlcek.jl")
include("distillation.jl")
include("quadrotor.jl")

function project!(l,x,u; marg = 1e-4)
    map!(x,l,x,u) do l,x,u
        max(l+marg, min(u-marg,x) )
    end
end

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

    project!(m.meta.lvar, x, m.meta.uvar)

    # println("Objective evaluation")
    tobj = @elapsed for t=1:100
        NLPModels.obj(m,x)
    end
    # println("Constraints evaluation")
    tcon = @elapsed for t=1:100
        NLPModels.cons!(m,x,c)
    end
    # println("Gradient evaluation")
    tgrad = @elapsed for t=1:100
        NLPModels.grad!(m,x,g)
    end
    # println("Jacobian evaluation")
    tjac = @elapsed for t=1:100
        NLPModels.jac_coord!(m,x,jac)
    end
    # println("Hessian evaluation")
    thess = @elapsed for t=1:100
        NLPModels.hess_coord!(m,x,y,hess)
    end
    # println("Jacobina sparsity evaluation")
    tjacs = @elapsed for t=1:100
        NLPModels.jac_structure!(m,jrows,jcols)
    end
    # println("Hessian sparsity evaluation")
    thesss = @elapsed for t=1:100
        NLPModels.hess_structure!(m,hrows,hcols)
    end

    return (
        tobj = tobj,
        tcon = tcon,
        tgrad = tgrad,
        tjac = tjac,
        thess = thess,
        tjacs = tjacs,
        thesss = thesss,
    )
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

for name in filter(names(ExaModelsExamples; all=true)) do x
    endswith(string(x), "model")
end
    @eval export $name
end 

end # module ExaModelsExamples
