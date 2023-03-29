struct WrapperModel{T,VT,T2,VI2,VT2,I <: NLPModels.AbstractNLPModel{T2,VT2}} <: NLPModels.AbstractNLPModel{T,VT}
    inner::I
    
    jrows::VI2
    jcols::VI2
    hrows::VI2
    hcols::VI2
    
    x::VT2
    y::VT2
    
    con::VT2
    grad::VT2
    jac::VT2
    hess::VT2
    
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters 
end


"""
WrapperModel(m)

Construct a WrapperModel (a subtype of `NLPModels.AbstractNLPModel{T,Vector{T}}`) from a generic NLP Model.

WrapperModel can be used to interface GPU-accelerated NLP models with solvers runing on CPUs.
"""
function WrapperModel(m::NLPModels.AbstractNLPModel)    
    return WrapperModel(
        m,
        similar(m.meta.x0, Int, m.meta.nnzj),
        similar(m.meta.x0, Int, m.meta.nnzj),
        similar(m.meta.x0, Int, m.meta.nnzh),
        similar(m.meta.x0, Int, m.meta.nnzh),
        similar(m.meta.x0, m.meta.nvar),
        similar(m.meta.x0, m.meta.ncon),
        similar(m.meta.x0, m.meta.ncon),
        similar(m.meta.x0, m.meta.nvar),
        similar(m.meta.x0, m.meta.nnzj),
        similar(m.meta.x0, m.meta.nnzh),
        NLPModels.NLPModelMeta(
            m.meta.nvar,
            x0 = Array(m.meta.x0),
            lvar = Array(m.meta.lvar),
            uvar = Array(m.meta.uvar),
            ncon = m.meta.ncon,
            y0 = Array(m.meta.y0),
            lcon = Array(m.meta.lcon),
            ucon = Array(m.meta.ucon),
            nnzj = m.meta.nnzj,
            nnzh = m.meta.nnzh,
            minimize = true
        ),
        NLPModels.Counters()
    )
end

function NLPModels.jac_structure!(
    m::M,
    rows::V,
    cols::V
    ) where {M <: WrapperModel, V <: AbstractVector}

    NLPModels.jac_structure!(m.inner, m.jrows, m.jcols)
    copyto!(rows, m.jrows)
    copyto!(cols, m.jcols)
end

function NLPModels.hess_structure!(
    m::M,
    rows::V,
    cols::V
    ) where {M <: WrapperModel, V <: AbstractVector}

    NLPModels.hess_structure!(m.inner, m.hrows, m.hcols)
    copyto!(rows, m.hrows)
    copyto!(cols, m.hcols)
end

function NLPModels.obj(
    m::M,
    x::V
    ) where {M <: WrapperModel, V <: AbstractVector}

    copyto!(m.x, x)
    return NLPModels.obj(m.inner, m.x)
end
function NLPModels.cons!(
    m::M,
    x::V,
    g::V
    ) where {M <: WrapperModel, V <: AbstractVector}

    copyto!(m.x, x) 
    NLPModels.cons!(m.inner, m.x, m.con)
    copyto!(g, m.con)
    return 
end
function NLPModels.grad!(
    m::M,
    x::V,
    f::V
    ) where {M <: WrapperModel, V <: AbstractVector}

    copyto!(m.x, x)
    NLPModels.grad!(m.inner, m.x, m.grad)
    copyto!(f, m.grad)
    return
end
function NLPModels.jac_coord!(
    m::M,
    x::V,
    jac::V
    ) where {M <: WrapperModel, V <: AbstractVector}

    copyto!(m.x, x)    
    NLPModels.jac_coord!(m.inner, m.x, m.jac)
    copyto!(jac, m.jac)
    return
end
function NLPModels.hess_coord!(
    m::M,
    x::V,
    y::V,
    hess::V;
    obj_weight = one(eltype(x))
    ) where {M <: WrapperModel, V <: AbstractVector}

    copyto!(m.x, x)
    copyto!(m.y, y)
    NLPModels.hess_coord!(m.inner, m.x, m.y, m.hess; obj_weight=obj_weight)
    copyto!(hess, m.hess)
    return
end
