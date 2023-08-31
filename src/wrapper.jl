struct WrapperNLPModel{
    T, VT, T2, VT2 <: AbstractVector{T2}, VI,
    I <: NLPModels.AbstractNLPModel{T2,VT2}
    } <: NLPModels.AbstractNLPModel{T,VT}

    inner::I

    x_buffer:: VT2
    y_buffer:: VT2
    
    cons_buffer::VT2
    grad_buffer::VT2
    
    jac_buffer::VT2
    jac_I_buffer::VI
    jac_J_buffer::VI

    hess_buffer::VT2
    hess_I_buffer::VI
    hess_J_buffer::VI
    
    meta::NLPModels.AbstractNLPModelMeta{T,VT}
    counters::NLPModels.Counters
end

WrapperNLPModel(m) = WrapperNLPModel(Vector{Float64},m)
function WrapperNLPModel(VT,m)
    nvar = NLPModels.get_nvar(m)
    ncon = NLPModels.get_ncon(m)
    nnzj = NLPModels.get_nnzj(m)
    nnzh = NLPModels.get_nnzh(m)
    
    x0   = VT(undef, nvar)
    lvar = VT(undef, nvar)
    uvar = VT(undef, nvar)
    
    y0   = VT(undef, ncon)
    lcon = VT(undef, ncon)
    ucon = VT(undef, ncon)    
    
    copyto!(x0, m.meta.x0)
    copyto!(lvar, m.meta.lvar)
    copyto!(uvar, m.meta.uvar)
    
    copyto!(y0, m.meta.y0)
    copyto!(lcon, m.meta.lcon)
    copyto!(ucon, m.meta.ucon)

    x_buffer = similar(m.meta.x0, nvar)
    y_buffer = similar(m.meta.x0, ncon)
    cons_buffer = similar(m.meta.x0, ncon)
    grad_buffer = similar(m.meta.x0, nvar)
    jac_buffer  = similar(m.meta.x0, nnzj)
    jac_I_buffer = similar(m.meta.x0, Int, nnzj)
    jac_J_buffer = similar(m.meta.x0, Int, nnzj)
    hess_buffer  = similar(m.meta.x0, nnzh)
    hess_I_buffer = similar(m.meta.x0, Int, nnzh)
    hess_J_buffer = similar(m.meta.x0, Int, nnzh)

    return WrapperNLPModel(
        m,
        x_buffer,
        y_buffer,
        cons_buffer,
        grad_buffer,
        jac_buffer,
        jac_I_buffer,
        jac_J_buffer,
        hess_buffer,
        hess_I_buffer,
        hess_J_buffer,
        NLPModels.NLPModelMeta(
            nvar,
            x0 = x0,
            lvar = lvar,
            uvar = uvar,
            ncon = ncon,
            y0 = y0,
            lcon = lcon,
            ucon = ucon,
            nnzj = nnzj,
            nnzh = nnzh,
            minimize = m.meta.minimize
        ),
        NLPModels.Counters()
    )
end

function NLPModels.jac_structure!(
    m::WrapperNLPModel,
    rows::AbstractVector,
    cols::AbstractVector
    )
    
    NLPModels.jac_structure!(m.inner, m.jac_I_buffer, m.jac_J_buffer)
    copyto!(rows, m.jac_I_buffer)
    copyto!(cols, m.jac_J_buffer)
end

function NLPModels.hess_structure!(
    m::WrapperNLPModel,
    rows::AbstractVector,
    cols::AbstractVector
    )

    NLPModels.hess_structure!(m.inner, m.hess_I_buffer, m.hess_J_buffer)
    copyto!(rows, m.hess_I_buffer)
    copyto!(cols, m.hess_J_buffer)
end

function NLPModels.obj(
    m::WrapperNLPModel,
    x::AbstractVector
    )

    copyto!(m.x_buffer, x)
    o = NLPModels.obj(m.inner, m.x_buffer)
    return o
end
function NLPModels.cons!(
    m::WrapperNLPModel,
    x::AbstractVector,
    g::AbstractVector
    )

    copyto!(m.x_buffer, x)
    NLPModels.cons!(m.inner, m.x_buffer, m.cons_buffer)
    copyto!(g, m.cons_buffer)
    return 
end
function NLPModels.grad!(
    m::WrapperNLPModel,
    x::AbstractVector,
    f::AbstractVector
    )

    copyto!(m.x_buffer, x)
    NLPModels.grad!(m.inner, m.x_buffer, m.grad_buffer)
    copyto!(f, m.grad_buffer)
    return
end
function NLPModels.jac_coord!(
    m::WrapperNLPModel,
    x::AbstractVector,
    jac::AbstractVector
    )

    copyto!(m.x_buffer, x)
    NLPModels.jac_coord!(m.inner, m.x_buffer, m.jac_buffer)
    copyto!(jac, m.jac_buffer)
    return
end
function NLPModels.hess_coord!(
    m::WrapperNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x))
    )

    copyto!(m.x_buffer, x)
    copyto!(m.y_buffer, y)
    NLPModels.hess_coord!(
        m.inner, m.x_buffer, m.y_buffer, m.hess_buffer;
        obj_weight=obj_weight
    )
    copyto!(unsafe_wrap(Array, pointer(hess), length(hess)), m.hess_buffer)
    return
end    
