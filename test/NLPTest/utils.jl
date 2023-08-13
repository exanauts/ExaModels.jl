struct WrapperNLPModel{
    T, VT, VI,
    I <: NLPModels.AbstractNLPModel{T,VT}
    } <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}

    inner::I

    x_buffer:: VT
    y_buffer:: VT
    
    cons_buffer::VT
    grad_buffer::VT
    
    jac_buffer::VT
    jac_I_buffer::VI
    jac_J_buffer::VI

    hess_buffer::VT
    hess_I_buffer::VI
    hess_J_buffer::VI
    
    meta::NLPModels.AbstractNLPModelMeta{Float64,Vector{Float64}}
end

function WrapperNLPModel(m)
    nvar = get_nvar(m)
    ncon = get_ncon(m)
    nnzj = get_nnzj(m)
    nnzh = get_nnzh(m)
    
    x0   = Vector{Float64}(undef, nvar)
    lvar = Vector{Float64}(undef, nvar)
    uvar = Vector{Float64}(undef, nvar)
    
    y0   = Vector{Float64}(undef, ncon)
    lcon = Vector{Float64}(undef, ncon)
    ucon = Vector{Float64}(undef, ncon)    
    
    copyto!(x0, m.meta.x0)
    copyto!(lvar, m.meta.lvar)
    copyto!(uvar, m.meta.uvar)
    
    copyto!(y0, m.meta.y0)
    copyto!(lcon, m.meta.lcon)
    copyto!(ucon, m.meta.ucon)

    x_buffer = similar(get_x0(m), nvar)
    y_buffer = similar(get_x0(m), ncon)
    cons_buffer = similar(get_x0(m), ncon)
    grad_buffer = similar(get_x0(m), nvar)
    jac_buffer  = similar(get_x0(m), nnzj)
    jac_I_buffer = similar(get_x0(m), Int, nnzj)
    jac_J_buffer = similar(get_x0(m), Int, nnzj)
    hess_buffer  = similar(get_x0(m), nnzh)
    hess_I_buffer = similar(get_x0(m), Int, nnzh)
    hess_J_buffer = similar(get_x0(m), Int, nnzh)

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
        NLPModelMeta(
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
        )
    )
end

function NLPModels.jac_structure!(
    m::WrapperNLPModel,
    rows::AbstractVector,
    cols::AbstractVector
    )
    
    jac_structure!(m.inner, m.jac_I_buffer, m.jac_J_buffer)
    copyto!(rows, m.jac_I_buffer)
    copyto!(cols, m.jac_J_buffer)
end

function NLPModels.hess_structure!(
    m::WrapperNLPModel,
    rows::AbstractVector,
    cols::AbstractVector
    )

    hess_structure!(m.inner, m.hess_I_buffer, m.hess_J_buffer)
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
    unsafe_copyto!(pointer(hess), pointer(m.hess_buffer), length(hess))

    return
end    
