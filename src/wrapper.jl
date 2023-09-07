struct WrapperNLPModel{
    T,
    VT,
    T2,
    VT2<:AbstractVector{T2},
    VI,
    I<:NLPModels.AbstractNLPModel{T2,VT2},
} <: NLPModels.AbstractNLPModel{T,VT}

    inner::I

    x_result::VT
    y_result::VT

    x_buffer::VT2
    y_buffer::VT2
    v_buffer::VT2

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

WrapperNLPModel(m) = WrapperNLPModel(Vector{Float64}, m)
function WrapperNLPModel(VT, m)
    nvar = NLPModels.get_nvar(m)
    ncon = NLPModels.get_ncon(m)
    nnzj = NLPModels.get_nnzj(m)
    nnzh = NLPModels.get_nnzh(m)

    x_result = VT(undef, nvar)
    y_result = VT(undef, ncon)

    x0 = VT(undef, nvar)
    lvar = VT(undef, nvar)
    uvar = VT(undef, nvar)

    y0 = VT(undef, ncon)
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
    v_buffer = similar(m.meta.x0, nvar)
    cons_buffer = similar(m.meta.x0, ncon)
    grad_buffer = similar(m.meta.x0, nvar)
    jac_buffer = similar(m.meta.x0, nnzj)
    jac_I_buffer = similar(m.meta.x0, Int, nnzj)
    jac_J_buffer = similar(m.meta.x0, Int, nnzj)
    hess_buffer = similar(m.meta.x0, nnzh)
    hess_I_buffer = similar(m.meta.x0, Int, nnzh)
    hess_J_buffer = similar(m.meta.x0, Int, nnzh)

    return WrapperNLPModel(
        m,
        x_result,
        y_result,
        x_buffer,
        y_buffer,
        v_buffer,
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
            minimize = m.meta.minimize,
        ),
        NLPModels.Counters(),
    )
end

function NLPModels.jac_structure!(
    m::WrapperNLPModel,
    rows::AbstractVector,
    cols::AbstractVector,
)

    NLPModels.jac_structure!(m.inner, m.jac_I_buffer, m.jac_J_buffer)
    copyto!(rows, m.jac_I_buffer)
    copyto!(cols, m.jac_J_buffer)
end

function NLPModels.hess_structure!(
    m::WrapperNLPModel,
    rows::AbstractVector,
    cols::AbstractVector,
)

    NLPModels.hess_structure!(m.inner, m.hess_I_buffer, m.hess_J_buffer)
    copyto!(rows, m.hess_I_buffer)
    copyto!(cols, m.hess_J_buffer)
end

function NLPModels.obj(m::WrapperNLPModel, x::AbstractVector)

    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    o = NLPModels.obj(m.inner, m.x_buffer)
    return o
end
function NLPModels.cons_nln!(m::WrapperNLPModel, x::AbstractVector, g::AbstractVector)

    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    NLPModels.cons_nln!(m.inner, m.x_buffer, m.cons_buffer)
    copyto!(m.y_result, m.cons_buffer)
    copyto!(g, m.y_result)
    return
end
function NLPModels.grad!(m::WrapperNLPModel, x::AbstractVector, f::AbstractVector)

    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    NLPModels.grad!(m.inner, m.x_buffer, m.grad_buffer)
    copyto!(m.x_result, m.grad_buffer)
    copyto!(f, m.x_result)
    return
end
function NLPModels.jac_coord!(m::WrapperNLPModel, x::AbstractVector, jac::AbstractVector)

    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    NLPModels.jac_coord!(m.inner, m.x_buffer, m.jac_buffer)
    copyto!(jac, m.jac_buffer)
    return
end
function NLPModels.hess_coord!(
    m::WrapperNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)

    copyto!(m.x_buffer, x)
    copyto!(m.y_buffer, y)
    NLPModels.hess_coord!(
        m.inner,
        m.x_buffer,
        m.y_buffer,
        m.hess_buffer;
        obj_weight = obj_weight,
    )
    copyto!(unsafe_wrap(Array, pointer(hess), length(hess)), m.hess_buffer)
    return
end
function NLPModels.jprod_nln!(
    m::WrapperNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)

    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    copyto!(m.x_result, v)
    copyto!(m.grad_buffer, m.x_result)
    NLPModels.jprod_nln!(m.inner, m.x_buffer, m.grad_buffer, m.cons_buffer)
    copyto!(m.y_result, m.cons_buffer)
    copyto!(Jv, m.y_result)
    return
end
function NLPModels.jtprod_nln!(
    m::WrapperNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)

    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    copyto!(m.y_result, v)
    copyto!(m.cons_buffer, m.y_result)
    NLPModels.jtprod_nln!(m.inner, m.x_buffer, m.cons_buffer, m.grad_buffer)
    copyto!(m.x_result, m.grad_buffer)
    copyto!(Jtv, m.x_result)
    return
end
function NLPModels.hprod!(
    m::WrapperNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
)

    copyto!(m.x_result, x)
    copyto!(m.x_buffer, m.x_result)
    copyto!(m.y_buffer, y)
    copyto!(m.x_result, v)
    copyto!(m.grad_buffer, m.x_result)
    NLPModels.hprod!(
        m.inner,
        m.x_buffer,
        m.y_buffer,
        m.grad_buffer,
        m.v_buffer;
        obj_weight = obj_weight,
    )
    copyto!(m.x_result, m.grad_buffer)
    copyto!(Hv, m.x_result)
    return
end
