module ExaModelsKernelAbstractions

import ExaModels: ExaModels, NLPModels
import KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, synchronize, CPU

ExaModels.convert_array(v, backend::CPU) = v

function getitr(gen::UnitRange{Int64})
    return gen
end
function getitr(gen::Base.Iterators.ProductIterator{NTuple{N,UnitRange{Int64}}}) where {N} end

function ExaModels.getptr(backend, array; cmp = (x, y) -> x != y)

    bitarray = similar(array, Bool, length(array) + 1)
    kergetptr(backend)(cmp, bitarray, array; ndrange = length(array) + 1)

    return ExaModels.findall(identity, bitarray)
end


struct KAExtension{T,VT<:AbstractVector{T},H,VI1,VI2,B}
    backend::B
    objbuffer::VT
    gradbuffer::VT
    gsparsity::VI1
    gptr::VI2
    conbuffer::VT
    conaugsparsity::VI1
    conaugptr::VI2
    prodhelper::H
end

function ExaModels.build_extension(
    c::C;
    prod = false,
    kwargs...,
) where {T,VT<:AbstractVector{T},B<:KernelAbstractions.Backend,C<:ExaModels.ExaCore{T,VT,B}}

    gsparsity = similar(c.x0, Tuple{Int,Int}, c.nnzg)

    _grad_structure!(c.backend, c.obj, gsparsity)

    if !isempty(gsparsity)
        ExaModels.sort!(gsparsity; lt = ((i, j), (k, l)) -> i < k)
    end
    gptr = ExaModels.getptr(c.backend, gsparsity; cmp = (x, y) -> x[1] != y[1])

    conaugsparsity = similar(c.x0, Tuple{Int,Int}, c.nconaug)
    _conaug_structure!(c.backend, c.con, conaugsparsity)
    if !isempty(conaugsparsity)
        ExaModels.sort!(conaugsparsity; lt = ((i, j), (k, l)) -> i < k)
    end
    conaugptr = ExaModels.getptr(c.backend, conaugsparsity; cmp = (x, y) -> x[1] != y[1])


    if prod
        # c.nnzj / c.nnzh include oracle contributions (if any).  The SIMD
        # structure builders below only populate entries for c.con / c.obj.
        # Keep dense value buffers at full size, but size sparsity coordinate
        # arrays to the SIMD-only nnz counts to avoid uninitialized entries.
        simd_nnzj = c.nnzj - sum(getproperty(o, :nnzj) for o in c.oracles)
        simd_nnzh = c.nnzh - sum(getproperty(o, :nnzh) for o in c.oracles)

        jacbuffer = similar(c.x0, c.nnzj)
        hessbuffer = similar(c.x0, c.nnzh)
        jacsparsityi = similar(c.x0, Tuple{Tuple{Int,Int},Int}, simd_nnzj)
        hesssparsityi = similar(c.x0, Tuple{Tuple{Int,Int},Int}, simd_nnzh)

        _jac_structure!(c.backend, c.con, jacsparsityi, nothing)

        jacsparsityj = copy(jacsparsityi)
        _obj_hess_structure!(c.backend, c.obj, hesssparsityi, nothing)
        _con_hess_structure!(c.backend, c.con, hesssparsityi, nothing)
        hesssparsityj = copy(hesssparsityi)

        if !isempty(jacsparsityi)
            ExaModels.sort!(jacsparsityi; lt = (((i, j), k), ((n, m), l)) -> i < n)
        end
        jacptri =
            ExaModels.getptr(c.backend, jacsparsityi; cmp = (x, y) -> x[1][1] != y[1][1])

        if !isempty(jacsparsityj)
            ExaModels.sort!(jacsparsityj; lt = (((i, j), k), ((n, m), l)) -> j < m)
        end
        jacptrj =
            ExaModels.getptr(c.backend, jacsparsityj; cmp = (x, y) -> x[1][2] != y[1][2])


        if !isempty(hesssparsityi)
            ExaModels.sort!(hesssparsityi; lt = (((i, j), k), ((n, m), l)) -> i < n)
        end
        hessptri =
            ExaModels.getptr(c.backend, hesssparsityi; cmp = (x, y) -> x[1][1] != y[1][1])
        if !isempty(hesssparsityj)
            ExaModels.sort!(hesssparsityj; lt = (((i, j), k), ((n, m), l)) -> j < m)
        end
        hessptrj =
            ExaModels.getptr(c.backend, hesssparsityj; cmp = (x, y) -> x[1][2] != y[1][2])

        prodhelper = (
            jacbuffer = jacbuffer,
            jacsparsityi = jacsparsityi,
            jacsparsityj = jacsparsityj,
            jacptri = jacptri,
            jacptrj = jacptrj,
            hessbuffer = hessbuffer,
            hesssparsityi = hesssparsityi,
            hesssparsityj = hesssparsityj,
            hessptri = hessptri,
            hessptrj = hessptrj,
        )
    else
        prodhelper = nothing
    end

    return KAExtension(
        c.backend,
        similar(c.x0, c.nobj),
        similar(c.x0, c.nnzg),
        gsparsity,
        gptr,
        similar(c.x0, c.nconaug),
        conaugsparsity,
        conaugptr,
        prodhelper,
    )
end

function _conaug_structure!(backend, cons, sparsity)
    if !isempty(cons.itr)
        kers(backend)(sparsity, cons.f, cons.itr, cons.oa; ndrange = length(cons.itr))
    end
    _conaug_structure!(backend, cons.inner, sparsity)
end
function _conaug_structure!(backend, cons::ExaModels.Constraint, sparsity)
    _conaug_structure!(backend, cons.inner, sparsity)
end
function _conaug_structure!(backend, cons::ExaModels.ConstraintNull, sparsity) end
@kernel function kers(sparsity, @Const(f), @Const(itr), @Const(oa))
    I = @index(Global)
    @inbounds sparsity[oa+I] = (ExaModels.offset0(f, itr, I), oa + I)
end



function _grad_structure!(backend, objs, gsparsity)
    ExaModels.sgradient!(backend, gsparsity, objs, nothing, nothing, NaN)
    _grad_structure!(backend, objs.inner, gsparsity)
end
function _grad_structure!(backend, objs::ExaModels.ObjectiveNull, gsparsity) end

function ExaModels.jac_structure!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    rows::V,
    cols::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    if !isempty(rows)
        _jac_structure!(m.ext.backend, m.cons, rows, cols)
    end
    return rows, cols
end
function _jac_structure!(backend, cons, rows, cols)
    ExaModels.sjacobian!(backend, rows, cols, cons, nothing, nothing, NaN)
    _jac_structure!(backend, cons.inner, rows, cols)
end
function _jac_structure!(backend, cons::ExaModels.ConstraintNull, rows, cols) end


function ExaModels.hess_structure!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    rows::V,
    cols::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    if !isempty(rows)
        _obj_hess_structure!(m.ext.backend, m.objs, rows, cols)
        _con_hess_structure!(m.ext.backend, m.cons, rows, cols)
        end
    return rows, cols
end

function _obj_hess_structure!(backend, objs, rows, cols)
    ExaModels.shessian!(backend, rows, cols, objs, nothing, nothing, NaN, NaN)
    _obj_hess_structure!(backend, objs.inner, rows, cols)
end
function _obj_hess_structure!(backend, objs::ExaModels.ObjectiveNull, rows, cols) end
function _con_hess_structure!(backend, cons, rows, cols)
    ExaModels.shessian!(backend, rows, cols, cons, nothing, nothing, NaN, NaN)
    _con_hess_structure!(backend, cons.inner, rows, cols)
end
function _con_hess_structure!(backend, cons::ExaModels.ConstraintNull, rows, cols) end


function ExaModels.obj(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
) where {T,VT,E<:KAExtension}
    if !isempty(m.ext.objbuffer)
        _obj(m.ext.backend, m.ext.objbuffer, m.objs, x, m.θ)
            result = ExaModels.sum(m.ext.objbuffer)
        return result
    else
        return zero(T)
    end
end
function _obj(backend, objbuffer, obj, x, θ)
    if !isempty(obj.itr)
        kerf(backend)(objbuffer, obj.f, obj.itr, x, θ; ndrange = length(obj.itr))
    end
    _obj(backend, objbuffer, obj.inner, x, θ)
end
function _obj(backend, objbuffer, f::ExaModels.ObjectiveNull, x, θ) end

function ExaModels.cons_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    y::AbstractVector,
) where {T,VT,E<:KAExtension}
    _cons_nln!(m.ext.backend, y, m.cons, x, m.θ)
    _conaugs!(m.ext.backend, m.ext.conbuffer, m.cons, x, m.θ)

    if length(m.ext.conaugptr) > 1
        compress_to_dense(m.ext.backend)(
            y,
            m.ext.conbuffer,
            m.ext.conaugptr,
            m.ext.conaugsparsity;
            ndrange = length(m.ext.conaugptr) - 1,
        )
        end
    return y
end
function _cons_nln!(backend, y, con::ExaModels.Constraint, x, θ)
    if !isempty(con.itr)
        kerf(backend)(y, con.f, con.itr, x, θ; ndrange = length(con.itr))
    end
    _cons_nln!(backend, y, con.inner, x, θ)
end
function _cons_nln!(backend, y, con::ExaModels.ConstraintNull, x, θ) end
function _cons_nln!(backend, y, con::ExaModels.ConstraintAug, x, θ)
    _cons_nln!(backend, y, con.inner, x, θ)
end



function _conaugs!(backend, y, con::ExaModels.ConstraintAug, x, θ)
    if !isempty(con.itr)
        kerf2(backend)(y, con.f, con.itr, x, θ, con.oa; ndrange = length(con.itr))
    end
    _conaugs!(backend, y, con.inner, x, θ)
end
function _conaugs!(backend, y, con::ExaModels.Constraint, x, θ)
    _conaugs!(backend, y, con.inner, x, θ)
end
function _conaugs!(backend, y, con::ExaModels.ConstraintNull, x, θ) end

function ExaModels.grad!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::V,
    y::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    gradbuffer = m.ext.gradbuffer

    if !isempty(gradbuffer)
        fill!(gradbuffer, zero(eltype(gradbuffer)))
        _grad!(m.ext.backend, m.ext.gradbuffer, m.objs, x, m.θ)

        fill!(y, zero(eltype(y)))
        compress_to_dense(m.ext.backend)(
            y,
            gradbuffer,
            m.ext.gptr,
            m.ext.gsparsity;
            ndrange = length(m.ext.gptr) - 1,
        )
        end

    return y
end
function _grad!(backend, y, objs, x, θ)
    ExaModels.sgradient!(backend, y, objs, x, θ, one(eltype(y)))
    _grad!(backend, y, objs.inner, x, θ)
end
function _grad!(backend, y, objs::ExaModels.ObjectiveNull, x, θ) end

function ExaModels.jac_coord!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::V,
    y::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    fill!(y, zero(eltype(y)))
    _jac_coord!(m.ext.backend, y, m.cons, x, m.θ)
    return y
end
function _jac_coord!(backend, y, cons, x, θ)
    ExaModels.sjacobian!(backend, y, nothing, cons, x, θ, one(eltype(y)))
    _jac_coord!(backend, y, cons.inner, x, θ)
end
function _jac_coord!(backend, y, cons::ExaModels.ConstraintNull, x, θ) end

function ExaModels.jprod_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
) where {T,VT,E<:KAExtension{T,VT,Nothing}}
    error("Prodhelper is not defined. Use ExaModels(c; prod=true) to use jprod_nln!")
end
function ExaModels.jtprod_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
) where {T,VT,E<:KAExtension{T,VT,Nothing}}
    error("Prodhelper is not defined. Use ExaModels(c; prod=true) to use jtprod_nln!")
end
function ExaModels.jprod_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}

    fill!(Jv, zero(eltype(Jv)))
    fill!(m.ext.prodhelper.jacbuffer, zero(eltype(Jv)))
    _jac_coord!(m.ext.backend, m.ext.prodhelper.jacbuffer, m.cons, x, m.θ)
    kerspmv(m.ext.backend)(
        Jv,
        v,
        m.ext.prodhelper.jacsparsityi,
        m.ext.prodhelper.jacbuffer,
        m.ext.prodhelper.jacptri,
        ndrange = length(m.ext.prodhelper.jacptri) - 1,
    )
    return Jv
end
function ExaModels.jtprod_nln!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}

    fill!(Jtv, zero(eltype(Jtv)))
    fill!(m.ext.prodhelper.jacbuffer, zero(eltype(Jtv)))
    _jac_coord!(m.ext.backend, m.ext.prodhelper.jacbuffer, m.cons, x, m.θ)
    kerspmv2(m.ext.backend)(
        Jtv,
        v,
        m.ext.prodhelper.jacsparsityj,
        m.ext.prodhelper.jacbuffer,
        m.ext.prodhelper.jacptrj,
        ndrange = length(m.ext.prodhelper.jacptrj) - 1,
    )
    return Jtv
end
function ExaModels.hprod!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}

    if isnothing(m.ext.prodhelper)
        error("Prodhelper is not defined. Use ExaModels(c; prod=true) to use hprod!")
    end

    fill!(Hv, zero(eltype(Hv)))
    fill!(m.ext.prodhelper.hessbuffer, zero(eltype(Hv)))

    _obj_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.objs, x, m.θ, obj_weight)
    _con_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.cons, x, m.θ, y)
    kersyspmv(m.ext.backend)(
        Hv,
        v,
        m.ext.prodhelper.hesssparsityi,
        m.ext.prodhelper.hessbuffer,
        m.ext.prodhelper.hessptri,
        ndrange = length(m.ext.prodhelper.hessptri) - 1,
    )
    kersyspmv2(m.ext.backend)(
        Hv,
        v,
        m.ext.prodhelper.hesssparsityj,
        m.ext.prodhelper.hessbuffer,
        m.ext.prodhelper.hessptrj,
        ndrange = length(m.ext.prodhelper.hessptrj) - 1,
    )

    return Hv
end
function ExaModels.hprod!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}

    if isnothing(m.ext.prodhelper)
        error("Prodhelper is not defined. Use ExaModels(c; prod=true) to use hprod!")
    end

    fill!(Hv, zero(eltype(Hv)))
    fill!(m.ext.prodhelper.hessbuffer, zero(eltype(Hv)))

    _obj_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.objs, x, m.θ, obj_weight)
    kersyspmv(m.ext.backend)(
        Hv,
        v,
        m.ext.prodhelper.hesssparsityi,
        m.ext.prodhelper.hessbuffer,
        m.ext.prodhelper.hessptri,
        ndrange = length(m.ext.prodhelper.hessptri) - 1,
    )
    kersyspmv2(m.ext.backend)(
        Hv,
        v,
        m.ext.prodhelper.hesssparsityj,
        m.ext.prodhelper.hessbuffer,
        m.ext.prodhelper.hessptrj,
        ndrange = length(m.ext.prodhelper.hessptrj) - 1,
    )

    return Hv
end

@kernel function kerspmv(y, @Const(x), @Const(coord), @Const(V), @Const(ptr))
    idx = @index(Global)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        y[i] += V[ind] * x[j]
    end
end
@kernel function kerspmv2(y, @Const(x), @Const(coord), @Const(V), @Const(ptr))
    idx = @index(Global)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        y[j] += V[ind] * x[i]
    end
end
@kernel function kersyspmv(y, @Const(x), @Const(coord), @Const(V), @Const(ptr))
    idx = @index(Global)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        y[i] += V[ind] * x[j]
    end
end
@kernel function kersyspmv2(y, @Const(x), @Const(coord), @Const(V), @Const(ptr))
    idx = @index(Global)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        if i != j
            y[j] += V[ind] * x[i]
        end
    end
end



function ExaModels.hess_coord!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
) where {T,VT,E<:KAExtension}
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.ext.backend, hess, m.objs, x, m.θ, obj_weight)
    return hess
end

function ExaModels.hess_coord!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(y)),
) where {T,VT,E<:KAExtension}
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.ext.backend, hess, m.objs, x, m.θ, obj_weight)
    _con_hess_coord!(m.ext.backend, hess, m.cons, x, m.θ, y)
    return hess
end
function _obj_hess_coord!(backend, hess, objs, x, θ, obj_weight)
    ExaModels.shessian!(backend, hess, nothing, objs, x, θ, obj_weight, zero(eltype(hess)))
    _obj_hess_coord!(backend, hess, objs.inner, x, θ, obj_weight)
end
function _obj_hess_coord!(backend, hess, objs::ExaModels.ObjectiveNull, x, θ, obj_weight) end
function _con_hess_coord!(backend, hess, cons, x, θ, y)
    ExaModels.shessian!(backend, hess, nothing, cons, x, θ, y, zero(eltype(hess)))
    _con_hess_coord!(backend, hess, cons.inner, x, θ, y)
end
function _con_hess_coord!(backend, hess, cons::ExaModels.ConstraintNull, x, θ, y) end


function ExaModels.sgradient!(
    backend::B,
    y,
    f,
    x,
    θ,
    adj,
) where {B<:KernelAbstractions.Backend}

    if !isempty(f.itr)
        kerg(backend)(y, f.f, f.itr, x, θ, adj; ndrange = length(f.itr))
    end
end

function ExaModels.sjacobian!(
    backend::B,
    y1,
    y2,
    f,
    x,
    θ,
    adj,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        kerj(backend)(y1, y2, f.f, f.itr, x, θ, adj; ndrange = length(f.itr))
    end
end

function ExaModels.shessian!(
    backend::B,
    y1,
    y2,
    f,
    x,
    θ,
    adj,
    adj2,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        kerh(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
    end
end

function ExaModels.shessian!(
    backend::B,
    y1,
    y2,
    f,
    x,
    θ,
    adj::V,
    adj2,
) where {B<:KernelAbstractions.Backend,V<:AbstractVector}
    if !isempty(f.itr)
        kerh2(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
    end
end

@kernel function kerh(
    y1,
    y2,
    @Const(f),
    @Const(itr),
    @Const(x),
    @Const(θ),
    @Const(adj1),
    @Const(adj2)
)
    I = @index(Global)
    @inbounds ExaModels.hrpass0(
        f(itr[I], ExaModels.SecondAdjointNodeSource(x), θ),
        f.comp2,
        y1,
        y2,
        ExaModels.offset2(f, I),
        0,
        adj1,
        adj2,
    )
end

@kernel function kerh2(
    y1,
    y2,
    @Const(f),
    @Const(itr),
    @Const(x),
    @Const(θ),
    @Const(adjs1),
    @Const(adj2)
)
    I = @index(Global)
    @inbounds ExaModels.hrpass0(
        f(itr[I], ExaModels.SecondAdjointNodeSource(x), θ),
        f.comp2,
        y1,
        y2,
        ExaModels.offset2(f, I),
        0,
        adjs1[ExaModels.offset0(f, itr, I)],
        adj2,
    )
end

@kernel function kerj(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj))
    I = @index(Global)
    @inbounds ExaModels.jrpass(
        f(itr[I], ExaModels.AdjointNodeSource(x), θ),
        f.comp1,
        ExaModels.offset0(f, itr, I),
        y1,
        y2,
        ExaModels.offset1(f, I),
        0,
        adj,
    )
end

@kernel function kerg(y, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj))
    I = @index(Global)
    @inbounds ExaModels.grpass(
        f(itr[I], ExaModels.AdjointNodeSource(x), θ),
        f.comp1,
        y,
        ExaModels.offset1(f, I),
        0,
        adj,
    )
end

@kernel function kerf(y, @Const(f), @Const(itr), @Const(x), @Const(θ))
    I = @index(Global)
    @inbounds y[ExaModels.offset0(f, itr, I)] = f(itr[I], x, θ)
end
@kernel function kerf2(y, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(oa))
    I = @index(Global)
    @inbounds y[oa+I] = f(itr[I], x, θ)
end


@kernel function compress_to_dense(y, @Const(y0), @Const(ptr), @Const(sparsity))
    I = @index(Global)
    @inbounds for j = ptr[I]:(ptr[I+1]-1)
        (k, l) = sparsity[j]
        y[k] += y0[l]
    end
end

@kernel function kergetptr(cmp, bitarray, @Const(array))
    I = @index(Global)
    @inbounds if I == 1
        bitarray[I] = true
    elseif I == length(array) + 1
        bitarray[I] = true
    else
        i0 = array[I-1]
        i1 = array[I]

        if cmp(i0, i1)
            bitarray[I] = true
        else
            bitarray[I] = false
        end
    end
end

ExaModels.getbackend(m::ExaModels.AbstractExaModel{T,VT,E}) where {T,VT,E<:KAExtension} =
    m.ext.backend

# ── GPU-specific overrides for ExaModelWithOracle ────────────────────────────
#
# When the backend is a KA GPU (CUDA, ROCm, …) the generic oracle.jl methods
# would fall back to CPU scalar loops on device arrays.  The specialisations
# below dispatch on E<:KAExtension and are therefore *more specific* than the
# plain ExaModelWithOracle methods, so Julia's method resolution picks them for
# GPU models.
#
# Design principle:
#   • SIMD symbolic part  → GPU kernels (same as the non-oracle path)
#   • Oracle callback part → CPU bridge: Array(gpu) → callback → copyto!(gpu, cpu)
#     Oracle callbacks are always CPU functions (including PyCall wrappers).

function ExaModels.jac_structure!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    rows::V,
    cols::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    if !isempty(rows)
        _jac_structure!(m.ext.backend, m.cons, rows, cols)
    end
    # Oracle sparsity is static; copyto! from CPU arrays works for any backend.
    for (i, oracle) in enumerate(m.oracles)
        off_j = m.oracle_jac_offsets[i]
        off_c = m.oracle_con_offsets[i]
        if oracle.nnzj > 0
            copyto!(view(rows, off_j+1 : off_j+oracle.nnzj),
                    oracle.jac_rows .+ off_c)
            copyto!(view(cols, off_j+1 : off_j+oracle.nnzj),
                    oracle.jac_cols)
        end
    end
    return rows, cols
end

function ExaModels.hess_structure!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    rows::V,
    cols::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    if !isempty(rows)
        _obj_hess_structure!(m.ext.backend, m.objs, rows, cols)
        _con_hess_structure!(m.ext.backend, m.cons, rows, cols)
    end
    for (i, oracle) in enumerate(m.oracles)
        off_h = m.oracle_hess_offsets[i]
        if oracle.nnzh > 0
            copyto!(view(rows, off_h+1 : off_h+oracle.nnzh), oracle.hess_rows)
            copyto!(view(cols, off_h+1 : off_h+oracle.nnzh), oracle.hess_cols)
        end
    end
    return rows, cols
end

function ExaModels.cons_nln!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    x::AbstractVector,
    y::AbstractVector,
) where {T,VT,E<:KAExtension}
    # SIMD part: GPU-accelerated kernels.
    _cons_nln!(m.ext.backend, y, m.cons, x, m.θ)
    _conaugs!(m.ext.backend, m.ext.conbuffer, m.cons, x, m.θ)
    if length(m.ext.conaugptr) > 1
        compress_to_dense(m.ext.backend)(
            y,
            m.ext.conbuffer,
            m.ext.conaugptr,
            m.ext.conaugsparsity;
            ndrange = length(m.ext.conaugptr) - 1,
        )
    end
    # Oracle part: route via _oracle_input (device array if gpu=true, else CPU copy).
    for (i, oracle) in enumerate(m.oracles)
        off  = m.oracle_con_offsets[i]
        xin  = ExaModels._oracle_input(oracle, x)
        cv   = similar(xin, oracle.ncon)
        oracle.f!(cv, xin)
        copyto!(view(y, off+1 : off+oracle.ncon), cv)
    end
    return y
end

function ExaModels.jac_coord!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    x::V,
    jac::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    fill!(jac, zero(eltype(jac)))
    # SIMD part: GPU-accelerated.
    _jac_coord!(m.ext.backend, jac, m.cons, x, m.θ)
    # Oracle part.
    for (i, oracle) in enumerate(m.oracles)
        off_j = m.oracle_jac_offsets[i]
        if oracle.nnzj > 0
            xin = ExaModels._oracle_input(oracle, x)
            jv  = similar(xin, oracle.nnzj)
            oracle.jac!(jv, xin)
            copyto!(view(jac, off_j+1 : off_j+oracle.nnzj), jv)
        end
    end
    return jac
end

function ExaModels.hess_coord!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
) where {T,VT,E<:KAExtension}
    fill!(hess, zero(eltype(hess)))
    # SIMD part: GPU-accelerated.
    _obj_hess_coord!(m.ext.backend, hess, m.objs, x, m.θ, obj_weight)
    _con_hess_coord!(m.ext.backend, hess, m.cons, x, m.θ, y)
    # Oracle part.
    for (i, oracle) in enumerate(m.oracles)
        off_h = m.oracle_hess_offsets[i]
        off_c = m.oracle_con_offsets[i]
        if oracle.nnzh > 0
            xin    = ExaModels._oracle_input(oracle, x)
            yslice = oracle.gpu ? view(y, off_c+1 : off_c+oracle.ncon) :
                                  view(Array(y), off_c+1 : off_c+oracle.ncon)
            hv = similar(xin, oracle.nnzh)
            oracle.hess!(hv, xin, yslice)
            copyto!(view(hess, off_h+1 : off_h+oracle.nnzh), hv)
        end
    end
    return hess
end

# ── jprod_nln! / jtprod_nln! / hprod! for ExaModelWithOracle on GPU ──────────
#
# Strategy:
#   • SIMD part: re-use the existing prod-helper GPU kernels (kerspmv etc.)
#     which operate entirely on device memory with no scalar indexing.
#   • Oracle part (gpu=true):  call jac!/hess! with device arrays, then use
#     kerspmv/kersyspmv directly on the oracle's own device sparsity buffers.
#   • Oracle part (gpu=false): CPU bridge; accumulate on host and copyto! back.
#
# For the "prod=false" case (no prodhelper) we still handle the oracle part
# correctly; the SIMD part falls back to the cpu path via the base method.

# Helper: build the ((row,col), global_index) sparsity vector for one oracle's
# Jacobian, in the format expected by kerspmv / kerspmv2.
function _oracle_jac_sparsity(oracle, off_j, off_c, backend, x0)
    n = oracle.nnzj
    cpu = [((oracle.jac_rows[k] + off_c, oracle.jac_cols[k]), off_j + k)
           for k in 1:n]
    dev = similar(x0, eltype(cpu), n)
    copyto!(dev, cpu)
    return dev
end

# Helper: build the symmetric sparsity vector for one oracle's Hessian.
function _oracle_hess_sparsity(oracle, off_h, backend, x0)
    n = oracle.nnzh
    cpu = [((oracle.hess_rows[k], oracle.hess_cols[k]), off_h + k)
           for k in 1:n]
    dev = similar(x0, eltype(cpu), n)
    copyto!(dev, cpu)
    return dev
end

# jprod_nln! – error if no prodhelper for the SIMD part
function ExaModels.jprod_nln!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
) where {T,VT,E<:KAExtension{T,VT,Nothing}}
    error("Prodhelper is not defined. Use ExaModel(c; prod=true) to use jprod_nln!")
end
function ExaModels.jtprod_nln!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
) where {T,VT,E<:KAExtension{T,VT,Nothing}}
    error("Prodhelper is not defined. Use ExaModel(c; prod=true) to use jtprod_nln!")
end

function ExaModels.jprod_nln!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}
    fill!(Jv, zero(eltype(Jv)))
    # SIMD symbolic part via GPU sparse-matvec kernel.
    fill!(m.ext.prodhelper.jacbuffer, zero(eltype(Jv)))
    _jac_coord!(m.ext.backend, m.ext.prodhelper.jacbuffer, m.cons, x, m.θ)
    let _n = length(m.ext.prodhelper.jacptri) - 1
        _n > 0 && kerspmv(m.ext.backend)(
            Jv, v,
            m.ext.prodhelper.jacsparsityi,
            m.ext.prodhelper.jacbuffer,
            m.ext.prodhelper.jacptri;
            ndrange = _n,
        )
    end
    # Oracle part.
    x0 = m.meta.x0  # used only to allocate similarly-typed device arrays
    for (i, oracle) in enumerate(m.oracles)
        off_j = m.oracle_jac_offsets[i]
        off_c = m.oracle_con_offsets[i]
        oracle.nnzj == 0 && continue
        xin    = ExaModels._oracle_input(oracle, x)
        jac_buf = similar(xin, oracle.nnzj)
        oracle.jac!(jac_buf, xin)
        if oracle.gpu
            # Full GPU path: build device sparsity and use kerspmv kernel.
            sp  = _oracle_jac_sparsity(oracle, off_j, off_c, m.ext.backend, x0)
            ptr = ExaModels.getptr(m.ext.backend, sp; cmp = (a, b) -> a[1][1] != b[1][1])
            copyto!(view(m.ext.prodhelper.jacbuffer, off_j+1 : off_j+oracle.nnzj), jac_buf)
            kerspmv(m.ext.backend)(
                Jv, v, sp, m.ext.prodhelper.jacbuffer, ptr;
                ndrange = length(ptr) - 1,
            )
        else
            # CPU bridge accumulation.
            jac_host = Array(jac_buf)
            v_host   = Array(v)
            delta    = zeros(T, length(Jv))
            for k in 1:oracle.nnzj
                delta[oracle.jac_rows[k] + off_c] += jac_host[k] * v_host[oracle.jac_cols[k]]
            end
            buf = similar(Jv)
            copyto!(buf, delta)
            Jv .+= buf
        end
    end
    return Jv
end

function ExaModels.jtprod_nln!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}
    fill!(Jtv, zero(eltype(Jtv)))
    # SIMD symbolic part.
    fill!(m.ext.prodhelper.jacbuffer, zero(eltype(Jtv)))
    _jac_coord!(m.ext.backend, m.ext.prodhelper.jacbuffer, m.cons, x, m.θ)
    let _n = length(m.ext.prodhelper.jacptrj) - 1
        _n > 0 && kerspmv2(m.ext.backend)(
            Jtv, v,
            m.ext.prodhelper.jacsparsityj,
            m.ext.prodhelper.jacbuffer,
            m.ext.prodhelper.jacptrj;
            ndrange = _n,
        )
    end
    # Oracle part.
    x0 = m.meta.x0
    for (i, oracle) in enumerate(m.oracles)
        off_j = m.oracle_jac_offsets[i]
        off_c = m.oracle_con_offsets[i]
        oracle.nnzj == 0 && continue
        xin     = ExaModels._oracle_input(oracle, x)
        jac_buf = similar(xin, oracle.nnzj)
        oracle.jac!(jac_buf, xin)
        if oracle.gpu
            # Build column-sorted sparsity and use kerspmv2 (J^T v kernel).
            sp  = _oracle_jac_sparsity(oracle, off_j, off_c, m.ext.backend, x0)
            ExaModels.sort!(sp; lt = (((r1,c1),_), ((r2,c2),__)) -> c1 < c2)
            ptr = ExaModels.getptr(m.ext.backend, sp; cmp = (a, b) -> a[1][2] != b[1][2])
            copyto!(view(m.ext.prodhelper.jacbuffer, off_j+1 : off_j+oracle.nnzj), jac_buf)
            kerspmv2(m.ext.backend)(
                Jtv, v, sp, m.ext.prodhelper.jacbuffer, ptr;
                ndrange = length(ptr) - 1,
            )
        else
            jac_host = Array(jac_buf)
            v_host   = Array(v)
            delta    = zeros(T, length(Jtv))
            for k in 1:oracle.nnzj
                delta[oracle.jac_cols[k]] += jac_host[k] * v_host[oracle.jac_rows[k] + off_c]
            end
            buf = similar(Jtv)
            copyto!(buf, delta)
            Jtv .+= buf
        end
    end
    return Jtv
end

function ExaModels.hprod!(
    m::ExaModels.ExaModelWithOracle{T,VT,E},
    x::AbstractVector,
    y::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(eltype(x)),
) where {T,VT,N<:NamedTuple,E<:KAExtension{T,VT,N}}
    fill!(Hv, zero(eltype(Hv)))
    fill!(m.ext.prodhelper.hessbuffer, zero(eltype(Hv)))
    # SIMD symbolic part.
    _obj_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.objs, x, m.θ, obj_weight)
    _con_hess_coord!(m.ext.backend, m.ext.prodhelper.hessbuffer, m.cons, x, m.θ, y)
    let _n = length(m.ext.prodhelper.hessptri) - 1
        _n > 0 && kersyspmv(m.ext.backend)(
            Hv, v,
            m.ext.prodhelper.hesssparsityi,
            m.ext.prodhelper.hessbuffer,
            m.ext.prodhelper.hessptri;
            ndrange = _n,
        )
    end
    let _n = length(m.ext.prodhelper.hessptrj) - 1
        _n > 0 && kersyspmv2(m.ext.backend)(
            Hv, v,
            m.ext.prodhelper.hesssparsityj,
            m.ext.prodhelper.hessbuffer,
            m.ext.prodhelper.hessptrj;
            ndrange = _n,
        )
    end
    # Oracle part.
    x0 = m.meta.x0
    for (i, oracle) in enumerate(m.oracles)
        off_h = m.oracle_hess_offsets[i]
        off_c = m.oracle_con_offsets[i]
        oracle.nnzh == 0 && continue
        xin    = ExaModels._oracle_input(oracle, x)
        yslice = oracle.gpu ? view(y, off_c+1 : off_c+oracle.ncon) :
                              view(Array(y), off_c+1 : off_c+oracle.ncon)
        hess_buf = similar(xin, oracle.nnzh)
        oracle.hess!(hess_buf, xin, yslice)
        if oracle.gpu
            sp  = _oracle_hess_sparsity(oracle, off_h, m.ext.backend, x0)
            ExaModels.sort!(sp; lt = (((r1,c1),_), ((r2,c2),__)) -> r1 < r2)
            ptri = ExaModels.getptr(m.ext.backend, sp; cmp = (a,b) -> a[1][1] != b[1][1])
            spj  = copy(sp)
            ExaModels.sort!(spj; lt = (((r1,c1),_), ((r2,c2),__)) -> c1 < c2)
            ptrj = ExaModels.getptr(m.ext.backend, spj; cmp = (a,b) -> a[1][2] != b[1][2])
            copyto!(view(m.ext.prodhelper.hessbuffer, off_h+1 : off_h+oracle.nnzh), hess_buf)
            kersyspmv(m.ext.backend)(
                Hv, v, sp, m.ext.prodhelper.hessbuffer, ptri;
                ndrange = length(ptri) - 1,
            )
            kersyspmv2(m.ext.backend)(
                Hv, v, spj, m.ext.prodhelper.hessbuffer, ptrj;
                ndrange = length(ptrj) - 1,
            )
        else
            h_host = Array(hess_buf)
            v_host = Array(v)
            delta  = zeros(T, length(Hv))
            for k in 1:oracle.nnzh
                r, c_ = oracle.hess_rows[k], oracle.hess_cols[k]
                delta[r] += h_host[k] * v_host[c_]
                r != c_ && (delta[c_] += h_host[k] * v_host[r])
            end
            buf = similar(Hv)
            copyto!(buf, delta)
            Hv .+= buf
        end
    end
    return Hv
end

function ExaModels._compress!(V, buffer, ptr, sparsity, backend)
    fill!(V, zero(eltype(V)))
    ker_compress!(backend)(V, buffer, ptr, sparsity; ndrange = length(ptr) - 1)
end

@kernel function ker_compress!(V, @Const(buffer), @Const(ptr), @Const(sparsity))
    i = @index(Global)
    @inbounds for j = ptr[i]:(ptr[i+1]-1)
        V[i] += buffer[sparsity[j][2]]
    end
end

function ExaModels._structure!(I, J, ptr, sparsity, backend)
    ker_structure!(backend)(I, J, ptr, sparsity, ndrange = length(ptr) - 1)
end

@kernel function ker_structure!(I, J, @Const(ptr), @Const(sparsity))
    i = @index(Global)
    @inbounds J[i], I[i] = sparsity[ptr[i]][1]
end

function ExaModels.get_compressed_sparsity(nnz, Ibuffer, Jbuffer, backend)
    sparsity = similar(Ibuffer, Tuple{Tuple{Int,Int},Int}, nnz)
    ker_get_compressed_sparsity(backend)(sparsity, Ibuffer, Jbuffer; ndrange = nnz)
    return sparsity
end
@kernel function ker_get_compressed_sparsity(sparsity, @Const(I), @Const(J))
    i = @index(Global)
    @inbounds sparsity[i] = ((J[i], I[i]), i)
end

end # module ExaModelsKernelAbstractions


