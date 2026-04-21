module ExaModelsKernelAbstractions

import ExaModels: ExaModels, NLPModels
import KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, synchronize, CPU

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
) where {T,VT<:AbstractArray{T},B<:KernelAbstractions.Backend,C<:ExaModels.ExaCore{T,VT,B}}

    gsparsity = similar(c.x0, Tuple{Int,Int}, c.nnzg)

    _grad_structure!(T, c.backend, c.obj, gsparsity)

    if !isempty(gsparsity)
        ExaModels.sort!(gsparsity; lt = ((i, j), (k, l)) -> i < k)
    end
    gptr = ExaModels.getptr(c.backend, gsparsity; cmp = (x, y) -> x[1] != y[1])

    conaugsparsity = similar(c.x0, Tuple{Int,Int}, c.nconaug)
    _conaug_structure!(T, c.backend, c.cons, conaugsparsity)
    if !isempty(conaugsparsity)
        ExaModels.sort!(conaugsparsity; lt = ((i, j), (k, l)) -> i < k)
    end
    conaugptr = ExaModels.getptr(c.backend, conaugsparsity; cmp = (x, y) -> x[1] != y[1])


    if prod
        jacbuffer = similar(c.x0, c.nnzj)
        hessbuffer = similar(c.x0, c.nnzh)
        jacsparsityi = similar(c.x0, Tuple{Tuple{Int,Int},Int}, c.nnzj)
        hesssparsityi = similar(c.x0, Tuple{Tuple{Int,Int},Int}, c.nnzh)

        ExaModels._jac_structure!(T, c.backend, c.cons, jacsparsityi, nothing)

        jacsparsityj = copy(jacsparsityi)
        ExaModels._obj_hess_structure!(T, c.backend, c.obj, hesssparsityi, nothing)
        ExaModels._con_hess_structure!(T, c.backend, c.cons, hesssparsityi, nothing)
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

_conaug_structure!(T, backend, ::Tuple{}, sparsity) = nothing
function _conaug_structure!(T, backend, (con, cons...), sparsity)
    _conaug_structure!(T, backend, cons, sparsity)
    con isa ExaModels.ConstraintAugmentation && !isempty(con.itr) &&
        kers(backend)(sparsity, con.f, con.itr, con.oa, con.dims; ndrange = length(con.itr))
end
@kernel function kers(sparsity, @Const(f), @Const(itr), @Const(oa), @Const(dims))
    I = @index(Global)
    @inbounds sparsity[oa+I] = (ExaModels.offset0(f, itr, I, dims), oa + I)
end



_grad_structure!(T, backend, ::Tuple{}, gsparsity) = nothing
function _grad_structure!(T, backend, (obj, objs...), gsparsity)
    _grad_structure!(T, backend, objs, gsparsity)
    ExaModels.sgradient!(gsparsity, obj, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), T(NaN), 1, 0, 0, length(gsparsity), backend)
end

function ExaModels.jac_structure!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    rows::V,
    cols::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    if !isempty(rows)
        ExaModels._jac_structure!(T, m.ext.backend, m.cons, rows, cols)
    end
    return rows, cols
end
ExaModels._jac_structure!(T, backend, ::Tuple{}, rows, cols) = nothing
function ExaModels._jac_structure!(T, backend, (con, cons...), rows, cols)
    ExaModels._jac_structure!(T, backend, cons, rows, cols)
    ExaModels.sjacobian!(rows, cols, con, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), T(NaN), 1, 0, 0, length(rows), backend)
end


function ExaModels.hess_structure!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    rows::V,
    cols::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    if !isempty(rows)
        ExaModels._obj_hess_structure!(T, m.ext.backend, m.objs, rows, cols)
        ExaModels._con_hess_structure!(T, m.ext.backend, m.cons, rows, cols)
    end
    return rows, cols
end

ExaModels._obj_hess_structure!(T, backend, ::Tuple{}, rows, cols) = nothing
function ExaModels._obj_hess_structure!(T, backend, (obj, objs...), rows, cols)
    ExaModels._obj_hess_structure!(T, backend, objs, rows, cols)
    ExaModels.shessian!(rows, cols, obj, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), T(NaN), T(NaN), 1, 0, 0, length(rows), backend)
end
ExaModels._con_hess_structure!(T, backend, ::Tuple{}, rows, cols) = nothing
function ExaModels._con_hess_structure!(T, backend, (con, cons...), rows, cols)
    ExaModels._con_hess_structure!(T, backend, cons, rows, cols)
    ExaModels.shessian!(rows, cols, con, ExaModels.NaNSource{T}(), ExaModels.NaNSource{T}(), T(NaN), T(NaN), 1, 0, 0, length(rows), backend)
end


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
_obj(backend, objbuffer, ::Tuple{}, x, θ) = nothing
function _obj(backend, objbuffer, (obj, objs...), x, θ)
    _obj(backend, objbuffer, objs, x, θ)
    if !isempty(obj.itr)
        kerf(backend)(objbuffer, obj.f, obj.itr, x, θ; ndrange = length(obj.itr))
    end
end

function ExaModels.obj!(
    m::ExaModels.BatchExaModel{T,VT,E},
    bx::AbstractMatrix,
    bf::AbstractVector,
) where {T,VT,E<:KAExtension}
    fill!(bf, zero(T))
    nb = ExaModels.get_nbatch(m)
    nvar = NLPModels.get_nvar(m)
    npar = size(m.θ, 1)
    ExaModels._obj!(bf, m.objs, vec(bx), vec(m.θ), nb, nvar, npar, m.ext.backend)
    return bf
end


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
_cons_nln!(backend, y, ::Tuple{}, x, θ) = nothing
function _cons_nln!(backend, y, (con, cons...), x, θ)
    _cons_nln!(backend, y, cons, x, θ)
    if con isa ExaModels.Constraint && !isempty(con.itr)
        kerf(backend)(y, con.f, con.itr, x, θ; ndrange = length(con.itr))
    end
end



_conaugs!(backend, y, ::Tuple{}, x, θ) = nothing
function _conaugs!(backend, y, (con, cons...), x, θ)
    _conaugs!(backend, y, cons, x, θ)
    if con isa ExaModels.ConstraintAugmentation && !isempty(con.itr)
        kerf2(backend)(y, con.f, con.itr, x, θ, con.oa; ndrange = length(con.itr))
    end
end

function ExaModels.grad!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::V,
    y::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    gradbuffer = m.ext.gradbuffer

    fill!(y, zero(eltype(y)))
    if !isempty(gradbuffer)
        fill!(gradbuffer, zero(eltype(gradbuffer)))
        _grad!(m.ext.backend, m.ext.gradbuffer, m.objs, x, m.θ)
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
_grad!(backend, y, ::Tuple{}, x, θ) = nothing
function _grad!(backend, y, (obj, objs...), x, θ)
    _grad!(backend, y, objs, x, θ)
    ExaModels.sgradient!(y, obj, x, θ, one(eltype(y)), 1, length(x), length(θ), length(y), backend)
end

function ExaModels.jac_coord!(
    m::ExaModels.AbstractExaModel{T,VT,E},
    x::V,
    y::V,
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    fill!(y, zero(eltype(y)))
    _jac_coord!(m.ext.backend, y, m.cons, x, m.θ)
    return y
end
_jac_coord!(backend, y, ::Tuple{}, x, θ) = nothing
function _jac_coord!(backend, y, (con, cons...), x, θ)
    _jac_coord!(backend, y, cons, x, θ)
    ExaModels.sjacobian!(y, nothing, con, x, θ, one(eltype(y)), 1, length(x), length(θ), length(y), backend)
end

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
    x::V,
    y::V,
    hess::V;
    obj_weight = one(eltype(y)),
) where {T,VT,E<:KAExtension,V<:AbstractVector}
    fill!(hess, zero(eltype(hess)))
    _obj_hess_coord!(m.ext.backend, hess, m.objs, x, m.θ, obj_weight)
    _con_hess_coord!(m.ext.backend, hess, m.cons, x, m.θ, y)
    return hess
end
_obj_hess_coord!(backend, hess, ::Tuple{}, x, θ, obj_weight) = nothing
function _obj_hess_coord!(backend, hess, (obj, objs...), x, θ, obj_weight)
    _obj_hess_coord!(backend, hess, objs, x, θ, obj_weight)
    ExaModels.shessian!(hess, nothing, obj, x, θ, obj_weight, zero(eltype(hess)), 1, length(x), length(θ), length(hess), backend)
end
_con_hess_coord!(backend, hess, ::Tuple{}, x, θ, y) = nothing
function _con_hess_coord!(backend, hess, (con, cons...), x, θ, y)
    _con_hess_coord!(backend, hess, cons, x, θ, y)
    ExaModels.shessian!(hess, nothing, con, x, θ, y, zero(eltype(hess)), 1, length(x), length(θ), length(y), length(hess), backend)
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

# ============================================================================
# Batch KA dispatch — single kernel launch for the whole batch
# ============================================================================

# --- Objective ---

function ExaModels._obj_eval!(bf, obj, x, θ, nb, nvar, npar, backend::KernelAbstractions.Backend)
    nitr = length(obj.itr)
    if nitr > 0
        buf = similar(x, nitr, nb)
        kerf_batch_fill(backend)(buf, obj.f, obj.itr, x, θ, nvar, npar, nitr; ndrange = nb * nitr)
        bf .+= vec(sum(buf; dims = 1))
    end
end

@kernel function kerf_batch_fill(buf, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(nvar), @Const(npar), @Const(nitr))
    I = @index(Global)
    s = (I - 1) ÷ nitr + 1
    k = (I - 1) % nitr + 1
    x_off = (s - 1) * nvar
    θ_off = (s - 1) * npar
    @inbounds buf[k, s] = f(itr[k], ExaModels.OffsetVector(x, x_off), ExaModels.OffsetVector(θ, θ_off))
end

# --- Constraints ---

function ExaModels._cons_nln_eval!(con, x, θ, g, nb, nvar, npar, ncon, backend::KernelAbstractions.Backend)
    nitr = length(con.itr)
    if nitr > 0
        kerf_con_batch(backend)(g, con.f, con.itr, x, θ, nvar, npar, ncon, nitr; ndrange = nb * nitr)
    end
end

@kernel function kerf_con_batch(g, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(nvar), @Const(npar), @Const(ncon), @Const(nitr))
    I = @index(Global)
    s = (I - 1) ÷ nitr + 1
    k = (I - 1) % nitr + 1
    x_off = (s - 1) * nvar
    θ_off = (s - 1) * npar
    g_off = (s - 1) * ncon
    @inbounds g[g_off + ExaModels.offset0(f, itr, k)] += f(itr[k], ExaModels.OffsetVector(x, x_off), ExaModels.OffsetVector(θ, θ_off))
end

# --- Gradient ---

function ExaModels.gradient!(y, f, x, θ, adj, nb::Integer, nvar::Integer, npar::Integer, backend::KernelAbstractions.Backend)
    nitr = length(f.itr)
    if nitr > 0
        kerg(backend)(y, f.f, f.itr, x, θ, adj, nvar, npar, nitr; ndrange = nb * nitr)
    end
    return y
end

@kernel function kerg(y, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj), @Const(nvar), @Const(npar), @Const(nitr))
    I = @index(Global)
    s = (I - 1) ÷ nitr + 1
    k = (I - 1) % nitr + 1
    x_off = (s - 1) * nvar
    θ_off = (s - 1) * npar
    y_off = (s - 1) * nvar
    @inbounds ExaModels.drpass(
        f(itr[k], ExaModels.AdjointNodeSource(ExaModels.OffsetVector(x, x_off)), ExaModels.OffsetVector(θ, θ_off)),
        ExaModels.OffsetVector(y, y_off),
        adj,
    )
end

# --- Sparse gradient ---

function ExaModels.sgradient!(y, f, x, θ, adj, nb::Integer, nvar::Integer, npar::Integer, nout::Integer, backend::KernelAbstractions.Backend)
    nitr = length(f.itr)
    if nitr > 0
        kerg_sparse(backend)(y, f.f, f.itr, x, θ, adj, nvar, npar, nout, nitr; ndrange = nb * nitr)
    end
    return y
end

@kernel function kerg_sparse(y, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj), @Const(nvar), @Const(npar), @Const(nout), @Const(nitr))
    I = @index(Global)
    s = (I - 1) ÷ nitr + 1
    k = (I - 1) % nitr + 1
    x_off = (s - 1) * nvar
    θ_off = (s - 1) * npar
    y_off = (s - 1) * nout
    @inbounds ExaModels.grpass(
        f(itr[k], ExaModels.AdjointNodeSource(ExaModels.OffsetVector(x, x_off)), ExaModels.OffsetVector(θ, θ_off)),
        f.comp1,
        ExaModels.OffsetVector(y, y_off),
        ExaModels.offset1(f, k),
        0,
        adj,
    )
end

# --- Jacobian ---

function ExaModels.sjacobian!(y1, y2, f, x, θ, adj, nb::Integer, nvar::Integer, npar::Integer, nout::Integer, backend::KernelAbstractions.Backend)
    nitr = length(f.itr)
    if nitr > 0
        kerj(backend)(y1, y2, f.f, f.itr, x, θ, adj, ExaModels._constraint_dims(f), nvar, npar, nout, nitr; ndrange = nb * nitr)
    end
end

@kernel function kerj(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj), @Const(dims), @Const(nvar), @Const(npar), @Const(nout), @Const(nitr))
    I = @index(Global)
    s = (I - 1) ÷ nitr + 1
    k = (I - 1) % nitr + 1
    x_off = (s - 1) * nvar
    θ_off = (s - 1) * npar
    y_off = (s - 1) * nout
    @inbounds ExaModels.jrpass(
        f(itr[k], ExaModels.AdjointNodeSource(ExaModels.OffsetVector(x, x_off)), ExaModels.OffsetVector(θ, θ_off)),
        f.comp1,
        ExaModels.offset0(f, itr, k, dims),
        ExaModels.OffsetVector(y1, y_off),
        y2,
        ExaModels.offset1(f, k),
        0,
        adj,
    )
end

# --- Hessian (objective) ---

function ExaModels.shessian!(y1, y2, f, x, θ, adj1, adj2, nb::Integer, nvar::Integer, npar::Integer, nout::Integer, backend::KernelAbstractions.Backend)
    nitr = length(f.itr)
    if nitr > 0
        kerh(backend)(y1, y2, f.f, f.itr, x, θ, adj1, adj2, nvar, npar, nout, nitr; ndrange = nb * nitr)
    end
end

@kernel function kerh(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj1), @Const(adj2), @Const(nvar), @Const(npar), @Const(nout), @Const(nitr))
    I = @index(Global)
    s = (I - 1) ÷ nitr + 1
    k = (I - 1) % nitr + 1
    x_off = (s - 1) * nvar
    θ_off = (s - 1) * npar
    y_off = (s - 1) * nout
    w_s = ExaModels._get_obj_weight(adj1, s)
    @inbounds ExaModels.hrpass0(
        f(itr[k], ExaModels.SecondAdjointNodeSource(ExaModels.OffsetVector(x, x_off)), ExaModels.OffsetVector(θ, θ_off)),
        f.comp2,
        ExaModels.OffsetVector(y1, y_off),
        y2,
        ExaModels.offset2(f, k),
        0,
        w_s,
        adj2,
    )
end

# --- Hessian (constraints) ---

function ExaModels.shessian!(y1, y2, f, x, θ, adj1s::AbstractVector, adj2, nb::Integer, nvar::Integer, npar::Integer, ncon::Integer, nout::Integer, backend::KernelAbstractions.Backend)
    nitr = length(f.itr)
    if nitr > 0
        kerh2(backend)(y1, y2, f.f, f.itr, x, θ, adj1s, adj2, ExaModels._constraint_dims(f), nvar, npar, ncon, nout, nitr; ndrange = nb * nitr)
    end
end

@kernel function kerh2(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj1s), @Const(adj2), @Const(dims), @Const(nvar), @Const(npar), @Const(ncon), @Const(nout), @Const(nitr))
    I = @index(Global)
    s = (I - 1) ÷ nitr + 1
    k = (I - 1) % nitr + 1
    x_off = (s - 1) * nvar
    θ_off = (s - 1) * npar
    y_off = (s - 1) * nout
    a_off = (s - 1) * ncon
    @inbounds ExaModels.hrpass0(
        f(itr[k], ExaModels.SecondAdjointNodeSource(ExaModels.OffsetVector(x, x_off)), ExaModels.OffsetVector(θ, θ_off)),
        f.comp2,
        ExaModels.OffsetVector(y1, y_off),
        y2,
        ExaModels.offset2(f, k),
        0,
        adj1s[a_off + ExaModels.offset0(f, itr, k, dims)],
        adj2,
    )
end

end # module ExaModelsKernelAbstractions


