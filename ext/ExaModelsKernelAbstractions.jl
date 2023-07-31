module ExaModelsKernelAbstractions

using ExaModels: ExaModels, NLPModels
using KernelAbstractions

function ExaModels.data(
    c::ExaModels.Core{T,VT,B}, gen
    ) where {T,VT,B <: KernelAbstractions.Backend}
    y = similar(c.x0, Base.@default_eltype(gen), size(gen))
    itr = getitr(gen.iter)
    
    kerdata(c.backend)(y,gen.f,itr; ndrange = length(gen))
    synchronize(c.backend)
    return y
end

function getitr(gen::UnitRange{Int64})
    return gen
end
function getitr(gen::Base.Iterators.ProductIterator{NTuple{N,UnitRange{Int64}}}) where N
end

@kernel function kerdata(y,@Const(f),@Const(itr))
    I = @index(Global)
    y[itr[I]...] = f(itr[I])
end

ExaModels.Core(backend::KernelAbstractions.CPU) = ExaModels.Core(backend=backend)

function getptr(backend,array)
    bitarray = similar(array,Bool,length(array)+1)
    kergetptr(backend)(bitarray,array; ndrange=length(array)+1)
    synchronize(backend)

    return findall(identity, bitarray)
end


struct KAExtension{T,VT <: AbstractVector{T}, VI1, VI2, B}
    backend::B
    objbuffer::VT
    gradbuffer::VT
    gsparsity::VI1
    gptr::VI2
    conbuffer::VT
    conaugsparsity::VI1
    conaugptr::VI2
end

function ExaModels.extension(w::C) where {T,VT,B <: KernelAbstractions.Backend, C <: ExaModels.Core{T,VT,B}}

    gsparsity = similar(w.x0,Tuple{Int,Int},w.nnzg)

    _grad_structure!(w.backend, w.obj, gsparsity)
    sort!(gsparsity; lt = ((i,j),(k,l)) -> i < k)
    gptr = getptr(w.backend,gsparsity)

    conaugsparsity = similar(w.x0, Tuple{Int,Int}, w.nconaug)
    _conaug_structure!(w.backend, w.con, conaugsparsity)
    length(conaugsparsity) > 0 && sort!(conaugsparsity; lt = ((i,j),(k,l)) -> i < k)
    conaugptr = getptr(w.backend,conaugsparsity)

    return KAExtension(
        w.backend,
        similar(w.x0, w.nobj),
        similar(w.x0, w.nnzg),
        gsparsity,
        gptr,
        similar(w.x0, w.nconaug),
        conaugsparsity,
        conaugptr
    )
end

function _conaug_structure!(backend, cons, sparsity)
    kers(backend)(sparsity, cons.f, cons.itr, cons.oa; ndrange = length(cons.itr))
    _conaug_structure!(backend, cons.inner, sparsity)
    synchronize(backend)
end
function _conaug_structure!(backend, cons::ExaModels.Constraint, sparsity)
    _conaug_structure!(backend, cons.inner, sparsity)
end
function _conaug_structure!(backend, cons::ExaModels.ConstraintNull, sparsity) end
@kernel function kers(sparsity,@Const(f),@Const(itr),@Const(oa))
    I = @index(Global)
    sparsity[oa+I] = (ExaModels.offset0(f,itr,I), oa+I)
end



function _grad_structure!(backend, objs, gsparsity)
    ExaModels.sgradient!(backend, gsparsity, objs, nothing, NaN16)
    _grad_structure!(backend, objs.inner, gsparsity)
    synchronize(backend)
end
function _grad_structure!(backend, objs::ExaModels.ObjectiveNull, gsparsity) end

function NLPModels.jac_structure!(m::ExaModels.Model{T,VT,E} where {T, VT, E <: KAExtension}, rows::V, cols::V) where V <: AbstractVector
    _jac_structure!(m.ext.backend, m.cons, rows, cols)
end
function _jac_structure!(backend, cons, rows, cols)
    ExaModels.sjacobian!(backend, rows, cols, cons, nothing, NaN16)
    _jac_structure!(backend, cons.inner, rows, cols)
    synchronize(backend)
end
function _jac_structure!(backend, cons::ExaModels.ConstraintNull, rows, cols) end

function NLPModels.hess_structure!(m::ExaModels.Model{T,VT,E} where {T, VT, E <: KAExtension}, rows::V, cols::V) where V <: AbstractVector
    _obj_hess_structure!(m.ext.backend, m.objs, rows, cols)
    _con_hess_structure!(m.ext.backend, m.cons, rows, cols)
end

function _obj_hess_structure!(backend, objs, rows, cols)
    ExaModels.shessian!(backend, rows, cols, objs, nothing, NaN16, NaN16)
    _obj_hess_structure!(backend, objs.inner, rows, cols)
    synchronize(backend)
end
function _obj_hess_structure!(backend, objs::ExaModels.ObjectiveNull, rows, cols) end
function _con_hess_structure!(backend, cons, rows, cols)
    ExaModels.shessian!(backend, rows, cols, cons, nothing, NaN16, NaN16)
    _con_hess_structure!(backend, cons.inner, rows, cols)
    synchronize(backend)
end
function _con_hess_structure!(backend, cons::ExaModels.ConstraintNull, rows, cols) end


function NLPModels.obj(m::ExaModels.Model{T,VT,E} where {T, VT, E <: KAExtension}, x::V) where V <: AbstractVector
    m.counters.neval_obj += 1
    m.counters.teval_obj += @elapsed begin
        _obj(m.ext.backend,m.ext.objbuffer,m.objs,x)
        result = sum(m.ext.objbuffer)
    end
    return result
end
function _obj(backend,objbuffer, obj, x)
    kerf(backend)(objbuffer, obj.f, obj.itr, x; ndrange=length(obj.itr))
    _obj(backend,objbuffer,obj.inner,x)
    synchronize(backend)
end
function _obj(backend, objbuffer, f::ExaModels.ObjectiveNull, x) end

function NLPModels.cons!(m::ExaModels.Model{T,VT,E} where {T, VT, E <: KAExtension}, x::V, y::V) where V <: AbstractVector
    m.counters.neval_cons += 1    
    m.counters.teval_cons += @elapsed begin
        _cons!(m.ext.backend,y,m.cons,x)
        _conaugs!(m.ext.backend,m.ext.conbuffer,m.cons,x)
        
        if length(m.ext.conaugptr) > 1
            compress_to_dense(m.ext.backend)(
                y, m.ext.conbuffer, m.ext.conaugptr, m.ext.conaugsparsity; ndrange=length(m.ext.conaugptr)-1
            )
            synchronize(m.ext.backend)
        end
    end
end
function _cons!(backend,y, con::ExaModels.Constraint, x)
    kerf(backend)(y, con.f, con.itr, x; ndrange=length(con.itr))
    _cons!(backend,y,con.inner,x)
    synchronize(backend)
end
function _cons!(backend, y, con::ExaModels.ConstraintNull, x) end
function _cons!(backend,y, con::ExaModels.ConstraintAug, x)
    _cons!(backend,y,con.inner,x)
end 



function _conaugs!(backend,y, con::ExaModels.ConstraintAug, x)
    kerf2(backend)(y, con.f, con.itr, x, con.oa; ndrange=length(con.itr))
    _conaugs!(backend,y,con.inner,x)
    synchronize(backend)
end
function _conaugs!(backend,y, con::ExaModels.Constraint, x)
    _conaugs!(backend,y,con.inner,x)
end
function _conaugs!(backend, y, con::ExaModels.ConstraintNull, x) end

function NLPModels.grad!(m::ExaModels.Model{T,VT,E} where {T, VT, E <: KAExtension}, x::V, y::V) where V <: AbstractVector
    m.counters.neval_grad += 1
    m.counters.teval_grad += @elapsed begin
        gradbuffer = m.ext.gradbuffer
        fill!(gradbuffer,zero(eltype(gradbuffer)))
        _grad!(m.ext.backend,m.ext.gradbuffer,m.objs,x)
        
        fill!(y,zero(eltype(y)))
        compress_to_dense(m.ext.backend)(
            y, gradbuffer, m.ext.gptr, m.ext.gsparsity; ndrange=length(m.ext.gptr)-1
        )
        synchronize(m.ext.backend)
    end
end
function _grad!(backend,y,objs,x)
    ExaModels.sgradient!(backend,y,objs,x,one(eltype(y)))
    _grad!(backend,y,objs.inner,x)
    synchronize(backend)
end
function _grad!(backend,y,objs::ExaModels.ObjectiveNull,x) end

function NLPModels.jac_coord!(m::ExaModels.Model{T,VT,E} where {T, VT, E <: KAExtension}, x::V, y::V) where V <: AbstractVector
    m.counters.neval_jac += 1
    m.counters.teval_jac += @elapsed begin
        fill!(y,zero(eltype(y)))
        _jac_coord!(m.ext.backend, y, m.cons, x)
    end
end
function _jac_coord!(backend, y, cons, x)
    ExaModels.sjacobian!(backend, y, nothing, cons, x, one(eltype(y)))
    _jac_coord!(backend, y, cons.inner, x)
    synchronize(backend)    
end
function _jac_coord!(backend, y, cons::ExaModels.ConstraintNull, x) end

function NLPModels.hess_coord!(m::ExaModels.Model{T,VT,E} where {T, VT, E <: KAExtension}, x::V, y::V, hess::V; obj_weight = one(eltype(y))) where V <: AbstractVector
    m.counters.neval_hess += 1
    m.counters.teval_hess += @elapsed begin
        fill!(hess,zero(eltype(hess)))
        _obj_hess_coord!(m.ext.backend, hess, m.objs, x, obj_weight)
        _con_hess_coord!(m.ext.backend, hess, m.cons, x, y)
    end
end
function _obj_hess_coord!(backend, hess, objs, x, obj_weight)
    ExaModels.shessian!(backend, hess, nothing, objs, x, obj_weight, zero(eltype(hess)))
    _obj_hess_coord!(backend, hess, objs.inner, x, obj_weight)
    synchronize(backend)
end
function _obj_hess_coord!(backend, hess, objs::ExaModels.ObjectiveNull, x, obj_weight) end
function _con_hess_coord!(backend, hess, cons, x, y)
    ExaModels.shessian!(backend, hess, nothing, cons, x, y, zero(eltype(hess))) 
    _con_hess_coord!(backend, hess, cons.inner, x, y)
    synchronize(backend)
end
function _con_hess_coord!(backend, hess, cons::ExaModels.ConstraintNull, x, y) end


function ExaModels.sgradient!(backend::B,y,f,x,adj) where B <: KernelAbstractions.Backend
    return kerg(backend)(y, f.f, f.itr, x, adj; ndrange = length(f.itr))
end

function ExaModels.sjacobian!(backend::B, y1, y2, f, x, adj) where B <: KernelAbstractions.Backend
    return kerj(backend)(y1, y2, f.f, f.itr, x, adj; ndrange=length(f.itr))
end

function ExaModels.shessian!(backend::B, y1, y2, f, x, adj, adj2) where B <: KernelAbstractions.Backend
    return kerh(backend)(y1, y2, f.f, f.itr, x, adj, adj2; ndrange=length(f.itr))
end

function ExaModels.shessian!(backend::B, y1, y2, f, x, adj::V, adj2) where {B <: KernelAbstractions.Backend, V <: AbstractVector}
    return kerh2(backend)(y1, y2, f.f, f.itr, x, adj, adj2; ndrange=length(f.itr))
end

@kernel function kerh(y1,y2,@Const(f),@Const(itr),@Const(x),@Const(adj1),@Const(adj2))
    I = @index(Global)
    ExaModels.hrpass0(f.f(itr[I], ExaModels.SecondAdjointNodeSource(x)), f.comp2, y1, y2, ExaModels.offset2(f,I), 0, adj1, adj2)
end

@kernel function kerh2(y1,y2,@Const(f),@Const(itr),@Const(x),@Const(adjs1),@Const(adj2))
    I = @index(Global)
    ExaModels.hrpass0(f.f(itr[I], ExaModels.SecondAdjointNodeSource(x)), f.comp2, y1, y2, ExaModels.offset2(f,I), 0, adjs1[ExaModels.offset0(f,itr,I)], adj2)
end

@kernel function kerj(y1,y2,@Const(f),@Const(itr),@Const(x),@Const(adj))
    I = @index(Global)
    ExaModels.jrpass(f.f(itr[I], ExaModels.AdjointNodeSource(x)), f.comp1, ExaModels.offset0(f,itr,I), y1, y2, ExaModels.offset1(f,I), 0, adj)
end

@kernel function kerg(y,@Const(f),@Const(itr),@Const(x),@Const(adj))
    I = @index(Global)
    ExaModels.grpass(f.f(itr[I], ExaModels.AdjointNodeSource(x)), f.comp1, y, ExaModels.offset1(f,I), 0, adj)
end

@kernel function kerf(y,@Const(f),@Const(itr),@Const(x))
    I = @index(Global)
    y[ExaModels.offset0(f,itr,I)] = f.f(itr[I],x)
end
@kernel function kerf2(y,@Const(f),@Const(itr),@Const(x),@Const(oa))
    I = @index(Global)
    y[oa+I] = f.f(itr[I],x)
end


@kernel function compress_to_dense(y,@Const(y0),@Const(ptr),@Const(sparsity))
    I = @index(Global)
    for j = ptr[I]:ptr[I+1]-1
        (k,l) = sparsity[j]
        y[k] += y0[l]
    end
end

@kernel function kergetptr(bitarray,@Const(array))
    I = @index(Global)
    if I == 1
        bitarray[I] = true
    elseif I == length(array)+1
        bitarray[I] = true
    else
        i0,j0 = array[I-1]
        i1,j1 = array[I]
        
        if i0 != i1
            bitarray[I] = true
        else
            bitarray[I] = false
        end
    end
end

end # module ExaModelsKernelAbstractions
