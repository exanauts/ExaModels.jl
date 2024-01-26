# TimedNLPModels

Base.@kwdef mutable struct CallbackStats
    obj_cnt::Int = 0
    cons_cnt::Int = 0
    grad_cnt::Int = 0
    jac_coord_cnt::Int = 0
    hess_coord_cnt::Int = 0
    jac_structure_cnt::Int = 0
    hess_structure_cnt::Int = 0
    obj_time::Float64 = 0.0
    cons_time::Float64 = 0.0
    grad_time::Float64 = 0.0
    jac_coord_time::Float64 = 0.0
    hess_coord_time::Float64 = 0.0
    jac_structure_time::Float64 = 0.0
    hess_structure_time::Float64 = 0.0
end

struct TimedNLPModel{T,VT,I<:NLPModels.AbstractNLPModel{T,VT}} <:
    NLPModels.AbstractNLPModel{T,VT}
    inner::I
    meta::NLPModels.AbstractNLPModelMeta{T,VT}
    stats::CallbackStats
    counters::NLPModels.Counters
end

function TimedNLPModel(m)
    return TimedNLPModel(m, m.meta, CallbackStats(), NLPModels.Counters())
end
function TimedNLPModel(c::ExaModels.ExaCore; kwargs...)
    m = ExaModels.Model(c; kwargs...)
    return TimedNLPModel(m)
end

function NLPModels.jac_structure!(
    m::M,
    rows::V,
    cols::V,
    ) where {M<:TimedNLPModel,V<:AbstractVector}

    m.stats.jac_structure_cnt += 1
    t = time()
    NLPModels.jac_structure!(m.inner, rows, cols)
    m.stats.jac_structure_time += time() - t
end

function NLPModels.hess_structure!(
    m::M,
    rows::V,
    cols::V,
    ) where {M<:TimedNLPModel,V<:AbstractVector}

    m.stats.hess_structure_cnt += 1
    t = time()
    NLPModels.hess_structure!(m.inner, rows, cols)
    m.stats.hess_structure_time += time() - t
end

function NLPModels.obj(m::TimedNLPModel, x::AbstractVector)

    m.stats.obj_cnt += 1
    t = time()
    o = NLPModels.obj(m.inner, x)
    m.stats.obj_time += time() - t
    return o
end
function NLPModels.cons!(m::TimedNLPModel, x::AbstractVector, g::AbstractVector)

    m.stats.cons_cnt += 1
    t = time()
    NLPModels.cons!(m.inner, x, g)
    m.stats.cons_time += time() - t
    return
end
function NLPModels.grad!(m::TimedNLPModel, x::AbstractVector, f::AbstractVector)

    m.stats.grad_cnt += 1
    t = time()
    NLPModels.grad!(m.inner, x, f)
    m.stats.grad_time += time() - t
    return
end
function NLPModels.jac_coord!(m::TimedNLPModel, x::AbstractVector, jac::AbstractVector)

    m.stats.jac_coord_cnt += 1
    t = time()
    NLPModels.jac_coord!(m.inner, x, jac)
    m.stats.jac_coord_time += time() - t
    return
end
function NLPModels.hess_coord!(
    m::TimedNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
    )

    m.stats.hess_coord_cnt += 1
    t = time()
    NLPModels.hess_coord!(m.inner, x, y, hess; obj_weight = obj_weight)
    m.stats.hess_coord_time += time() - t
    return
end

function Base.print(io::IO, e::TimedNLPModel)
    tot = 0.0
    for f in fieldnames(CallbackStats)
        if endswith(string(f), "cnt")
            Printf.@printf "%20s:  %13i times\n" f getfield(e.stats, f)
        else
            t = getfield(e.stats, f)
            Printf.@printf "%20s:  %13.6f secs\n" f t
            tot += t
        end
    end
    println("------------------------------------------")
    Printf.@printf "       total AD time:  %13.6f secs\n" tot
end
Base.show(io::IO, ::MIME"text/plain", e::TimedNLPModel) = Base.print(io, e);


# CompressedNLPModels

struct CompressedNLPModel{
	  T,
	  VT <: AbstractVector{T},
	  VI <: AbstractVector{Int},
	  VI2 <: AbstractVector{Tuple{Tuple{Int, Int}, Int}},
	  M <: NLPModels.AbstractNLPModel{T, VT},
    } <: NLPModels.AbstractNLPModel{T, VT}

	  inner::M
	  jptr::VI
	  jsparsity::VI2
	  hptr::VI
	  hsparsity::VI2
	  buffer::VT

	  meta::NLPModels.NLPModelMeta{T, VT}
	  counters::NLPModels.Counters
end

function getptr(array)
	  return push!(
		    pushfirst!(
			      findall(
				        _is_sparsity_not_equal.(@view(array[1:end-1]), @view(array[2:end])),
			      ) .+= 1,
			      1,
		    ),
		    length(array) + 1,
	  )
end
_is_sparsity_not_equal(a,b) = first(a) != first(b)

function CompressedNLPModel(m)

	  nnzj = NLPModels.get_nnzj(m)
	  Ibuffer = Vector{Int}(undef, nnzj)
	  Jbuffer = Vector{Int}(undef, nnzj)
	  NLPModels.jac_structure!(m, Ibuffer, Jbuffer)

	  jsparsity = map(
		    (k, i, j) -> ((j,i), k),
		    1:nnzj,
		    Ibuffer,
		    Jbuffer,
	  )
	  sort!(jsparsity; lt = (a,b) -> a[1] < b[1])
	  jptr = getptr(jsparsity)

	  nnzh = NLPModels.get_nnzh(m)
	  resize!(Ibuffer, nnzh)
	  resize!(Jbuffer, nnzh)
	  NLPModels.hess_structure!(m, Ibuffer, Jbuffer)

	  hsparsity = map(
		    (k, i, j) -> ((j,i), k),
		    1:nnzh,
		    Ibuffer,
		    Jbuffer,
	  )
	  sort!(hsparsity; lt = (a,b) -> a[1] < b[1])
	  hptr = getptr(hsparsity)

	  buffer = similar(m.meta.x0, max(nnzj, nnzh))

	  meta = NLPModels.NLPModelMeta(
		    m.meta.nvar,
		    ncon = m.meta.ncon,
		    nnzj = length(jptr)-1,
		    nnzh = length(hptr)-1,
		    x0 = m.meta.x0,
		    lvar = m.meta.lvar,
		    uvar = m.meta.uvar,
		    y0 = m.meta.y0,
		    lcon = m.meta.lcon,
		    ucon = m.meta.ucon,
	  )

	  counters = NLPModels.Counters()

	  return CompressedNLPModel(
		    m,
		    jptr,
		    jsparsity,
		    hptr,
		    hsparsity,
		    buffer,
		    meta,
		    counters,
	  )
end

function NLPModels.obj(m::CompressedNLPModel, x::AbstractVector)
	  NLPModels.obj(m.inner, x)
end

function NLPModels.grad!(m::CompressedNLPModel, x::AbstractVector, y::AbstractVector)
	  NLPModels.grad!(m.inner, x, y)
end

function NLPModels.cons!(m::CompressedNLPModel, x::AbstractVector, g::AbstractVector)
	  NLPModels.cons!(m.inner, x, g)
end

function NLPModels.jac_coord!(m::CompressedNLPModel, x::AbstractVector, j::AbstractVector)
	  NLPModels.jac_coord!(m.inner, x, m.buffer)
	  _compress!(j, m.buffer, m.jptr, m.jsparsity)
end

function NLPModels.hess_coord!(m::CompressedNLPModel, x::AbstractVector, y::AbstractVector, h::AbstractVector; obj_weight = 1.0)
	  NLPModels.hess_coord!(m.inner, x, y, m.buffer; obj_weight = obj_weight)
	  _compress!(h, m.buffer, m.hptr, m.hsparsity)
end

function NLPModels.jac_structure!(m::CompressedNLPModel, I::AbstractVector, J::AbstractVector)
	  _structure!(I, J, m.jptr, m.jsparsity)
end

function NLPModels.hess_structure!(m::CompressedNLPModel, I::AbstractVector, J::AbstractVector)
	  _structure!(I, J, m.hptr, m.hsparsity)
end

function _compress!(V, buffer, ptr, sparsity)
    fill!(V, zero(eltype(V)))
	  @simd for i in 1:length(ptr)-1
		    for j in ptr[i]:ptr[i+1]-1
            V[i] += buffer[sparsity[j][2]]
        end
	  end
end

function _structure!(I, J, ptr, sparsity)
	  @simd for i in 1:length(ptr)-1
		    J[i], I[i] = sparsity[ptr[i]][1]
	  end
end

export TimedNLPModel, CompressedNLPModel





