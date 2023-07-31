Base.@kwdef mutable struct ADBenchmarkStats
    obj_cnt::Int = 0
    cons_cnt::Int = 0
    grad_cnt::Int = 0
    jac_coord_cnt::Int = 0
    hess_coord_cnt::Int = 0
    jac_structure_cnt::Int = 0
    hess_structure_cnt::Int = 0
    obj_time::Float64 = 0.
    cons_time::Float64 = 0.
    grad_time::Float64 = 0.
    jac_coord_time::Float64 = 0.
    hess_coord_time::Float64 = 0.
    jac_structure_time::Float64 = 0.
    hess_structure_time::Float64 = 0.
end

struct ADBenchmarkModel{T,VT,I <: NLPModels.AbstractNLPModel{T,VT}} <: NLPModels.AbstractNLPModel{T,VT}
    inner::I
    meta::NLPModels.AbstractNLPModelMeta{T,VT}
    counters::NLPModels.Counters
    stats::ADBenchmarkStats
end

function ADBenchmarkModel(m)
    return ADBenchmarkModel(
        m,
        m.meta,
        m.counters,
        ADBenchmarkStats()
    )
end
function ADBenchmarkModel(c::ExaModels.Core; kwargs...)
    m = ExaModels.Model(c;kwargs...)
    return ADBenchmarkModel(m)
end

function NLPModels.jac_structure!(
    m::M,
    rows::V,
    cols::V
    ) where {M <: ADBenchmarkModel, V <: AbstractVector}

    m.stats.jac_structure_cnt += 1
    t = time()
    NLPModels.jac_structure!(m.inner, rows, cols)
    m.stats.jac_structure_time += time() - t
end

function NLPModels.hess_structure!(
    m::M,
    rows::V,
    cols::V
    ) where {M <: ADBenchmarkModel, V <: AbstractVector}

    m.stats.hess_structure_cnt += 1
    t = time()
    NLPModels.hess_structure!(m.inner, rows, cols)
    m.stats.hess_structure_time += time() - t
end

function NLPModels.obj(
    m::M,
    x::V
    ) where {M <: ADBenchmarkModel, V <: AbstractVector}

    m.stats.obj_cnt += 1
    t = time()
    o = NLPModels.obj(m.inner, x)
    m.stats.obj_time += time() - t
    return o
end
function NLPModels.cons!(
    m::M,
    x::V,
    g::V
    ) where {M <: ADBenchmarkModel, V <: AbstractVector}

    m.stats.cons_cnt += 1
    t = time()
    NLPModels.cons!(m.inner, x, g)
    m.stats.cons_time += time() - t
    return 
end
function NLPModels.grad!(
    m::M,
    x::V,
    f::V
    ) where {M <: ADBenchmarkModel, V <: AbstractVector}

    m.stats.grad_cnt += 1
    t = time()
    NLPModels.grad!(m.inner, x, f)
    m.stats.grad_time += time() - t
    return
end
function NLPModels.jac_coord!(
    m::M,
    x::V,
    jac::V
    ) where {M <: ADBenchmarkModel, V <: AbstractVector}

    m.stats.jac_coord_cnt += 1
    t = time()
    NLPModels.jac_coord!(m.inner, x, jac)
    m.stats.jac_coord_time += time() - t
    return
end
function NLPModels.hess_coord!(
    m::M,
    x::V,
    y::V,
    hess::V;
    obj_weight = one(eltype(x))
    ) where {M <: ADBenchmarkModel, V <: AbstractVector}

    m.stats.hess_coord_cnt += 1
    t = time()
    NLPModels.hess_coord!(m.inner, x, y, hess; obj_weight=obj_weight)
    m.stats.hess_coord_time += time() - t
    return
end    


using Printf
     
function Base.print(io::IO,e::ADBenchmarkModel)
    tot = 0.
    for f in fieldnames(ADBenchmarkStats)
        if endswith(string(f),"cnt")
            @printf "%20s:  %13i times\n" f getfield(e.stats,f)
        else
            t = getfield(e.stats,f)
            @printf "%20s:  %13.6f secs\n" f t
            tot += t
        end
    end
    println("------------------------------------------")
    @printf "       total AD time:  %13.6f secs\n" tot
end
Base.show(io::IO,::MIME"text/plain",e::ADBenchmarkModel) = Base.print(io,e);
# Base.string(e::ADBenchmarkModel) = ""


