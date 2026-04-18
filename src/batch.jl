# ============================================================================
# Batch ExaModel — dispatch on VT <: AbstractMatrix
# ============================================================================
#
# Following the same pattern as TwoStageExaModel:
# - BatchExaModelTag stored in ExaCore's tag field
# - BatchExaCore = ExaCore{T, <:AbstractMatrix, B, <:BatchExaModelTag}
# - EachInstance() marker for per-instance declarations
# - ExaModel(c) handles both batch and non-batch via dispatch

import .BatchNLPModels: obj!

# ============================================================================
# Tag and marker types
# ============================================================================

"""
    EachInstance

Marker type used with [`add_var`](@ref), [`add_par`](@ref), and [`add_con`](@ref)
to indicate that the declaration is replicated for each instance in a batch model.

## Example
```julia
core = BatchExaCore(3)
c, v = add_var(core, EachInstance(), 10)
c, _ = add_obj(c, v[i, s]^2 for i in 1:10, s in 1:3)
```
"""
struct EachInstance end

struct BatchExaModelTag <: AbstractExaModelTag
    nbatch::Int
end

# ============================================================================
# Type aliases
# ============================================================================

"""
    BatchExaCore{T,VT,B}

Type alias for an [`ExaCore`](@ref) whose `tag` is a [`BatchExaModelTag`]
and whose storage arrays are matrices (columns = instances).
"""
const BatchExaCore{T,VT,B} = ExaCore{T,VT,B,<:BatchExaModelTag}

"""
    BatchExaModel{T,VT,E,V,P,O,C,R,M}

Type alias for an [`ExaModel`](@ref) built from a [`BatchExaCore`](@ref).
"""
const BatchExaModel{T,VT,E,V,P,O,C,R,M} = ExaModel{T,VT,E,V,P,O,C,<:BatchExaModelTag,R,M}

# ============================================================================
# get_nbatch
# ============================================================================

get_nbatch(m::BatchExaModel) = m.tag.nbatch
get_nbatch(c::BatchExaCore) = c.tag.nbatch
get_nbatch(m::AbstractExaModel) = 1

# ============================================================================
# _meta_dims — per-instance dimensions for batch
# ============================================================================

function _meta_dims(c::C) where {T, VT <: AbstractArray{T}, C <: BatchExaCore{T, VT}}
    nb = c.tag.nbatch
    return (c.nvar ÷ nb, c.ncon ÷ nb, c.nnzj ÷ nb, c.nnzh ÷ nb)
end

# ============================================================================
# BatchExaCore constructor
# ============================================================================

"""
    BatchExaCore(nbatch; T = Float64, backend = nothing, kwargs...)

Create an [`ExaCore`](@ref) for building batch optimization models with
`nbatch` independent instances.

Storage arrays (`x0`, `lvar`, `uvar`, `θ`, `y0`, `lcon`, `ucon`) are
`Matrix{T}` with `nbatch` columns. Use [`add_var`](@ref), [`add_par`](@ref),
[`add_obj`](@ref), and [`add_con`](@ref) with [`EachInstance()`](@ref) to
declare per-instance components.

## Example
```julia
core = BatchExaCore(3)
c, v = add_var(core, EachInstance(), 10; start = 1.0, lvar = 0.0, uvar = 10.0)
nb = get_nbatch(c)
c, _ = add_obj(c, v[i, s]^2 for i in 1:10, s in 1:nb)
model = ExaModel(c)
```
"""
function BatchExaCore(nbatch::Integer; T::Type{<:AbstractFloat} = Float64, backend = nothing, kwargs...)
    x0 = convert_array(zeros(T, 0, nbatch), backend)
    return ExaCore(
        :Generic,
        backend,
        (),   # var
        (),   # par
        (),   # obj
        (),   # cons
        0, 0, 0, 0, 0,  # nvar, npar, ncon, nconaug, nobj
        0, 0, 0, 0,     # nnzc, nnzg, nnzj, nnzh
        x0,
        similar(x0),    # θ
        similar(x0),    # lvar
        similar(x0),    # uvar
        similar(x0),    # y0
        similar(x0),    # lcon
        similar(x0),    # ucon
        true,           # minimize
        BatchExaModelTag(nbatch),
        (;),            # refs
    )
end

# ============================================================================
# add_var / add_par / add_con for BatchExaCore
# ============================================================================

"""
    add_var(core::BatchExaCore, ::EachInstance, dims...; start = 0, lvar = -Inf, uvar = Inf, name = nothing)

Add per-instance variables to a batch core. Creates `prod(dims) * nbatch`
variables total — one copy of the block per instance. The returned `Variable`
has dimensions `(dims..., nbatch)`.
"""
function add_var(
    c::C,
    ::EachInstance,
    ns...;
    name = nothing,
    start = zero(T),
    lvar = T(-Inf),
    uvar = T(Inf),
) where {T, VT <: AbstractArray{T}, C <: BatchExaCore{T, VT}}
    nbatch = c.tag.nbatch
    len_per = total(ns)       # per-instance count
    len_total = len_per * nbatch  # fused total
    nvar = c.nvar + len_total

    # Append per-instance rows to matrix storage
    x0 = append!(c.backend, c.x0, start, len_per)
    lv = append!(c.backend, c.lvar, lvar, len_per)
    uv = append!(c.backend, c.uvar, uvar, len_per)

    v = Variable((ns..., nbatch), len_total, c.nvar, _val_name(name), nothing)
    (ExaCore(c; var = (v, c.var...), nvar = nvar, x0 = x0, lvar = lv, uvar = uv,
             refs = add_refs(c.refs, name, v)), v)
end

"""
    add_par(core::BatchExaCore, ::EachInstance, value::AbstractArray; name = nothing)
    add_par(core::BatchExaCore, ::EachInstance, dims...; name = nothing, start = 0)

Add per-instance parameters to a batch core. The parameter values are
replicated for each instance.
"""
function add_par(
    c::C,
    ::EachInstance,
    value::AbstractArray;
    name = nothing,
) where {T, VT <: AbstractArray{T}, C <: BatchExaCore{T, VT}}
    nbatch = c.tag.nbatch
    len_per = length(value)
    len_total = len_per * nbatch
    npar = c.npar + len_total
    θ = append!(c.backend, c.θ, value, len_per)
    p = Parameter((Base.size(value)..., nbatch), len_total, c.npar, nothing)
    (ExaCore(c; par = (p, c.par...), θ = θ, npar = npar, refs = add_refs(c.refs, name, p)), p)
end

function add_par(
    c::C,
    ::EachInstance,
    ns...;
    name = nothing,
    start = zero(T),
) where {T, VT <: AbstractArray{T}, C <: BatchExaCore{T, VT}}
    nbatch = c.tag.nbatch
    len_per = total(ns)
    len_total = len_per * nbatch
    npar = c.npar + len_total
    θ = append!(c.backend, c.θ, start, len_per)
    p = Parameter((ns..., nbatch), len_total, c.npar, nothing)
    (ExaCore(c; par = (p, c.par...), θ = θ, npar = npar, refs = add_refs(c.refs, name, p)), p)
end

"""
    add_con(core::BatchExaCore, ::EachInstance, gen; start = 0, lcon = 0, ucon = 0, name = nothing)

Add per-instance constraints to a batch core.
"""
function add_con(
    c::C,
    ::EachInstance,
    ns...;
    name = nothing,
    tag = nothing,
    start = zero(T),
    lcon = zero(T),
    ucon = zero(T),
) where {T, VT <: AbstractArray{T}, C <: BatchExaCore{T, VT}}
    gen = _get_generator(ns)
    dims = _get_con_dims(ns)
    gen = _adapt_gen(gen)
    f = _simdfunction(T, gen.f(DataSource()), c.ncon, c.nnzj, c.nnzh)
    pars = gen.iter

    nitr = length(pars)
    o = c.ncon
    ncon = c.ncon + nitr
    nnzj = c.nnzj + nitr * f.o1step
    nnzh = c.nnzh + nitr * f.o2step

    # Append per-instance rows to matrix storage
    nbatch = c.tag.nbatch
    nitr_per = nitr ÷ nbatch
    y0 = append!(c.backend, c.y0, start, nitr_per)
    lc = append!(c.backend, c.lcon, lcon, nitr_per)
    uc = append!(c.backend, c.ucon, ucon, nitr_per)

    con = Constraint(f, convert_array(pars, c.backend), o, dims, nothing)
    (ExaCore(c; ncon = ncon, nnzj = nnzj, nnzh = nnzh, y0 = y0, lcon = lc, ucon = uc,
             cons = (con, c.cons...), refs = add_refs(c.refs, name, con)), con)
end

# ============================================================================
# Batch show
# ============================================================================

function Base.show(io::IO, m::BatchExaModel{T, VT}) where {T, VT}
    nb = get_nbatch(m)
    nv = NLPModels.get_nvar(m)
    nc = NLPModels.get_ncon(m)
    println(io, "An ExaModel{$T, $VT, ...} (batch)")
    println(io, "  Instances: $nb")
    println(io, "  Variables per instance: $nv")
    println(io, "  Constraints per instance: $nc")
    println(io, "  Total variables: $(nb * nv)")
    return println(io, "  Total constraints: $(nb * nc)")
end

# ============================================================================
# Accessors
# ============================================================================

"""
    get_model(model)

For batch models, returns a flat (Vector-based) ExaModel for direct NLPModels
interface usage (e.g. MadNLP/Ipopt). For regular models, returns self.
"""
get_model(model::ExaModel) = model
function get_model(model::BatchExaModel)
    nb = get_nbatch(model)
    nvar = NLPModels.get_nvar(model) * nb
    ncon = NLPModels.get_ncon(model) * nb
    nnzj = NLPModels.get_nnzj(model) * nb
    nnzh = NLPModels.get_nnzh(model) * nb
    meta = BatchNLPModelMeta(
        nvar, vec(model.meta.x0), vec(model.meta.lvar), vec(model.meta.uvar),
        ncon, vec(model.meta.y0), vec(model.meta.lcon), vec(model.meta.ucon);
        nnzj = nnzj, nnzh = nnzh, minimize = model.meta.minimize,
    )
    return ExaModel(
        model.name, model.vars, model.pars, model.objs, model.cons,
        vec(model.θ), meta, NLPModels.Counters(), nothing, nothing, model.refs,
    )
end

"""
    var_indices(model, i) -> UnitRange

Variable index range for instance `i` in the fused model's global variable vector.
"""
var_indices(model::BatchExaModel, i::Int) =
    ((i - 1) * NLPModels.get_nvar(model) + 1):(i * NLPModels.get_nvar(model))

"""
    cons_block_indices(model, i) -> UnitRange

Constraint index range for instance `i` in the fused model's global constraint vector.
"""
cons_block_indices(model::BatchExaModel, i::Int) =
    ((i - 1) * NLPModels.get_ncon(model) + 1):(i * NLPModels.get_ncon(model))

# ============================================================================
# Objective buffer evaluation
# ============================================================================

_count_nobj(::Tuple{}) = 0
_count_nobj(objs::Tuple) = length(first(objs).itr) + _count_nobj(Base.tail(objs))

function _eval_objbuffer!(objbuffer, objs::Tuple, x, θ)
    _eval_objbuffer!(objbuffer, Base.tail(objs), x, θ)
    o = first(objs)
    for i in eachindex(o.itr)
        objbuffer[offset0(o, i)] = o.f(o.itr[i], x, θ)
    end
    return
end
_eval_objbuffer!(objbuffer, ::Tuple{}, x, θ) = nothing

# ============================================================================
# Batch API: obj / obj!
# ============================================================================

function obj!(m::BatchExaModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
    nb = get_nbatch(m)
    nobj_total = _count_nobj(m.objs)
    nobj_per = nobj_total ÷ nb
    objbuffer = Vector{T}(undef, nobj_total)
    _eval_objbuffer!(objbuffer, m.objs, vec(bx), vec(m.θ))
    obj_mat = reshape(objbuffer, nobj_per, nb)
    bf .= vec(sum(obj_mat; dims = 1))
    return bf
end

function obj(m::BatchExaModel{T}, bx::AbstractMatrix) where {T}
    bf = Vector{T}(undef, get_nbatch(m))
    obj!(m, bx, bf)
    return bf
end

# ============================================================================
# Batch API: grad!
# ============================================================================

function NLPModels.grad!(m::BatchExaModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
    fill!(vec(bg), zero(T))
    _grad!(m.objs, vec(bx), vec(m.θ), vec(bg))
    return bg
end

# ============================================================================
# Batch API: cons!
# ============================================================================

function NLPModels.cons!(m::BatchExaModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where {T}
    fill!(vec(bc), zero(T))
    _cons_nln!(m.cons, vec(bx), vec(m.θ), vec(bc))
    return bc
end

# ============================================================================
# Batch API: jac_structure! / jac_coord!
# ============================================================================

function NLPModels.jac_structure!(
    m::BatchExaModel{T},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T}
    nb = get_nbatch(m)
    nnzj_per = NLPModels.get_nnzj(m)
    total_nnzj = nnzj_per * nb
    full_rows = zeros(Int, total_nnzj)
    full_cols = zeros(Int, total_nnzj)
    _jac_structure!(T, m.cons, full_rows, full_cols)
    for k in 1:nnzj_per
        rows[k] = full_rows[k]
        cols[k] = full_cols[k]
    end
    return rows, cols
end

function NLPModels.jac_coord!(m::BatchExaModel{T}, bx::AbstractMatrix, jvals::AbstractVector) where {T}
    fill!(jvals, zero(T))
    _jac_coord!(m.cons, vec(bx), vec(m.θ), jvals)
    return jvals
end

# ============================================================================
# Batch API: hess_structure! / hess_coord!
# ============================================================================

# Helpers for hess permutation
_count_hess_nnz(::Tuple{}) = 0
function _count_hess_nnz(objs::Tuple)
    o = first(objs)
    return o.f.o2step * length(o.itr) + _count_hess_nnz(Base.tail(objs))
end

function _build_hess_perm(ns, nnzh_obj_per, nnzh_con_per)
    nnzh_per = nnzh_obj_per + nnzh_con_per
    perm = Vector{Int}(undef, ns * nnzh_per)
    for s in 1:ns
        base = (s - 1) * nnzh_per
        for k in 1:nnzh_obj_per
            perm[base + k] = (s - 1) * nnzh_obj_per + k
        end
        for k in 1:nnzh_con_per
            perm[base + nnzh_obj_per + k] =
                ns * nnzh_obj_per + (s - 1) * nnzh_con_per + k
        end
    end
    return perm
end

function _batch_hess_perm(m::BatchExaModel)
    nb = get_nbatch(m)
    nnzh_obj_per = _count_hess_nnz(m.objs) ÷ nb
    nnzh_con_per = _count_hess_nnz(m.cons) ÷ nb
    return _build_hess_perm(nb, nnzh_obj_per, nnzh_con_per)
end

function NLPModels.hess_structure!(
    m::BatchExaModel{T},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T}
    nb = get_nbatch(m)
    nnzh_per = NLPModels.get_nnzh(m)
    total_nnzh = nnzh_per * nb
    full_rows = zeros(Int, total_nnzh)
    full_cols = zeros(Int, total_nnzh)
    _obj_hess_structure!(T, m.objs, full_rows, full_cols)
    _con_hess_structure!(T, m.cons, full_rows, full_cols)
    perm = _batch_hess_perm(m)
    for k in 1:nnzh_per
        idx = perm[k]
        rows[k] = full_rows[idx]
        cols[k] = full_cols[idx]
    end
    return rows, cols
end

function NLPModels.hess_coord!(
    m::BatchExaModel{T},
    bx::AbstractMatrix,
    by::AbstractMatrix,
    bobj_weight::AbstractVector,
    hvals::AbstractVector,
) where {T}
    x_flat = vec(bx)
    y_flat = vec(by)
    nb = get_nbatch(m)
    nh = NLPModels.get_nnzh(m)
    total_nnzh = nh * nb
    perm = _batch_hess_perm(m)
    hess_buffer = Vector{T}(undef, total_nnzh)

    bobj_weight_cpu = Array(bobj_weight)
    if allequal(bobj_weight_cpu)
        w = first(bobj_weight_cpu)
        fill!(hess_buffer, zero(T))
        _obj_hess_coord!(m.objs, x_flat, m.θ, hess_buffer, w)
        _con_hess_coord!(m.cons, x_flat, m.θ, y_flat, hess_buffer, w)
        hvals .= hess_buffer[perm]
    else
        hess_obj = Vector{T}(undef, total_nnzh)
        fill!(hess_obj, zero(T))
        _obj_hess_coord!(m.objs, x_flat, m.θ, hess_obj, one(T))
        fill!(hess_buffer, zero(T))
        _con_hess_coord!(m.cons, x_flat, m.θ, y_flat, hess_buffer, zero(T))
        w = [bobj_weight_cpu[(i - 1) ÷ nh + 1] for i in 1:length(perm)]
        hvals .= w .* hess_obj[perm] .+ hess_buffer[perm]
    end
    return hvals
end

# ============================================================================
# Error guards: vector-argument NLPModels API on batch models
# ============================================================================

_batch_vector_error(name, m) = throw(ArgumentError(
    "$name on batch ExaModel requires matrix arguments. " *
    "Use the batch API or get_model(m) for the fused model.",
))

function obj(m::BatchExaModel, x::AbstractVector)
    _batch_vector_error("obj", m)
end

function cons_nln!(m::BatchExaModel, x::AbstractVector, c::AbstractVector)
    _batch_vector_error("cons_nln!", m)
end

function NLPModels.grad!(m::BatchExaModel, x::AbstractVector, g::AbstractVector)
    _batch_vector_error("grad!", m)
end

function NLPModels.jac_coord!(m::BatchExaModel, x::AbstractVector, jac::AbstractVector)
    _batch_vector_error("jac_coord!", m)
end

function NLPModels.hess_coord!(
    m::BatchExaModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    _batch_vector_error("hess_coord!", m)
end

function NLPModels.hess_coord!(
    m::BatchExaModel,
    x::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    _batch_vector_error("hess_coord!", m)
end
