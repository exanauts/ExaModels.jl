# deprecated.jl — legacy mutable API built on top of the functional ExaCore API.
#
# ExaCore() (the default, concrete = Val(false)) returns a LegacyExaCore, which
# lets callers treat the core as mutable via the old positional API.
# ExaCore(concrete = Val(true)) bypasses this wrapper and returns an ExaCore directly.
#
# The legacy named wrappers (variable, parameter, …) are defined here for
# LegacyExaCore and forward to the corresponding add_* functions on the inner ExaCore.

# ---------------------------------------------------------------------------
# LegacyExaCore struct
# ---------------------------------------------------------------------------

"""
    LegacyExaCore{T,VT,B,S}

A mutable wrapper around an immutable [`ExaCore`](@ref) that provides the
legacy mutating API (`variable`, `constraint`, etc.).

`ExaCore()` returns a `LegacyExaCore` by default. Use
`ExaCore(concrete = Val(true))` to obtain the bare immutable `ExaCore`
required for AOT compilation.
"""
mutable struct LegacyExaCore{T, VT <: AbstractArray{T}, B, S} <: AbstractExaCore{T,VT,B,S}
    inner::Any  # ExaCore{T,VT,B,S,...} — type erased so the tuple type params can grow
end

# Override the Val{false} dispatch defined in nlp.jl so ExaCore() returns a LegacyExaCore.
@inline function _make_exacore(::Val{false}, ::Type{T}, backend; kwargs...) where {T}
    @warn "`ExaCore()` is deprecated, and will be removed in v0.11. Use `ExaCore(concrete = Val(true))` for the immutable ExaCore. The default behavior for `ExaCore()` will change to return the immutable ExaCore in v0.11."
    inner = _exa_core(; x0 = convert_array(zeros(T, 0), backend), backend, kwargs...)
    return LegacyExaCore{T, typeof(inner.x0), typeof(backend), typeof(inner.tag)}(inner)
end

# ---------------------------------------------------------------------------
# Property forwarding for LegacyExaCore
# ---------------------------------------------------------------------------

function Base.getproperty(c::LegacyExaCore, s::Symbol)
    s === :inner && return getfield(c, :inner)
    return getproperty(getfield(c, :inner), s)
end

function Base.show(io::IO, c::LegacyExaCore{T,VT,B}) where {T,VT,B}
    return Base.show(io, getfield(c, :inner))
end

function ExaModel(c::LegacyExaCore; kwargs...)
    return ExaModel(c.inner; kwargs...)
end

function set_parameter!(c::LegacyExaCore, param::Parameter, values::AbstractArray)
    return set_parameter!(c.inner, param, values)
end

# ---------------------------------------------------------------------------
# Legacy named wrappers (deprecated)
# ---------------------------------------------------------------------------

"""
    variable(core, dims...; start = 0, lvar = -Inf, uvar = Inf, kwargs...)

Deprecated. Use [`add_var`](@ref) instead.
"""
function variable(c::LegacyExaCore, args...; kwargs...)
    new_core, v = add_var(c.inner, args...; kwargs...)
    c.inner = new_core
    return v
end

"""
    parameter(core, start::AbstractArray)

Deprecated. Use [`add_par`](@ref) instead.
"""
function parameter(c::LegacyExaCore, start::AbstractArray; kwargs...)
    new_core, p = add_par(c.inner, start; kwargs...)
    c.inner = new_core
    return p
end

"""
    objective(core, generator)

Deprecated. Use [`add_obj`](@ref) instead.
"""
function objective(c::LegacyExaCore, args...; kwargs...)
    new_core, o = add_obj(c.inner, args...; kwargs...)
    c.inner = new_core
    return o
end

"""
    constraint(core, generator; start = 0, lcon = 0, ucon = 0, kwargs...)

Deprecated. Use [`add_con`](@ref) instead.
"""
function constraint(c::LegacyExaCore, args...; kwargs...)
    new_core, con = add_con(c.inner, args...; kwargs...)
    c.inner = new_core
    return con
end

"""
    constraint!(core, c1, generator)

Deprecated. Use [`add_con!`](@ref) instead.
"""
function constraint!(c::LegacyExaCore, args...; kwargs...)
    new_core, aug = add_con!(c.inner, args...; kwargs...)
    c.inner = new_core
    return aug
end

"""
    subexpr(core, generator; kwargs...)

Deprecated. Use [`add_expr`](@ref) instead.
"""
function subexpr(c::LegacyExaCore, args...; kwargs...)
    new_core, ex = add_expr(c.inner, args...; kwargs...)
    c.inner = new_core
    return ex
end

# ---------------------------------------------------------------------------
# Forwarding add_* methods for LegacyExaCore so that @add_var/@add_con/etc.
# macros (which expand to `core, v = add_*(core, ...)`) work with the legacy
# mutable wrapper.  Each method delegates to the inner ExaCore, mutates
# c.inner in place, and returns (c, result) to match the functional API.
# ---------------------------------------------------------------------------

function add_var(c::LegacyExaCore, args...; kwargs...)
    new_core, v = add_var(c.inner, args...; kwargs...)
    c.inner = new_core
    return (c, v)
end

function add_par(c::LegacyExaCore, args...; kwargs...)
    new_core, p = add_par(c.inner, args...; kwargs...)
    c.inner = new_core
    return (c, p)
end

function add_obj(c::LegacyExaCore, args...; kwargs...)
    new_core, o = add_obj(c.inner, args...; kwargs...)
    c.inner = new_core
    return (c, o)
end

function add_con(c::LegacyExaCore, args...; kwargs...)
    new_core, con = add_con(c.inner, args...; kwargs...)
    c.inner = new_core
    return (c, con)
end

function add_con!(c::LegacyExaCore, args...; kwargs...)
    new_core, aug = add_con!(c.inner, args...; kwargs...)
    c.inner = new_core
    return (c, aug)
end

function add_expr(c::LegacyExaCore, args...; kwargs...)
    new_core, ex = add_expr(c.inner, args...; kwargs...)
    c.inner = new_core
    return (c, ex)
end

export LegacyExaCore, variable, parameter, objective, constraint, constraint!, subexpr
