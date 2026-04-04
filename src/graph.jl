# ── Abstract node types ───────────────────────────────────────────────────────

"""
    AbstractNode

Root abstract type for all nodes in the ExaModels symbolic computation graph.
Every node is callable as `node(i, x, θ)` and returns the scalar value of the
expression at the current evaluation point, where `i` is the iteration index,
`x` the primal variable vector, and `θ` the parameter vector.
"""
abstract type AbstractNode end

"""
    AbstractAdjointNode

Abstract type for nodes in the first-order (gradient) forward-pass tree.
"""
abstract type AbstractAdjointNode end

"""
    AbstractSecondAdjointNode

Abstract type for nodes in the second-order (Hessian) reverse-pass tree.
"""
abstract type AbstractSecondAdjointNode end

"""
    Null{T} <: AbstractNode

A leaf node that always evaluates to a fixed scalar.  The value is stored in a
field (as opposed to [`Constant`](@ref), which encodes the value as a type
parameter).  Used when the constant is not known at compile time.

`Null()` creates a zero-valued node; `Null(v)` creates a node that returns
`T(v)` where `T` is the element type of the primal vector.
"""
struct Null{T} <: AbstractNode
    value::T
end
Null() = Null(nothing)

"""
    Constant{T} <: AbstractNode

A compile-time constant node whose value `T` is embedded as a **type
parameter**.

## Design rationale

Because `T` is a type parameter rather than a struct field, the Julia compiler
(and `juliac --trim=safe`) can propagate `T` concretely through every
downstream `Node1` / `Node2` type, making the entire computation graph
type-stable with no runtime storage for the constant.

## Construction

```julia
Constant(2)      # → Constant{2}()
Constant(3.14)   # → Constant{3.14}()
```

## Evaluation

```julia
c = Constant(42)
c(i, x, θ)  # → 42
```

## Algebraic simplification

`specialization.jl` registers identities for `Constant` combined with
[`AbstractNode`](@ref) operands:

| expression             | simplification       |
|:-----------------------|:---------------------|
| `x + Constant(0)`      | `x`                  |
| `x * Constant(1)`      | `x`                  |
| `x / Constant(1)`      | `x`                  |
| `x ^ Constant(0)`      | `Constant(1)`        |
| `x ^ Constant(1)`      | `x`                  |
| `x ^ Constant(2)`      | `Node1(abs2, x)`     |
| `x ^ Constant(-1)`     | `inv(x)`             |
| `Constant(0) * x`      | `Constant(0)`        |
| `Constant(0) / x`      | `Constant(0)`        |

These rules fire at **model-construction time** (when the `ExaCore` is being
built), collapsing trivial branches before the graph is finalized.
"""
struct Constant{T} <: AbstractNode end
@inline Constant(T) = Constant{T}()
@inline (::Constant{T})(i, x, θ) where {T} = T

Base.:^(::Constant{0}, x::AbstractNode) = Constant(0)
Base.:^(::Constant{1}, x::AbstractNode) = Constant(1)




"""
    AdjointNull{V} <: AbstractAdjointNode

Leaf node in the first-order adjoint tree representing a constant (zero
contribution to the gradient).  Carries the primal value `x` so that upstream
nodes can read it without recomputation.
"""
struct AdjointNull{V} <: AbstractAdjointNode
    x::V
end

"""
    SecondAdjointNull{V} <: AbstractSecondAdjointNode

Leaf node in the second-order adjoint tree representing a constant (zero
contribution to both the gradient and the Hessian).  Carries the primal value
`x` for upstream consumption.
"""
struct SecondAdjointNull{V} <: AbstractSecondAdjointNode
    x::V
end

"""
    VarSource <: AbstractNode

Sentinel node used as the root of a variable array.  Indexing a `VarSource`
with an integer `i` returns `Var(i)`, building the leaf node that refers to
the `i`-th decision variable.
"""
struct VarSource <: AbstractNode end

"""
    Var{I} <: AbstractNode

A leaf node representing the `i`-th decision variable.

When evaluated as `v(i, x, θ)`, returns `x[v.i]` (or `x[v.i(i,x,θ)]` when
the index is itself a node).
"""
struct Var{I} <: AbstractNode
    i::I
end

struct ParameterSource <: AbstractNode end
struct ParameterNode{I} <: AbstractNode
    i::I
end

"""
    DataSource <: AbstractNode

Sentinel node used as the root of a parameter (data) array.  Indexing or
accessing fields on a `DataSource` returns a [`DataIndexed`](@ref) node that
encodes the access path as nested type parameters for zero-overhead data
lookup.
"""
struct DataSource <: AbstractNode end

"""
    DataIndexed{I, J} <: AbstractNode

A node representing the lookup `inner.J` (or `inner[J]`) where `inner` is a
`DataSource` or another `DataIndexed`.  The field/index `J` is encoded as a type
parameter so that `getfield` / `getindex` can be resolved at compile time.

Constructed implicitly via `getproperty` / `getindex` on a [`DataSource`](@ref).
"""
struct DataIndexed{I, J} <: AbstractNode
    inner::I
end

@inline DataIndexed(inner::I, n) where {I} = DataIndexed{I, n}(inner)
@inline DataIndexed(inner::I, s::Constant{n}) where {I, n} = DataIndexed{I, n}(inner)
"""
    Node1{F, I} <: AbstractNode

An interior node representing a unary operation `F` applied to one child node.

Evaluated as `F(inner(i, x, θ))`.

# Fields
- `inner::I`: child node
"""
struct Node1{F,I} <: AbstractNode
    inner::I
end

"""
    Node2{F, I1, I2} <: AbstractNode

An interior node representing a binary operation `F` applied to two children.
Either child can be a node or a plain scalar (e.g. `Real`), allowing numeric
coefficients to be stored directly without wrapping in [`Constant`](@ref).

Evaluated as `F(inner1(i, x, θ), inner2(i, x, θ))`.

# Fields
- `inner1::I1`: left child
- `inner2::I2`: right child
"""
struct Node2{F,I1,I2} <: AbstractNode
    inner1::I1
    inner2::I2
end

struct FirstFixed{F}
    inner::F
end
struct SecondFixed{F}
    inner::F
end

@inline Base.getproperty(n::DataSource, s::Symbol) = DataIndexed(n, s)
@inline Base.getindex(n::DataSource, i) = DataIndexed(n, i)
@inline Base.indexed_iterate(n::P, idx, start = 1) where {P <: Union{DataSource, DataIndexed}} = (DataIndexed(n, idx), idx + 1)

@inline Base.getproperty(v::DataIndexed{I, n}, s::Symbol) where {I, n} = DataIndexed(v, s)
@inline Base.getindex(v::DataIndexed{I, n}, i) where {I, n} = DataIndexed(v, i)
@inline Base.indexed_iterate(v::DataIndexed{I, n}, idx, start = 1) where {I, n} = (DataIndexed(v, idx), idx + 1)


@inline Base.getindex(n::VarSource, i) = Var(i)
@inline Base.getindex(::ParameterSource, i) = ParameterNode(i)

@inline Node1(f::F, inner::I) where {F,I} = Node1{F,I}(inner)
@inline Node2(f::F, inner1::I1, inner2::I2) where {F,I1,I2} = Node2{F,I1,I2}(inner1, inner2)


struct Identity end

@inline (v::Var{I})(i, x, θ) where {I<:AbstractNode} = @inbounds x[v.i(i, x, θ)]
@inline (v::Var{I})(i, x, θ) where {I} = @inbounds x[v.i]
@inline (v::Var{I})(i::Identity, x, θ) where {I <: AbstractNode} = x[v]
@inline (v::Var{I})(i::Identity, x, θ) where {I <: Real} = x[v]

@inline (v::ParameterNode{I})(i, x, θ) where {I<:AbstractNode} = @inbounds θ[v.i(i, x, θ)]
@inline (v::ParameterNode{I})(::Any, x, θ) where {I} = @inbounds θ[v.i]
@inline (v::ParameterNode{I})(::Identity, x, θ) where {I<:AbstractNode} = @inbounds θ[v.i]

@inline (v::DataSource)(i, x, θ) = i
@inline (v::DataIndexed{I, n})(i, x, θ) where {I, n} = @inbounds getfield(getfield(v, :inner)(i, x, θ), n)

@inline (v::DataIndexed)(i::Identity, x, θ) = eltype(θ)(NaN)
@inline (v::DataSource)(i::Identity, x, θ) = eltype(θ)(NaN)

"""
    AdjointNode1{F, T, I} <: AbstractAdjointNode

Gradient-pass node for a unary operation `F`.  Stores the primal value and the
partial derivative alongside the child so a single forward sweep simultaneously
computes both.

# Fields
- `x::T`: primal value `F(inner.x)`
- `y::T`: derivative `F'(inner.x)`
- `inner::I`: child node
"""
struct AdjointNode1{F,T,I} <: AbstractAdjointNode
    x::T
    y::T
    inner::I
end

"""
    AdjointNode2{F, T, I1, I2} <: AbstractAdjointNode

Gradient-pass node for a binary operation `F`.  Stores the primal value and
both partial derivatives.

# Fields
- `x::T`: primal value `F(inner1.x, inner2.x)`
- `y1::T`: partial derivative w.r.t. first argument
- `y2::T`: partial derivative w.r.t. second argument
- `inner1::I1`: left child
- `inner2::I2`: right child
"""
struct AdjointNode2{F,T,I1,I2} <: AbstractAdjointNode
    x::T
    y1::T
    y2::T
    inner1::I1
    inner2::I2
end

"""
    AdjointNodeVar{I, T} <: AbstractAdjointNode

Leaf node in the gradient-pass tree corresponding to a decision variable.

# Fields
- `i::I`: variable index
- `x::T`: primal value `x[i]` at the current evaluation point
"""
struct AdjointNodeVar{I,T} <: AbstractAdjointNode
    i::I
    x::T
end

"""
    AdjointNodeSource{VT}

Factory for [`AdjointNodeVar`](@ref) leaves.  Indexing with `i` returns
`AdjointNodeVar(i, inner[i])`, seeding the gradient-pass tree with the current
primal value.

# Fields
- `inner::VT`: primal variable vector (or `nothing` for a zero-valued seed)
"""
struct AdjointNodeSource{VT}
    inner::VT
end

@inline AdjointNode1(f::F, x::T, y, inner::I) where {F,T,I} =
    AdjointNode1{F,T,I}(x, y, inner)
@inline AdjointNode2(f::F, x::T, y1, y2, inner1::I1, inner2::I2) where {F,T,I1,I2} =
    AdjointNode2{F,T,I1,I2}(x, y1, y2, inner1, inner2)

@inline Base.getindex(x::I, i) where {I<:AdjointNodeSource{Nothing}} =
    AdjointNodeVar(i, 0)
@inline Base.getindex(x::I, i) where {I<:AdjointNodeSource} =
    @inbounds AdjointNodeVar(i, x.inner[i])


"""
    SecondAdjointNode1{F, T, I} <: AbstractSecondAdjointNode

Hessian-pass node for a unary operation `F`.  A single forward sweep stores
the primal value, first derivative, and second derivative needed for the
reverse Hessian accumulation.

# Fields
- `x::T`: primal value `F(inner.x)`
- `y::T`: first derivative `F'(inner.x)`
- `h::T`: second derivative `F''(inner.x)`
- `inner::I`: child node
"""
struct SecondAdjointNode1{F,T,I} <: AbstractSecondAdjointNode
    x::T
    y::T
    h::T
    inner::I
end

"""
    SecondAdjointNode2{F, T, I1, I2} <: AbstractSecondAdjointNode

Hessian-pass node for a binary operation `F`.

# Fields
- `x::T`: primal value `F(inner1.x, inner2.x)`
- `y1::T`: partial derivative w.r.t. first argument
- `y2::T`: partial derivative w.r.t. second argument
- `h11::T`: second partial w.r.t. first argument
- `h12::T`: mixed second partial
- `h22::T`: second partial w.r.t. second argument
- `inner1::I1`: left child
- `inner2::I2`: right child
"""
struct SecondAdjointNode2{F,T,I1,I2} <: AbstractSecondAdjointNode
    x::T
    y1::T
    y2::T
    h11::T
    h12::T
    h22::T
    inner1::I1
    inner2::I2
end

"""
    SecondAdjointNodeVar{I, T} <: AbstractSecondAdjointNode

Leaf node in the Hessian-pass tree corresponding to a decision variable.

# Fields
- `i::I`: variable index
- `x::T`: primal value `x[i]` at the current evaluation point
"""
struct SecondAdjointNodeVar{I,T} <: AbstractSecondAdjointNode
    i::I
    x::T
end

"""
    SecondAdjointNodeSource{VT}

Factory for [`SecondAdjointNodeVar`](@ref) leaves.  Indexing with `i` returns
`SecondAdjointNodeVar(i, inner[i])`, seeding the Hessian-pass tree.

# Fields
- `inner::VT`: primal variable vector (or `nothing` for a zero-valued seed)
"""
struct SecondAdjointNodeSource{VT}
    inner::VT
end

@inline SecondAdjointNode1(f::F, x::T, y, h, inner::I) where {F,T,I} =
    SecondAdjointNode1{F,T,I}(x, y, h, inner)
@inline SecondAdjointNode2(
    f::F,
    x::T,
    y1,
    y2,
    h11,
    h12,
    h22,
    inner1::I1,
    inner2::I2,
) where {F,T,I1,I2} =
    SecondAdjointNode2{F,T,I1,I2}(x, y1, y2, h11, h12, h22, inner1, inner2)

@inline Base.getindex(x::I, i) where {I<:SecondAdjointNodeSource{Nothing}} =
    SecondAdjointNodeVar(i, 0)
@inline Base.getindex(x::I, i) where {I<:SecondAdjointNodeSource} =
    @inbounds SecondAdjointNodeVar(i, x.inner[i])


# Pretty printing for node types

_opname(::Type{F}) where {F} = string(F)
_opname(::Type{typeof(+)}) = "+"
_opname(::Type{typeof(-)}) = "-"
_opname(::Type{typeof(*)}) = "*"
_opname(::Type{typeof(/)}) = "/"
_opname(::Type{typeof(^)}) = "^"

# Helper to print expression tree
function _print_tree(io::IO, node::Null{Nothing}, indent::Int)
    print(io, " "^indent, "0")
end
function _print_tree(io::IO, node::Null{T}, indent::Int) where {T}
    print(io, " "^indent, node.value)
end
function _print_tree(io::IO, node::Var{I}, indent::Int) where {I<:AbstractNode}
    print(io, " "^indent, "x[")
    _print_tree(io, node.i, 0)
    print(io, "]")
end
function _print_tree(io::IO, node::Var{I}, indent::Int) where {I}
    print(io, " "^indent, "x[", node.i, "]")
end
function _print_tree(io::IO, node::ParameterNode{I}, indent::Int) where {I<:AbstractNode}
    print(io, " "^indent, "θ[")
    _print_tree(io, node.i, 0)
    print(io, "]")
end
function _print_tree(io::IO, node::ParameterNode{I}, indent::Int) where {I}
    print(io, " "^indent, "θ[", node.i, "]")
end
function _print_tree(io::IO, node::ParSource, indent::Int)
    print(io, " "^indent, "i")
end
function _print_tree(io::IO, node::ParIndexed{I,n}, indent::Int) where {I,n}
    _print_tree(io, node.inner, 0)
    print(io, ".", n)
end
function _print_tree(io::IO, node::VarSource, indent::Int)
    print(io, " "^indent, "x")
end
function _print_tree(io::IO, node::ParameterSource, indent::Int)
    print(io, " "^indent, "θ")
end
function _print_tree(io::IO, node::Node1{F}, indent::Int) where {F}
    print(io, " "^indent, _opname(F), "(")
    _print_tree(io, node.inner, 0)
    print(io, ")")
end
# Check if a node is a zero constant
_is_zero(::Null{Nothing}) = true
_is_zero(n::Null) = n.value == 0
_is_zero(n::Real) = n == 0
_is_zero(_) = false

# Check if a node is a one constant
_is_one(n::Null) = n.value == 1
_is_one(n::Real) = n == 1
_is_one(_) = false

function _print_tree(io::IO, node::Node2{F}, indent::Int) where {F}
    op = _opname(F)
    a, b = node.inner1, node.inner2
    # Simplify identity operations
    if op == "+" && _is_zero(b)
        return _print_tree(io, a, indent)
    elseif op == "+" && _is_zero(a)
        return _print_tree(io, b, indent)
    elseif op == "-" && _is_zero(b)
        return _print_tree(io, a, indent)
    elseif op == "*" && _is_one(b)
        return _print_tree(io, a, indent)
    elseif op == "*" && _is_one(a)
        return _print_tree(io, b, indent)
    elseif op == "*" && (_is_zero(a) || _is_zero(b))
        return print(io, " "^indent, "0")
    elseif op == "^" && _is_one(b)
        return _print_tree(io, a, indent)
    elseif op == "/" && _is_one(b)
        return _print_tree(io, a, indent)
    end
    if op == "^"
        print(io, " "^indent)
        _print_tree(io, a, 0)
        print(io, "^")
        _print_tree(io, b, 0)
    elseif op in ("*", "+", "-")
        print(io, " "^indent)
        _print_tree(io, a, 0)
        print(io, " ", op, " ")
        _print_tree(io, b, 0)
    elseif op == "/"
        print(io, " "^indent, "(")
        _print_tree(io, a, 0)
        print(io, " / ")
        _print_tree(io, b, 0)
        print(io, ")")
    else
        print(io, " "^indent, op, "(")
        _print_tree(io, a, 0)
        print(io, ", ")
        _print_tree(io, b, 0)
        print(io, ")")
    end
end

# Handle Pair nodes (used in constraint augmentation: index => expr)
function _print_tree(io::IO, node::Pair{P,S}, indent::Int) where {P,S<:AbstractNode}
    _print_tree(io, node.second, indent)
end

# Compact expression string for a node
function _expr_string(node)
    buf = IOBuffer()
    _print_tree(buf, node, 0)
    s = String(take!(buf))
    # Strip outermost parentheses
    if length(s) >= 2 && s[1] == '(' && s[end] == ')'
        # Verify they are matching (not e.g. "(a) + (b)")
        depth = 0
        matched = true
        for (k, ch) in enumerate(s)
            depth += (ch == '(') - (ch == ')')
            if depth == 0 && k < length(s)
                matched = false
                break
            end
        end
        matched && return s[2:end-1]
    end
    return s
end

# Fallback for non-node types (e.g., Real constants in SIMDFunction.f)
function _print_tree(io::IO, node::Real, indent::Int)
    print(io, " "^indent, node)
end

# Generic fallback for unknown types
function _print_tree(io::IO, node, indent::Int)
    print(io, " "^indent, node)
end

# --- Symbolic expression nodes ---

function Base.show(io::IO, node::Null{Nothing})
    print(io, "Null(0)")
end
function Base.show(io::IO, node::Null{T}) where {T}
    print(io, "Null(", node.value, ")")
end

function Base.show(io::IO, node::VarSource)
    print(io, "VarSource()")
end

function Base.show(io::IO, node::Var{I}) where {I}
    print(io, _expr_string(node))
end

function Base.show(io::IO, node::ParameterSource)
    print(io, "ParameterSource()")
end

function Base.show(io::IO, node::ParameterNode{I}) where {I}
    print(io, _expr_string(node))
end

function Base.show(io::IO, node::ParSource)
    print(io, "p")
end

function Base.show(io::IO, node::ParIndexed{I,n}) where {I,n}
    print(io, _expr_string(node))
end

function Base.show(io::IO, node::Node1{F,I}) where {F,I}
    print(io, _expr_string(node))
end

function Base.show(io::IO, node::Node2{F,I1,I2}) where {F,I1,I2}
    print(io, _expr_string(node))
end

# --- First-order adjoint nodes ---

function Base.show(io::IO, node::AdjointNull{V}) where {V}
    print(io, "AdjointNull(x = ", node.x, ")")
end

function Base.show(io::IO, node::AdjointNodeVar{I,T}) where {I,T}
    print(io, "AdjointNodeVar(i = ", node.i, ", x = ", node.x, ")")
end

function Base.show(io::IO, node::AdjointNode1{F,T,I}) where {F,T,I}
    print(io, "AdjointNode1{", _opname(F), "}(x = ", node.x, ", y = ", node.y, ")")
end

function Base.show(io::IO, node::AdjointNode2{F,T,I1,I2}) where {F,T,I1,I2}
    print(io, "AdjointNode2{", _opname(F), "}(x = ", node.x, ", y1 = ", node.y1, ", y2 = ", node.y2, ")")
end

function Base.show(io::IO, node::AdjointNodeSource{VT}) where {VT}
    print(io, "AdjointNodeSource{", VT, "}()")
end

# --- Second-order adjoint nodes ---

function Base.show(io::IO, node::SecondAdjointNull{V}) where {V}
    print(io, "SecondAdjointNull(x = ", node.x, ")")
end

function Base.show(io::IO, node::SecondAdjointNodeVar{I,T}) where {I,T}
    print(io, "SecondAdjointNodeVar(i = ", node.i, ", x = ", node.x, ")")
end

function Base.show(io::IO, node::SecondAdjointNode1{F,T,I}) where {F,T,I}
    print(io, "SecondAdjointNode1{", _opname(F), "}(x = ", node.x, ", y = ", node.y, ", h = ", node.h, ")")
end

function Base.show(io::IO, node::SecondAdjointNode2{F,T,I1,I2}) where {F,T,I1,I2}
    print(io, "SecondAdjointNode2{", _opname(F), "}(x = ", node.x,
        ", y1 = ", node.y1, ", y2 = ", node.y2,
        ", h11 = ", node.h11, ", h12 = ", node.h12, ", h22 = ", node.h22, ")")
end

function Base.show(io::IO, node::SecondAdjointNodeSource{VT}) where {VT}
    print(io, "SecondAdjointNodeSource{", VT, "}()")
end

# --- MIME "text/plain" for detailed display ---

function Base.show(io::IO, ::MIME"text/plain", node::AbstractNode)
    println(io, typeof(node))
    println(io, "  Expression: ", _expr_string(node))
end

function Base.show(io::IO, ::MIME"text/plain", node::AdjointNode1{F,T,I}) where {F,T,I}
    println(io, "AdjointNode1")
    println(io, "  Operation: ", _opname(F))
    println(io, "  Value (x): ", node.x)
    println(io, "  Sensitivity (y): ", node.y)
    print(io, "  Inner: ", node.inner)
end

function Base.show(io::IO, ::MIME"text/plain", node::AdjointNode2{F,T,I1,I2}) where {F,T,I1,I2}
    println(io, "AdjointNode2")
    println(io, "  Operation: ", _opname(F))
    println(io, "  Value (x): ", node.x)
    println(io, "  Sensitivity (y1): ", node.y1)
    println(io, "  Sensitivity (y2): ", node.y2)
    println(io, "  Inner1: ", node.inner1)
    print(io, "  Inner2: ", node.inner2)
end

function Base.show(io::IO, ::MIME"text/plain", node::AdjointNull{V}) where {V}
    println(io, "AdjointNull")
    print(io, "  Value (x): ", node.x)
end

function Base.show(io::IO, ::MIME"text/plain", node::AdjointNodeVar{I,T}) where {I,T}
    println(io, "AdjointNodeVar")
    println(io, "  Index (i): ", node.i)
    print(io, "  Value (x): ", node.x)
end

function Base.show(io::IO, ::MIME"text/plain", node::SecondAdjointNode1{F,T,I}) where {F,T,I}
    println(io, "SecondAdjointNode1")
    println(io, "  Operation: ", _opname(F))
    println(io, "  Value (x): ", node.x)
    println(io, "  1st-order (y): ", node.y)
    println(io, "  2nd-order (h): ", node.h)
    print(io, "  Inner: ", node.inner)
end

function Base.show(io::IO, ::MIME"text/plain", node::SecondAdjointNode2{F,T,I1,I2}) where {F,T,I1,I2}
    println(io, "SecondAdjointNode2")
    println(io, "  Operation: ", _opname(F))
    println(io, "  Value (x): ", node.x)
    println(io, "  1st-order (y1): ", node.y1)
    println(io, "  1st-order (y2): ", node.y2)
    println(io, "  2nd-order (h11): ", node.h11)
    println(io, "  2nd-order (h12): ", node.h12)
    println(io, "  2nd-order (h22): ", node.h22)
    println(io, "  Inner1: ", node.inner1)
    print(io, "  Inner2: ", node.inner2)
end

function Base.show(io::IO, ::MIME"text/plain", node::SecondAdjointNull{V}) where {V}
    println(io, "SecondAdjointNull")
    print(io, "  Value (x): ", node.x)
end

function Base.show(io::IO, ::MIME"text/plain", node::SecondAdjointNodeVar{I,T}) where {I,T}
    println(io, "SecondAdjointNodeVar")
    println(io, "  Index (i): ", node.i)
    print(io, "  Value (x): ", node.x)
end

@inline (v::Null{Nothing})(i, x::V, θ) where {T,V<:AbstractVector{T}} = zero(T)
@inline (v::Null{N})(i, x::V, θ) where {N,T,V<:AbstractVector{T}} = T(v.value)
@inline (v::Null{Nothing})(i, x::AdjointNodeSource{T}, θ) where {T} = AdjointNull(zero(eltype(T)))
@inline (v::Null{N})(i, x::AdjointNodeSource{T}, θ) where {N, T} = AdjointNull(eltype(T)(v.value))
@inline (v::Null{Nothing})(i, x::SecondAdjointNodeSource{T}, θ) where {T} = SecondAdjointNull(zero(eltype(T)))
@inline (v::Null{N})(i, x::SecondAdjointNodeSource{T}, θ) where {N, T} = SecondAdjointNull(eltype(T)(v.value))

# ── SumNode / ProdNode ────────────────────────────────────────────────────────

"""
    SumNode{I} <: AbstractNode

A node representing the sum of a tuple of child nodes.

Constructed by [`exa_sum`](@ref).  Within `@add_obj`, `@add_con`, and `@add_expr` macros,
`sum(body for k in range)` is automatically rewritten to
`exa_sum(k -> body, Val(range))` with the `Val` hoisted outside the generator
closure for type stability under `juliac --trim=safe`.

In adjoint / second-adjoint mode the children are evaluated and folded via
`reduce(+, …)` (or `reduce(*, …)` for [`ProdNode`](@ref)), reusing the existing
registered `+` / `*` dispatch.  No dedicated adjoint node types are needed.
"""
struct SumNode{I} <: AbstractNode
    inners::I
end

"""
    ProdNode{I} <: AbstractNode

A node representing the product of a tuple of child nodes.

Constructed by [`exa_prod`](@ref).  See [`SumNode`](@ref) for design notes.
"""
struct ProdNode{I} <: AbstractNode
    inners::I
end

# ── Primal evaluation (x::AbstractVector → scalar) ───────────────────────────

@inline (n::SumNode{Tuple{}})(i, x::V, θ) where {T, V<:AbstractVector{T}} = zero(T)
@inline (n::SumNode)(i, x::V, θ) where {T, V<:AbstractVector{T}} =
    mapreduce(inner -> inner(i, x, θ), +, n.inners)

@inline (n::ProdNode{Tuple{}})(i, x::V, θ) where {T, V<:AbstractVector{T}} = one(T)
@inline (n::ProdNode)(i, x::V, θ) where {T, V<:AbstractVector{T}} =
    mapreduce(inner -> inner(i, x, θ), *, n.inners)

# ── Adjoint tree (gradient) ───────────────────────────────────────────────────

@inline (n::SumNode{Tuple{}})(i, x::AdjointNodeSource{VT}, θ) where {VT} =
    AdjointNull(zero(eltype(VT)))
@inline (n::SumNode)(i, x::AdjointNodeSource, θ) =
    reduce(+, map(inner -> inner(i, x, θ), n.inners))

@inline (n::ProdNode{Tuple{}})(i, x::AdjointNodeSource{VT}, θ) where {VT} =
    AdjointNull(one(eltype(VT)))
@inline (n::ProdNode)(i, x::AdjointNodeSource, θ) =
    reduce(*, map(inner -> inner(i, x, θ), n.inners))

# ── Second-adjoint tree (Hessian) ─────────────────────────────────────────────

@inline (n::SumNode{Tuple{}})(i, x::SecondAdjointNodeSource{VT}, θ) where {VT} =
    SecondAdjointNull(zero(eltype(VT)))
@inline (n::SumNode)(i, x::SecondAdjointNodeSource, θ) =
    reduce(+, map(inner -> inner(i, x, θ), n.inners))

@inline (n::ProdNode{Tuple{}})(i, x::SecondAdjointNodeSource{VT}, θ) where {VT} =
    SecondAdjointNull(one(eltype(VT)))
@inline (n::ProdNode)(i, x::SecondAdjointNodeSource, θ) =
    reduce(*, map(inner -> inner(i, x, θ), n.inners))
