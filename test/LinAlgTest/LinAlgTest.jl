module LinAlgTest

using ExaModels
using Test, ForwardDiff, LinearAlgebra

# Reuse the AD testing infrastructure from ADTest
# We test that linear algebra operations on ExaModels nodes produce correct
# gradients and Hessians by comparing against ForwardDiff.

function gradient(f, x)
    T = eltype(x)
    y = fill!(similar(x), zero(T))
    ExaModels.gradient!(y, (p, x, θ) -> f(x), x, nothing, nothing, one(T))
    return y
end

function sgradient(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(nothing), nothing)
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    a1 = unique(y1)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    n = length(a1)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Tuple{Int, Int}, n)

    ExaModels.sgradient!(buffer_I, ff, nothing, nothing, nothing, comp, 0, NaN)
    ExaModels.sgradient!(buffer, ff, nothing, x, nothing, comp, 0, one(T))

    y = zeros(length(x))
    y[collect(i for (i, j) in buffer_I)] += buffer

    return y
end

function shessian(f, x)
    T = eltype(x)

    ff = f(ExaModels.VarSource())
    t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(nothing), nothing)
    y2 = []
    ExaModels.hrpass0(t, nothing, y2, nothing, nothing, 0, NaN, NaN)

    a2 = unique(y2)
    comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

    n = length(a2)
    buffer = fill!(similar(x, n), zero(T))
    buffer_I = similar(x, Int, n)
    buffer_J = similar(x, Int, n)

    ExaModels.shessian!(
        buffer_I,
        buffer_J,
        ff,
        nothing,
        nothing,
        nothing,
        comp,
        0,
        NaN,
        NaN,
    )
    ExaModels.shessian!(buffer, nothing, ff, nothing, x, nothing, comp, 0, one(T), zero(T))

    y = zeros(length(x), length(x))
    for (k, (i, j)) in enumerate(zip(buffer_I, buffer_J))
        if i == j
            y[i, j] += buffer[k]
        else
            y[i, j] += buffer[k]
            y[j, i] += buffer[k]
        end
    end
    return y
end

# Helper: extract individual node elements from source as a Vector
# x[1:3] on VarSource returns Var(1:3), not a vector of Var nodes.
# We must use [x[1], x[2], x[3]] to get a Vector{<:AbstractNode}.
_vec(x, inds) = [x[i] for i in inds]
_mat(x, rows, cols) = [x[rows[i] + cols[j] - 1] for i in eachindex(rows), j in eachindex(cols)]

# Linear algebra test functions that operate on flat variable vectors
# and reshape them into matrices/vectors for linalg operations.
# NOTE: All vector/matrix construction uses explicit element indexing.

const LINALG_FUNCTIONS = [
    # dot product: dot([x1,x2,x3], [x4,x5,x6])
    ("linalg-dot-node-node", x -> dot(_vec(x, 1:3), _vec(x, 4:6))),
    # dot product with constant vector
    ("linalg-dot-real-node", x -> dot([1.0, 2.0, 3.0], _vec(x, 1:3))),
    ("linalg-dot-node-real", x -> dot(_vec(x, 1:3), [1.0, 2.0, 3.0])),
    # sum
    ("linalg-sum", x -> sum(_vec(x, 1:4))),
    # norm (2-norm)
    ("linalg-norm2", x -> norm(_vec(x, 1:3))),
    # norm (3-norm)
    ("linalg-norm3", x -> norm(_vec(x, 1:3), 3)),
    # matrix-vector product: [x1 x3; x2 x4] * [x5; x6]
    (
        "linalg-matvec-node-node",
        x -> begin
            A = [x[1] x[3]; x[2] x[4]]
            v = [x[5], x[6]]
            r = A * v
            r[1] + r[2]
        end,
    ),
    # matrix-vector product with real matrix
    (
        "linalg-matvec-real-node",
        x -> begin
            A = [1.0 2.0; 3.0 4.0]
            v = [x[1], x[2]]
            r = A * v
            r[1] + r[2]
        end,
    ),
    # trace
    (
        "linalg-tr",
        x -> begin
            A = [x[1] x[3]; x[2] x[4]]
            tr(A)
        end,
    ),
    # determinant 2x2
    (
        "linalg-det-2x2",
        x -> begin
            A = [x[1] x[2]; x[3] x[4]]
            det(A)
        end,
    ),
    # determinant 3x3
    (
        "linalg-det-3x3",
        x -> begin
            A = [x[1] x[2] x[3]; x[4] x[5] x[6]; x[7] x[8] x[9]]
            det(A)
        end,
    ),
    # cross product components summed
    (
        "linalg-cross",
        x -> begin
            c = cross(_vec(x, 1:3), _vec(x, 4:6))
            c[1] + c[2] + c[3]
        end,
    ),
    # vector addition
    (
        "linalg-vec-add",
        x -> begin
            r = _vec(x, 1:3) + _vec(x, 4:6)
            r[1] + r[2] + r[3]
        end,
    ),
    # vector subtraction
    (
        "linalg-vec-sub",
        x -> begin
            r = _vec(x, 1:3) - _vec(x, 4:6)
            r[1] + r[2] + r[3]
        end,
    ),
    # scalar * vector
    (
        "linalg-scalar-vec",
        x -> begin
            r = x[1] * _vec(x, 2:4)
            r[1] + r[2] + r[3]
        end,
    ),
    # matrix * matrix (sum of elements to get scalar)
    (
        "linalg-matmul",
        x -> begin
            A = [x[1] x[2]; x[3] x[4]]
            B = [x[5] x[6]; x[7] x[8]]
            C = A * B
            C[1, 1] + C[1, 2] + C[2, 1] + C[2, 2]
        end,
    ),
    # composite: norm of matvec
    (
        "linalg-composite-norm-matvec",
        x -> begin
            A = [1.0 2.0; 3.0 4.0]
            v = [x[1], x[2]]
            norm(A * v)
        end,
    ),
    # composite: det * dot
    (
        "linalg-composite-det-dot",
        x -> begin
            A = [x[1] x[2]; x[3] x[4]]
            det(A) * dot(_vec(x, 5:6), _vec(x, 7:8))
        end,
    ),
]

function runtests()
    return @testset "Linear Algebra test" begin
        for (name, f) in LINALG_FUNCTIONS
            x0 = 0.5 .+ rand(10)  # avoid zero for norm derivatives
            @testset "$name" begin
                g_fd = ForwardDiff.gradient(f, x0)
                h_fd = ForwardDiff.hessian(f, x0)
                @test gradient(f, x0) ≈ g_fd atol = 1.0e-6
                @test sgradient(f, x0) ≈ g_fd atol = 1.0e-6
                @test shessian(f, x0) ≈ h_fd atol = 1.0e-6
            end
        end
    end
end

end # module
