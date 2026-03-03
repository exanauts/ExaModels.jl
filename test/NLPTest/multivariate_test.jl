# Tests for @register_multivariate

# ---------------------------------------------------------------------------
# Define test functions outside the test functions so they are registered once
# ---------------------------------------------------------------------------

# f3(x,y,z) = x^2 + 2y^2 + 3z^2
_f3(x, y, z) = x^2 + 2y^2 + 3z^2

function _grad_f3!(g, x, y, z)
    g[1] = 2x
    g[2] = 4y
    g[3] = 6z
end

function _hess_f3!(H, x, y, z)
    # upper-triangular, row-major: (1,1),(1,2),(1,3),(2,2),(2,3),(3,3)
    H[1] = 2.0  # d²/dx²
    H[2] = 0.0  # d²/dxdy
    H[3] = 0.0  # d²/dxdz
    H[4] = 4.0  # d²/dy²
    H[5] = 0.0  # d²/dydz
    H[6] = 6.0  # d²/dz²
end

@register_multivariate(_f3, 3, _grad_f3!, _hess_f3!)

# g2(x,y) = sin(x) * cos(y)
_g2(x, y) = sin(x) * cos(y)

function _grad_g2!(g, x, y)
    g[1] = cos(x) * cos(y)
    g[2] = -sin(x) * sin(y)
end

function _hess_g2!(H, x, y)
    # (1,1),(1,2),(2,2)
    H[1] = -sin(x) * cos(y)   # d²/dx²
    H[2] = -cos(x) * sin(y)   # d²/dxdy
    H[3] = -sin(x) * cos(y)   # d²/dy²
end

@register_multivariate(_g2, 2, _grad_g2!, _hess_g2!)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

"""
Finite-difference gradient of ExaModel objective or single constraint.
"""
function fd_gradient(m, x0; h = 1e-5)
    n = length(x0)
    g = zeros(n)
    f0 = NLPModels.obj(m, x0)
    for i in 1:n
        xp = copy(x0); xp[i] += h
        g[i] = (NLPModels.obj(m, xp) - f0) / h
    end
    return g
end

function fd_jacobian(m, x0; h = 1e-5)
    n = length(x0)
    ncon = m.meta.ncon
    J = zeros(ncon, n)
    c0 = NLPModels.cons(m, x0)
    for i in 1:n
        xp = copy(x0); xp[i] += h
        J[:, i] = (NLPModels.cons(m, xp) - c0) / h
    end
    return J
end

function fd_hessian_lag(m, x0, y0; h = 1e-5)
    n = length(x0)
    H = zeros(n, n)
    g0 = NLPModels.grad(m, x0) .+ NLPModels.jtprod(m, x0, y0)
    for i in 1:n
        xp = copy(x0); xp[i] += h
        gi = NLPModels.grad(m, xp) .+ NLPModels.jtprod(m, xp, y0)
        H[:, i] = (gi - g0) / h
    end
    return (H + H') / 2  # symmetrise
end

# ---------------------------------------------------------------------------
# Actual tests
# ---------------------------------------------------------------------------

"""
Test @register_multivariate with a 3-argument quadratic function as objective.
Compares ExaModels gradient/Hessian to finite differences.
"""
function test_multivariate_objective(backend)
    @testset "register_multivariate: 3-arg objective" begin
        N = 6
        c = ExaCore(; backend = backend)
        x = variable(c, N; start = [Float64(i) for i in 1:N])

        # objective: sum_i _f3(x[i], x[i+1], x[i+2])
        objective(c, _f3(x[i], x[i + 1], x[i + 2]) for i in 1:N-2)

        m = ExaModel(c)
        w = WrapperNLPModel(m)
        x0 = copy(w.meta.x0)

        g_exa = NLPModels.grad(w, x0)
        g_fd  = fd_gradient(w, x0)
        @test g_exa ≈ g_fd atol = 1e-4

        # Hessian (no constraints, so y = [])
        hI = zeros(Int, w.meta.nnzh)
        hJ = zeros(Int, w.meta.nnzh)
        hV = zeros(w.meta.nnzh)
        NLPModels.hess_structure!(w, hI, hJ)
        NLPModels.hess_coord!(w, x0, Float64[], hV)

        H_exa = zeros(N, N)
        for k in 1:w.meta.nnzh
            H_exa[hI[k], hJ[k]] += hV[k]
            if hI[k] != hJ[k]
                H_exa[hJ[k], hI[k]] += hV[k]
            end
        end

        H_fd = fd_hessian_lag(w, x0, Float64[])
        @test H_exa ≈ H_fd atol = 1e-3
    end
end

"""
Test @register_multivariate with a 2-argument function as constraint.
Compares ExaModels Jacobian to finite differences.
"""
function test_multivariate_constraint(backend)
    @testset "register_multivariate: 2-arg constraint" begin
        N = 4
        c = ExaCore(; backend = backend)
        x = variable(c, N; start = [0.5 + 0.1 * Float64(i) for i in 1:N])

        # constraint: _g2(x[i], x[i+1]) ∈ [-1, 1]  for i = 1:N-1
        constraint(
            c,
            _g2(x[i], x[i + 1]) for i in 1:N-1;
            lcon = -ones(N - 1),
            ucon = ones(N - 1),
        )

        # trivial objective to make a valid NLP
        objective(c, x[i]^2 for i in 1:N)

        m = ExaModel(c)
        w = WrapperNLPModel(m)
        x0 = copy(w.meta.x0)

        # Jacobian
        jI = zeros(Int, w.meta.nnzj)
        jJ = zeros(Int, w.meta.nnzj)
        jV = zeros(w.meta.nnzj)
        NLPModels.jac_structure!(w, jI, jJ)
        NLPModels.jac_coord!(w, x0, jV)

        J_exa = zeros(N - 1, N)
        for k in 1:w.meta.nnzj
            J_exa[jI[k], jJ[k]] += jV[k]
        end

        J_fd = fd_jacobian(w, x0)
        @test J_exa ≈ J_fd atol = 1e-4

        # Hessian of Lagrangian
        y0 = randn(N - 1)
        hI = zeros(Int, w.meta.nnzh)
        hJ = zeros(Int, w.meta.nnzh)
        hV = zeros(w.meta.nnzh)
        NLPModels.hess_structure!(w, hI, hJ)
        NLPModels.hess_coord!(w, x0, y0, hV)

        H_exa = zeros(N, N)
        for k in 1:w.meta.nnzh
            H_exa[hI[k], hJ[k]] += hV[k]
            if hI[k] != hJ[k]
                H_exa[hJ[k], hI[k]] += hV[k]
            end
        end

        H_fd = fd_hessian_lag(w, x0, y0)
        @test H_exa ≈ H_fd atol = 1e-3
    end
end

"""
Test that @register_multivariate interoperates correctly with ExaModels'
native symbolic operations (Node1, Node2) in the same expression.
"""
function test_multivariate_mixed_expression(backend)
    @testset "register_multivariate: mixed with native ops" begin
        N = 6
        c = ExaCore(; backend = backend)
        x = variable(c, N; start = [0.3 * Float64(i) for i in 1:N])

        # objective: sum_i  _f3(x[i], sin(x[i+1]), x[i+2]^2)
        # This tests that NodeN can be a child of a Node1/Node2 and vice versa.
        objective(
            c,
            _f3(x[i], sin(x[i + 1]), x[i + 2]^2) for i in 1:N-2
        )

        m = ExaModel(c)
        w = WrapperNLPModel(m)
        x0 = copy(w.meta.x0)

        g_exa = NLPModels.grad(w, x0)
        g_fd  = fd_gradient(w, x0)
        @test g_exa ≈ g_fd atol = 1e-4
    end
end

"""
Run all @register_multivariate tests.
"""
function test_multivariate(backend)
    test_multivariate_objective(backend)
    test_multivariate_constraint(backend)
    test_multivariate_mixed_expression(backend)
end
