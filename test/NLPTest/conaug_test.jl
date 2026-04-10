using ExaModels
using Test
import NLPModels

# Tests for:
#   - add_con!(core, c1[i] += expr for ...) — single-generator form
#   - add_con(core, N, M) / add_con(core, r1, r2) — multi-dimensional empty constraints
#   - @add_con! two-argument form

function test_conaug_1d(backend)
    @testset "add_con! single-generator matches old syntax (1D)" begin
        N = 9

        # Old syntax: add_con!(c, g, i => expr for ...)
        c1 = ExaCore(; backend)
        c1, x1 = add_var(c1, N + 1)
        c1, g1 = add_con(c1, N; lcon = -1.0, ucon = 1.0)
        c1, _  = add_con!(c1, g1, i => x1[i] + x1[i+1] for i = 1:N)

        # New syntax: add_con!(c, g[i] += expr for ...)
        c2 = ExaCore(; backend)
        c2, x2 = add_var(c2, N + 1)
        c2, g2 = add_con(c2, N; lcon = -1.0, ucon = 1.0)
        c2, _  = add_con!(c2, g2[i] += x2[i] + x2[i+1] for i = 1:N)

        m1 = ExaModel(c1)
        m2 = ExaModel(c2)

        @test m1.meta.nnzj == m2.meta.nnzj
        @test m1.meta.nnzh == m2.meta.nnzh
        x0 = ExaModels.convert_array(rand(N + 1), backend)
        @test NLPModels.cons(m1, x0) ≈ NLPModels.cons(m2, x0)
        @test NLPModels.grad(m1, x0) ≈ NLPModels.grad(m2, x0)
    end

    @testset "@add_con! two-argument form (1D)" begin
        N = 9
        c = ExaCore(; backend)
        @add_var(c, x, N + 1)
        @add_con(c, g, Constant(0) for _ = 1:N; lcon = -1.0, ucon = 1.0)
        @add_con!(c, g[i] += x[i] + x[i+1] for i = 1:N)

        m = ExaModel(c)
        x0 = ExaModels.convert_array(ones(N + 1), backend)
        g_vals = NLPModels.cons(m, x0)
        @test all(≈(2.0), g_vals)
    end
end

function test_conaug_2d(backend)
    @testset "add_con(core, dims...) — 2D integer dims" begin
        N, M = 3, 4
        c = ExaCore(; backend)
        c, x = add_var(c, N, M)
        c, g = add_con(c, N, M; lcon = -Inf, ucon = 0.0)
        @test Base.size(g.itr) == (N, M)
        itr = [(i, j) for i = 1:N-1, j = 1:M][:]
        c, _ = add_con!(c, g[i, j] += x[i, j] - x[i+1, j] for (i, j) in itr)
        m = ExaModel(c)
        @test m.meta.ncon == N * M
        @test m.meta.nnzj == length(itr) * 2
    end

    @testset "add_con(core, dims...) — range-based 2D dims" begin
        N, M = 3, 4
        r1, r2 = 1:N, 2:M+1
        c = ExaCore(; backend)
        c, x = add_var(c, N, M)
        c, g = add_con(c, r1, r2; lcon = -Inf, ucon = 0.0)
        @test Base.size(g.itr) == (length(r1), length(r2))
        @test g.itr.ns == (r1, r2)

        itr = [(i, j) for i in r1, j in r2 if i < N][:]
        c, _ = add_con!(c, g[i, j] += x[i, j-1] - x[i+1, j-1] for (i, j) in itr)
        m = ExaModel(c)
        @test m.meta.ncon == length(r1) * length(r2)
        @test m.meta.nnzj == length(itr) * 2
    end

    @testset "add_con(core, dims...) — 3D integer dims" begin
        N, M, K = 2, 3, 4
        c = ExaCore(; backend)
        c, x = add_var(c, N * M * K)  # need at least one variable
        c, g = add_con(c, N, M, K; lcon = 0.0, ucon = 0.0)
        @test Base.size(g.itr) == (N, M, K)
        @test length(g.itr) == N * M * K
        m = ExaModel(c)
        @test m.meta.ncon == N * M * K
    end

    @testset "range-based 2D matches integer 2D (same result)" begin
        N, M = 3, 4
        itr = [(i, j) for i = 1:N-1, j = 1:M][:]

        # Integer dims
        c1 = ExaCore(; backend)
        c1, x1 = add_var(c1, N, M)
        c1, g1 = add_con(c1, N, M; lcon = -1.0, ucon = 1.0)
        c1, _  = add_con!(c1, g1[i, j] += x1[i, j] - x1[i+1, j] for (i, j) in itr)

        # Range dims (1:N, 1:M) — same as integer N, M
        c2 = ExaCore(; backend)
        c2, x2 = add_var(c2, N, M)
        c2, g2 = add_con(c2, 1:N, 1:M; lcon = -1.0, ucon = 1.0)
        c2, _  = add_con!(c2, g2[i, j] += x2[i, j] - x2[i+1, j] for (i, j) in itr)

        m1 = ExaModel(c1)
        m2 = ExaModel(c2)
        @test m1.meta.nnzj == m2.meta.nnzj
        @test m1.meta.nnzh == m2.meta.nnzh

        x0 = ExaModels.convert_array(rand(N * M), backend)
        @test NLPModels.cons(m1, x0) ≈ NLPModels.cons(m2, x0)
    end
end

function test_conaug(backend)
    @testset "1D constraint augmentation" begin
        test_conaug_1d(backend)
    end
    @testset "Multi-dimensional constraint" begin
        test_conaug_2d(backend)
    end
end
