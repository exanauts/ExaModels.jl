using ExaModels
using Test
import NLPModels

# Tests for Constant{T}, add_var(gen), and named variable/constraint access

function test_const(backend)
    @testset "Constant basic usage" begin
        c = ExaCore(; backend)
        N = 5
        @add_var(c, x, N; start = 1.0)

        # Const(N) keeps type stable — N is not captured in the type parameter
        @add_obj(c, Constant(N) * x[i]^2 for i in 1:N)

        m = ExaModel(c)
        @test m.meta.nvar == N
        @test m.meta.nnzo > 0

        x0 = ExaModels.convert_array(ones(N), backend)
        @test NLPModels.obj(m, x0) ≈ N * N  # sum of N*x[i]^2 = N*N*1
    end
end

function test_add_var_gen(backend)
    @testset "add_var with generator (equality constraints)" begin
        N = 4
        targets = [2.0, 3.0, 4.0, 5.0]
        pars = collect(enumerate(targets))

        c = ExaCore(; backend)
        @add_var(c, y, N; start = 1.0)

        # add_var(gen) creates variables x[i] with x[i] == targets[i]
        c, x = add_var(c, i - 0.0 for i in targets; start = 2.0)

        m = ExaModel(c)

        # N original vars + N new vars; N equality constraints from generator
        @test m.meta.nvar == 2 * N
        @test m.meta.ncon == N
        # equality constraints: lcon == ucon == 0
        @test all(iszero, m.meta.lcon)
        @test all(iszero, m.meta.ucon)
    end
end

function test_named_var_access(backend)
    @testset "Named variable access via core.name and model.name" begin
        c = ExaCore(; backend)
        N = 5
        @add_var(c, x, N; start = 0.5)
        @add_obj(c, (x[i] - 1)^2 for i in 1:N)

        m = ExaModel(c)

        # Access Variable via core.x and model.x
        @test c.x isa ExaModels.Variable
        @test m.x isa ExaModels.Variable
        @test c.x === m.x
    end

    @testset "Named constraint access via model.name" begin
        c = ExaCore(; backend)
        @add_var(c, x, 4; start = 0.0)
        @add_con(c, g, x[i] + x[i+1] for i in 1:3)

        m = ExaModel(c)

        @test c.g isa ExaModels.Constraint
        @test m.g isa ExaModels.Constraint
    end

    @testset "Named objective access via model.name" begin
        c = ExaCore(; backend)
        @add_var(c, x, 3; start = 1.0)
        @add_obj(c, f, x[i]^2 for i in 1:3)

        m = ExaModel(c)

        @test c.f isa ExaModels.Objective
        @test m.f isa ExaModels.Objective
    end
end

function test_features(backend)
    @testset "Const" begin
        test_const(backend)
    end
    @testset "add_var(gen)" begin
        test_add_var_gen(backend)
    end
    @testset "Named variable/constraint access" begin
        test_named_var_access(backend)
    end
end
