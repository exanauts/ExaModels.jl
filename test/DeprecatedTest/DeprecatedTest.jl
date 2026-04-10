module DeprecatedTest

using Test
using ExaModels
import NLPModels

function runtests()
    @testset "Deprecated API (LegacyExaCore)" begin

        @testset "ExaCore() returns LegacyExaCore by default" begin
            c = ExaCore()
            @test c isa LegacyExaCore
        end

        @testset "ExaCore(concrete=Val(true)) returns ExaCore" begin
            c = ExaCore(concrete = Val(true))
            @test c isa ExaCore
            @test !(c isa LegacyExaCore)
        end

        @testset "ExaCore(Float32) returns LegacyExaCore{Float32}" begin
            c = ExaCore(Float32)
            @test c isa LegacyExaCore{Float32}
        end

        @testset "variable / objective / ExaModel round-trip" begin
            c = ExaCore()
            x = variable(c, 10; start = 1.0)
            @test x isa ExaModels.Variable
            objective(c, x[i]^2 for i = 1:10)
            m = ExaModel(c)
            @test m.meta.nvar == 10
            @test NLPModels.obj(m, ones(10)) ≈ 10.0
        end

        @testset "constraint / constraint!" begin
            c = ExaCore()
            x = variable(c, 10; lvar = 0.0, uvar = 1.0)
            con = constraint(c, x[i] + x[i+1] for i = 1:9; lcon = -2.0, ucon = 2.0)
            @test con isa ExaModels.Constraint
            constraint!(c, con, i => x[i+1] for i = 1:9)
            m = ExaModel(c)
            @test m.meta.nvar == 10
            @test m.meta.ncon == 9
        end

        @testset "parameter" begin
            c = ExaCore()
            x = variable(c, 5; start = 1.0)
            θ = parameter(c, ones(5))
            @test θ isa ExaModels.Parameter
            objective(c, (x[i] - θ[i])^2 for i = 1:5)
            m = ExaModel(c)
            @test m.meta.nvar == 5
            @test NLPModels.obj(m, ones(5)) ≈ 0.0 atol = 1e-12
        end

        @testset "subexpr" begin
            c = ExaCore()
            x = variable(c, 10; start = 1.0)
            s = subexpr(c, x[i]^2 for i = 1:10)
            @test s isa ExaModels.Expression
            objective(c, s[i] + s[i+1] for i = 1:9)
            m = ExaModel(c)
            @test m.meta.nvar == 10  # reduced form: no auxiliary variables
            @test NLPModels.obj(m, ones(10)) ≈ 18.0  # sum_{i=1}^{9} (1^2 + 1^2) = 18
        end

        @testset "add_var / add_obj still work on LegacyExaCore" begin
            c = ExaCore()
            c, x = add_var(c, 5; start = 2.0)
            @test x isa ExaModels.Variable
            c, _ = add_obj(c, x[i]^2 for i = 1:5)
            m = ExaModel(c)
            @test NLPModels.obj(m, 2.0 * ones(5)) ≈ 20.0
        end

        @testset "property forwarding to inner ExaCore" begin
            c = ExaCore()
            variable(c, 3)
            @test c.nvar == 3
            @test c.backend === nothing
        end

    end
end

end # module DeprecatedTest
