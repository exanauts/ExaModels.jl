function test_expression()
    return @testset "AD Expression Tests" begin
        @testset "Basic tests" begin
            m = ExaCore()
            v = variable(m, 5)
            e1 = expression(m, (4,), v[i] * v[i + 1] for i in 1:4)
            e2 = expression(m, (4,), e1[i] + v[i] for i in 1:4)
            c = constraint(m, e2[i] / i for i in 1:4; ucon = 10.0)
            o = objective(m, e2[i] for i in 1:4)
            mod = ExaModel(m)

            x = Float64[i for i in 1:mod.meta.nvar]

            # Test Jacobian structure
            jac_rows = zeros(Int, mod.meta.nnzj)
            jac_cols = zeros(Int, mod.meta.nnzj)
            jac_structure!(mod, jac_rows, jac_cols)
            @test mod.meta.nnzj > 0

            # Test Jacobian values
            jac_buffer = zeros(mod.meta.nnzj)
            jac_coord!(mod, x, jac_buffer)
            @test all(isfinite, jac_buffer)

            # Test Hessian structure
            hess_rows = zeros(Int, mod.meta.nnzh)
            hess_cols = zeros(Int, mod.meta.nnzh)
            hess_structure!(mod, hess_rows, hess_cols)
            @test mod.meta.nnzh > 0

            # Test Hessian values (objective only)
            hess_buffer = zeros(mod.meta.nnzh)
            hess_coord!(mod, x, hess_buffer; obj_weight = 1.0)
            @test all(isfinite, hess_buffer)

            # Test Hessian values (with constraints)
            y = ones(mod.meta.ncon)
            hess_buffer2 = zeros(mod.meta.nnzh)
            hess_coord!(mod, x, y, hess_buffer2; obj_weight = 1.0)
            @test all(isfinite, hess_buffer2)
        end

        @testset "Simple quadratic" begin
            # Test a simple quadratic: f(x) = x^2, constraint: x^2 - 1 = 0
            # Hessian of f = 2
            # Hessian of constraint = 2
            m = ExaCore()
            v = variable(m, 1)
            o = objective(m, v[1]^2)
            c = constraint(m, v[1]^2; lcon = 1.0, ucon = 1.0)
            mod = ExaModel(m)

            x = [3.0]  # arbitrary point

            # Hessian structure
            hess_rows = zeros(Int, mod.meta.nnzh)
            hess_cols = zeros(Int, mod.meta.nnzh)
            hess_structure!(mod, hess_rows, hess_cols)

            # Objective Hessian only
            hess_obj = zeros(mod.meta.nnzh)
            hess_coord!(mod, x, hess_obj; obj_weight = 1.0)
            @test any(h ≈ 2.0 for h in hess_obj)

            # Full Hessian (obj + constraints)
            y = [1.0]  # constraint multiplier
            hess_full = zeros(mod.meta.nnzh)
            hess_coord!(mod, x, y, hess_full; obj_weight = 1.0)
            @test sum(hess_full) ≈ 4.0
        end

        @testset "Expression cross-derivative" begin
            # Test with expression: e = x*y, f(e) = e, c(e) = e - 1 = 0
            m = ExaCore()
            v = variable(m, 2)
            e1 = expression(m, (1,), v[1] * v[2] for _ in 1:1)  # e = x*y
            o = objective(m, e1[1] for _ in 1:1)  # f = e = x*y
            c = constraint(m, e1[1] for _ in 1:1; lcon = 1.0, ucon = 1.0)  # c = x*y = 1
            mod = ExaModel(m)

            x = zeros(mod.meta.nvar)
            x[1:2] .= [2.0, 3.0]

            # Hessian structure
            hess_rows = zeros(Int, mod.meta.nnzh)
            hess_cols = zeros(Int, mod.meta.nnzh)
            hess_structure!(mod, hess_rows, hess_cols)

            # Objective Hessian only
            hess_obj = zeros(mod.meta.nnzh)
            hess_coord!(mod, x, hess_obj; obj_weight = 1.0)
            @test any(h ≈ 1.0 for h in hess_obj)

            # Full Hessian
            y = [1.0]
            hess_full = zeros(mod.meta.nnzh)
            hess_coord!(mod, x, y, hess_full; obj_weight = 1.0)
            @test any(h ≈ 2.0 for h in hess_full) || (sum(hess_full) ≈ 2.0)
        end

        @testset "Nested expressions" begin
            # Test nested expressions: e1 = x^2, e2 = e1 + x, f = e2
            m = ExaCore()
            v = variable(m, 1)
            e1 = expression(m, (1,), v[1]^2 for _ in 1:1)  # e1 = x^2
            e2 = expression(m, (1,), e1[1] + v[1] for _ in 1:1)  # e2 = e1 + x = x^2 + x
            o = objective(m, e2[1] for _ in 1:1)  # f = e2 = x^2 + x
            mod = ExaModel(m)

            x = zeros(mod.meta.nvar)
            x[1] = 5.0

            # Hessian structure
            hess_rows = zeros(Int, mod.meta.nnzh)
            hess_cols = zeros(Int, mod.meta.nnzh)
            hess_structure!(mod, hess_rows, hess_cols)

            # Objective Hessian only
            hess_obj = zeros(mod.meta.nnzh)
            hess_coord!(mod, x, hess_obj; obj_weight = 1.0)
            # Sum of hessian entries should be 2
            @test sum(hess_obj) ≈ 2.0
        end
    end
end
