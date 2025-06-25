using ExaModels
using Test

function luksan_vlcek_obj_param(x, θ, i, j)
    # θ[1] = 100, θ[2] = 1
    return θ[1] * (x[i-1, j]^2 - x[i, j])^2 + (x[i-1, j] - θ[2])^2
end

function luksan_vlcek_con1_param(x, θ, i, j)
    # θ[3] = 3, θ[4] = 2, θ[5] = 5
    return θ[3] * x[i+1, j]^3 + θ[4] * x[i+2, j] - θ[5]
end

function luksan_vlcek_con2_param(x, θ, i, j)
    # θ[6] = 4, θ[7] = 3
    return sin(x[i+1, j] - x[i+2, j])sin(x[i+1, j] + x[i+2, j]) + θ[6] * x[i+1, j] -
           x[i, j]exp(x[i, j] - x[i+1, j]) - θ[7]
end

function exa_luksan_vlcek_parametric(backend, N; M = 1)
    c = ExaCore(backend = backend)
    x = variable(c, N, M; start = [luksan_vlcek_x0(i) for i = 1:N, j = 1:M])
    
    param_values = [100.0, 1.0, 3.0, 2.0, 5.0, 4.0, 3.0]
    θ = parameter(c, param_values)
    
    s = constraint(c, luksan_vlcek_con1_param(x, θ, i, j) for i = 1:(N-2), j = 1:M)
    constraint!(c, s, (i, j) => luksan_vlcek_con2_param(x, θ, i, j) for i = 1:(N-2), j = 1:M)
    objective(c, luksan_vlcek_obj_param(x, θ, i, j) for i = 2:N, j = 1:M)

    return ExaModel(c; prod = true), (x,), (s,), θ
end

function test_parametric_vs_nonparametric(backend)
    @testset "Parametric vs Non-parametric Luksan Model" begin
        N = 5
        M = 1

        m_param, vars_param, cons_param, θ = exa_luksan_vlcek_parametric(backend, N; M = M)
        m_orig, vars_orig, cons_orig = _exa_luksan_vlcek_model(backend, N; M = M)

        @testset "Model Structure" begin
            @test m_param.meta.nvar == m_orig.meta.nvar
            @test m_param.meta.ncon == m_orig.meta.ncon
            @test m_param.meta.nnzj == m_orig.meta.nnzj
            @test m_param.meta.nnzh == m_orig.meta.nnzh
            @test m_param.meta.x0 ≈ m_orig.meta.x0
            @test m_param.meta.lvar ≈ m_orig.meta.lvar
            @test m_param.meta.uvar ≈ m_orig.meta.uvar
            @test m_param.meta.y0 ≈ m_orig.meta.y0
            @test m_param.meta.lcon ≈ m_orig.meta.lcon
            @test m_param.meta.ucon ≈ m_orig.meta.ucon
        end
        
        x_test = copy(m_orig.meta.x0)
        y_test = randn(m_orig.meta.ncon)
        @testset "Function Evaluations" begin
            obj_param = NLPModels.obj(m_param, x_test)
            obj_orig = NLPModels.obj(m_orig, x_test)
            @test obj_param ≈ obj_orig atol=1e-12
            
            cons_param = NLPModels.cons(m_param, x_test)
            cons_orig = NLPModels.cons(m_orig, x_test)
            @test cons_param ≈ cons_orig atol=1e-12
            
            grad_param = NLPModels.grad(m_param, x_test)
            grad_orig = NLPModels.grad(m_orig, x_test)
            @test grad_param ≈ grad_orig atol=1e-12
            
            if m_orig.meta.ncon > 0
                u = randn(m_orig.meta.nvar)
                v = randn(m_orig.meta.ncon)
                
                jprod_param = NLPModels.jprod(m_param, x_test, u)
                jprod_orig = NLPModels.jprod(m_orig, x_test, u)
                @test jprod_param ≈ jprod_orig atol=1e-12
                
                jtprod_param = NLPModels.jtprod(m_param, x_test, v)
                jtprod_orig = NLPModels.jtprod(m_orig, x_test, v)
                @test jtprod_param ≈ jtprod_orig atol=1e-12
                
                hprod_param = NLPModels.hprod(m_param, x_test, y_test, u)
                hprod_orig = NLPModels.hprod(m_orig, x_test, y_test, u)
                @test hprod_param ≈ hprod_orig atol=1e-12
            end
        end
        
        @testset "Parameter Modification" begin
            original_param_values = copy(m_param.θ)
            
            m_param.θ[1] = 200.0
            
            obj_modified = NLPModels.obj(m_param, x_test)
            obj_orig = NLPModels.obj(m_orig, x_test)
            
            @test obj_modified ≠ obj_orig
            
            m_param.θ[1] = original_param_values[1]
            obj_restored = NLPModels.obj(m_param, x_test)
            @test obj_restored ≈ obj_orig atol=1e-12
        end
    end
end 