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

function exa_luksan_vlcek_parametric(backend, N; M=1, use_parameters=true, param_values=nothing)
    c = ExaCore(backend = backend)
    x = variable(c, N, M; start = [luksan_vlcek_x0(i) for i = 1:N, j = 1:M])
    
    if use_parameters
        θ = parameter(c, zeros(7))
        if !isnothing(param_values)
            c.θ .= ExaModels.convert_array(param_values, backend)
        else
            c.θ .= ExaModels.convert_array([100.0, 1.0, 3.0, 2.0, 5.0, 4.0, 3.0], backend)
        end

        s = constraint(c, luksan_vlcek_con1_param(x, θ, i, j) for i = 1:(N-2), j = 1:M)
        constraint!(c, s, (i, j) => luksan_vlcek_con2_param(x, θ, i, j) for i = 1:(N-2), j = 1:M)
        objective(c, luksan_vlcek_obj_param(x, θ, i, j) for i = 2:N, j = 1:M)
    else
        if isnothing(param_values)
            param_values = [100.0, 1.0, 3.0, 2.0, 5.0, 4.0, 3.0]
        end

        p1 = param_values[1]
        p2 = param_values[2]
        p3 = param_values[3]
        p4 = param_values[4]
        p5 = param_values[5]
        p6 = param_values[6]
        p7 = param_values[7]

        obj_func = (x, i, j) -> p1 * (x[i-1, j]^2 - x[i, j])^2 + (x[i-1, j] - p2)^2
        con1_func = (x, i, j) -> p3 * x[i+1, j]^3 + p4 * x[i+2, j] - p5
        con2_func = (x, i, j) -> sin(x[i+1, j] - x[i+2, j])sin(x[i+1, j] + x[i+2, j]) + p6 * x[i+1, j] - x[i, j]exp(x[i, j] - x[i+1, j]) - p7
        
        s = constraint(c, con1_func(x, i, j) for i = 1:(N-2), j = 1:M)
        constraint!(c, s, (i, j) => con2_func(x, i, j) for i = 1:(N-2), j = 1:M)
        objective(c, obj_func(x, i, j) for i = 2:N, j = 1:M)
    end
    
    return ExaModel(c; prod = true), c, (x,), (s,)
end

function test_function_evaluations(model1, core1, model2, core2)
    x_test = ExaModels.convert_array(randn(core1.nvar),core2.backend)

    obj1 = NLPModels.obj(model1, x_test)
    obj2 = NLPModels.obj(model2, x_test)
    @test obj1 ≈ obj2 atol=1e-12

    if core1.ncon > 0
        con1 = NLPModels.cons(model1, x_test)
        con2 = NLPModels.cons(model2, x_test)
        @test con1 ≈ con2 atol=1e-12
    end

    grad1 = NLPModels.grad(model1, x_test)
    grad2 = NLPModels.grad(model2, x_test)
    @test grad1 ≈ grad2 atol=1e-12
    
    u = ExaModels.convert_array(randn(core1.nvar),core2.backend)
    v = ExaModels.convert_array(randn(core1.ncon),core2.backend)
    jprod_param = NLPModels.jprod(model2, x_test, u)
    jprod_orig = NLPModels.jprod(model1, x_test, u)
    @test jprod_param ≈ jprod_orig atol=1e-12

    jtprod_param = NLPModels.jtprod(model2, x_test, v)
    jtprod_orig = NLPModels.jtprod(model1, x_test, v)
    @test jtprod_param ≈ jtprod_orig atol=1e-12

    y_test = ExaModels.convert_array(randn(core1.ncon),core2.backend)
    hprod_param = NLPModels.hprod(model2, x_test, y_test, u)
    hprod_orig = NLPModels.hprod(model1, x_test, y_test, u)
    @test hprod_param ≈ hprod_orig atol=1e-12
end

function test_parametric_vs_nonparametric(backend)
    @testset "Parametric" begin
        @testset "Metadata" begin
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend, 4, M=3, use_parameters=true)
            m_nonparam, c_nonparam, _, _ = exa_luksan_vlcek_parametric(
                backend, 4, M=3, use_parameters=false)
            @test c_param.nvar == c_nonparam.nvar
            @test c_param.ncon == c_nonparam.ncon
            @test ExaModels.depth(c_param.obj) == ExaModels.depth(c_nonparam.obj) 
            @test ExaModels.depth(c_param.con) == ExaModels.depth(c_nonparam.con)
            @test c_param.npar == 7
            @test c_nonparam.npar == 0
        end

        @testset "Default" begin
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=true)
            m_nonparam, c_nonparam, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=false)
            test_function_evaluations(m_param, c_param, m_nonparam, c_nonparam)
        end
        @testset "Modified Parameters objective" begin
            modified_params1 = [200.0, 2.0, 3.0, 2.0, 5.0, 4.0, 3.0]
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=true, param_values=modified_params1)
            m_nonparam, c_nonparam, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=false, param_values=modified_params1)
            test_function_evaluations(m_param, c_param, m_nonparam, c_nonparam)
        end
        
        @testset "Modified Parameters constraints" begin
            modified_params2 = [100.0, 1.0, 6.0, 4.0, 10.0, -8.0, 6.0]
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=true, param_values=modified_params2)
            m_nonparam, c_nonparam, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=false, param_values=modified_params2)
            test_function_evaluations(m_param, c_param, m_nonparam, c_nonparam)
        end
        @testset "Modified Parameters all" begin
            modified_params3 = [150.0, 0.5, 2.5, 1.5, 7.5, 3.5, 4.5]
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=true, param_values=modified_params3)
            m_nonparam, c_nonparam, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=false, param_values=modified_params3)
            test_function_evaluations(m_param, c_param, m_nonparam, c_nonparam)
        end
        
        @testset "Modify after build" begin
            m_param, c_param, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=true)
            new_params = [75.0, 1.5, 4.0, 3.0, 6.0, 5.0, 2.0]
            c_param.θ .= ExaModels.convert_array(new_params, backend)
            m_nonparam, c_nonparam, _, _ = exa_luksan_vlcek_parametric(
                backend, 3, M=2, use_parameters=false, param_values=new_params)
            test_function_evaluations(m_param, c_param, m_nonparam, c_nonparam)
        end
    end
end 