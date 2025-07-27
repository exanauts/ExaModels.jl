module NLPTest

using ExaModels
using Downloads, Test
using NLPModels, NLPModelsJuMP, NLPModelsIpopt, NLPModelsTest
using JuMP, PowerModels, MadNLP, Percival

import ..BACKENDS

const NLP_TEST_ARGUMENTS = [
    ("luksan_vlcek", 3),
    ("luksan_vlcek", 20),
    ("ac_power", "pglib_opf_case3_lmbd.m"),
    ("ac_power", "pglib_opf_case14_ieee.m"),
    ("struct_ac_power", "pglib_opf_case3_lmbd.m"),
    ("struct_ac_power", "pglib_opf_case14_ieee.m"),
]

const SOLVERS = [
    ("ipopt", nlp -> ipopt(nlp; print_level = 0)),
    ("madnlp", nlp -> madnlp(nlp; print_level = MadNLP.ERROR)),
    ("percival", nlp -> percival(nlp)),
]

const EXCLUDE1 = [("ac_power", "percival")]
const EXCLUDE2 = []

for backend in BACKENDS
    if "oneAPIBackend()" == string(backend)
        push!(EXCLUDE2, ("percival", backend))
    end
end

include("luksan.jl")
include("power.jl")
include("parameter_test.jl")
include("power_struct.jl")

function test_nlp(m1, m2; full = false)
    @testset "NLP meta tests" begin
        list = [:nvar, :ncon, :x0, :lvar, :uvar, :y0, :lcon, :ucon]

        if full
            append!(list, [:nnzj, :nnzh])
        end

        for field in list
            @testset "$field" begin
                @test getfield(m1.meta, field) == getfield(m2.meta, field)
            end
        end
    end

    @testset "NLP callback tests" begin
        x0 = copy(m2.meta.x0)
        y0 = randn(eltype(m2.meta.x0), m2.meta.ncon)
        u = randn(eltype(m2.meta.x0), m2.meta.nvar)
        v = randn(eltype(m2.meta.x0), m2.meta.ncon)

        @test NLPModels.obj(m1, x0) ≈ NLPModels.obj(m2, x0) atol = 1e-6
        @test NLPModels.cons(m1, x0) ≈ NLPModels.cons(m2, x0) atol = 1e-6
        @test NLPModels.grad(m1, x0) ≈ NLPModels.grad(m2, x0) atol = 1e-6
        @test NLPModels.jprod(m1, x0, u) ≈ NLPModels.jprod(m2, x0, u) atol = 1e-6
        @test NLPModels.jtprod(m1, x0, v) ≈ NLPModels.jtprod(m2, x0, v) atol = 1e-6
        @test NLPModels.hprod(m1, x0, y0, u) ≈ NLPModels.hprod(m2, x0, y0, u) atol = 1e-6

        if full
            jac_buffer1 = zeros(m1.meta.nnzj)
            jac_buffer2 = zeros(m2.meta.nnzj)
            jac_I_buffer1 = zeros(Int, m1.meta.nnzj)
            jac_I_buffer2 = zeros(Int, m2.meta.nnzj)
            jac_J_buffer1 = zeros(Int, m1.meta.nnzj)
            jac_J_buffer2 = zeros(Int, m2.meta.nnzj)

            hess_buffer1 = zeros(m1.meta.nnzh)
            hess_buffer2 = zeros(m2.meta.nnzh)
            hess_I_buffer1 = zeros(Int, m1.meta.nnzh)
            hess_I_buffer2 = zeros(Int, m2.meta.nnzh)
            hess_J_buffer1 = zeros(Int, m1.meta.nnzh)
            hess_J_buffer2 = zeros(Int, m2.meta.nnzh)

            NLPModels.jac_coord!(m1, x0, jac_buffer1)
            NLPModels.jac_coord!(m2, x0, jac_buffer2)
            NLPModels.hess_coord!(m1, x0, y0, hess_buffer1)
            NLPModels.hess_coord!(m2, x0, y0, hess_buffer2)
            NLPModels.jac_structure!(m1, jac_I_buffer1, jac_J_buffer1)
            NLPModels.jac_structure!(m2, jac_I_buffer2, jac_J_buffer2)
            NLPModels.hess_structure!(m1, hess_I_buffer1, hess_J_buffer1)
            NLPModels.hess_structure!(m2, hess_I_buffer2, hess_J_buffer2)

            @test jac_buffer1 ≈ jac_buffer2 atol = 1e-6
            @test hess_buffer1 ≈ hess_buffer2 atol = 1e-6
            @test jac_I_buffer1 == jac_I_buffer2
            @test jac_J_buffer1 == jac_J_buffer2
            @test hess_I_buffer1 == hess_I_buffer2
            @test hess_J_buffer1 == hess_J_buffer2
        end
    end
end

function test_nlp_solution(result1, result2)

    @testset "solution test" begin
        @test result1.status == result2.status
        for field in [:solution, :multipliers, :multipliers_L, :multipliers_U]
            @testset "$field" begin
                @test getfield(result1, field) ≈ getfield(result2, field) atol = 1e-6
            end
        end
    end
end

dual_lb(x) = has_lower_bound(x) ? dual(LowerBoundRef(x)) : 0.0
dual_ub(x) = has_upper_bound(x) ? dual(UpperBoundRef(x)) : 0.0

function test_api(result1, vars1, cons1, vars2, cons2)
    @testset "API test" begin
        for (var1, var2) in zip(vars1, vars2)
            @test solution(result1, var1) ≈ [value(var) for var in var2] atol = 1e-6
            @test multipliers_L(result1, var1) ≈ [dual_lb(var) for var in var2] atol = 1e-6
            @test multipliers_U(result1, var1) ≈ [-dual_ub(var) for var in var2] atol = 1e-6
        end
        for (con1, con2) in zip(cons1, cons2)
            @test multipliers(result1, con1) ≈ [-dual.(con) for con in con2] atol = 1e-6
        end
    end
end

function runtests()
    @testset "NLP test" begin
        for backend in BACKENDS
            @testset "$backend" begin
                for (name, args) in NLP_TEST_ARGUMENTS
                    @testset "$name $args" begin

                        exa_model = getfield(@__MODULE__, Symbol("_exa_$(name)_model"))
                        jump_model = getfield(@__MODULE__, Symbol("_jump_$(name)_model"))

                        m, vars0, cons0 = exa_model(nothing, args)
                        m0 = WrapperNLPModel(m)

                        m, vars2, cons2 = jump_model(nothing, args)
                        m2 = MathOptNLPModel(m)

                        set_optimizer(m, MadNLP.Optimizer)
                        set_optimizer_attribute(m, "print_level", MadNLP.ERROR)
                        optimize!(m)

                        m, vars1, cons1 = exa_model(backend, args)
                        m1 = WrapperNLPModel(m)

                        @testset "Backend test" begin
                            test_nlp(m0, m1; full = true)
                        end
                        @testset "Comparison to JuMP" begin
                            test_nlp(m1, m2; full = false)

                            for (sname, solver) in SOLVERS
                                if (name, sname) in EXCLUDE1 || (sname, backend) in EXCLUDE2
                                    continue
                                end

                                result1 = solver(m1)
                                result2 = solver(m2)

                                @testset "$sname" begin
                                    test_nlp_solution(result1, result2)
                                end
                            end
                        end
                        result1 = madnlp(m1; print_level = MadNLP.ERROR)
                        test_api(result1, vars1, cons1, vars2, cons2)
                    end
                end

                m3 = WrapperNLPModel(exa_luksan_vlcek_model(nothing, 3; M = 2))
                m4 = jump_luksan_vlcek_model(nothing, 3; M = 2)

                @testset "Multi-column constraints" begin
                    test_nlp(m3, m4; full = false)
                end

                @testset "Parameter Test" begin
                    test_parametric_vs_nonparametric(backend)
                end

                @testset "Struct data" begin
                    m, vars, cons = exa_ac_power_model(backend, filename)
                end
            end
        end
    end
end

function __init__()
    if haskey(ENV, "EXA_MODELS_DEPOT")
        global TMPDIR = ENV["EXA_MODELS_DEPOT"]
    else
        global TMPDIR = tempname()
        mkdir(TMPDIR)
    end
    PowerModels.silence()
end


end # NLPTest
