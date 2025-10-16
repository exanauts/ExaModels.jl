module NLPTest

using ExaModels
using Downloads, Test
using NLPModels, NLPModelsJuMP, NLPModelsIpopt, NLPModelsTest
using JuMP, PowerModels, MadNLP, Percival

import ..BACKENDS

const NLP_TEST_ARGUMENTS = [
    ("luksan_expr", 3),
    ("luksan_expr", 20),
    ("luksan_struct", 3),
    ("luksan_struct", 20),
    ("luksan_vlcek", 3),
    ("luksan_vlcek", 20),
    ("ac_power", "pglib_opf_case3_lmbd.m"),
    ("ac_power", "pglib_opf_case14_ieee.m"),
]

const SOLVERS = [
    ("ipopt", nlp -> ipopt(nlp; print_level = 0)),
    ("madnlp", nlp -> madnlp(nlp; print_level = MadNLP.ERROR)),
    ("percival", nlp -> percival(nlp)),
]

const EXCLUDE1 = [("ac_power", "percival"), ("struct_ac_power", "percival")]
const EXCLUDE2 = []

for backend in BACKENDS
    if "oneAPIBackend()" == string(backend)
        push!(EXCLUDE2, ("percival", backend))
    end
    if "OpenCLBackend()" == string(backend)
        push!(EXCLUDE3, ("percival", backend))
    end
end

include("luksan.jl")
include("power.jl")
include("luksan_struct.jl")
include("luksan_expr.jl")
include("parameter_test.jl")

# first argument must be an examodel (this makes testing expression() models easier)
function test_nlp((m1, var1, con1), (m2, var2, con2); full = false)
    @testset "NLP meta tests" begin
        list = [:x0, :y0]
        if full
            append!(list, [:nnzj, :nnzh])
        end

        @testset "nvar" begin
            @test length(var1) == length(var2)
        end
        for field in [:lvar, :uvar]
            @testset "$field" begin
                m1_field = getfield(m1.meta, field)[var1]
                m2_field = getfield(m2.meta, field)[var2]
                @test m1_field == m2_field
            end
        end
        @testset "ncon" begin
            @test length(con1) == length(con2)
        end
        for field in [:lcon, :ucon]
            @testset "$field" begin
                m1_field = getfield(m1.meta, field)[con1]
                m2_field = getfield(m2.meta, field)[con2]
                @test m1_field == m2_field
            end
        end
    end
end

function test_nlp_callbacks(x, y, (m1, var1, con1), (m2, var2, con2); full=false)
    @testset "NLP callback tests" begin
        u = randn(eltype(m1.meta.x0), m1.meta.nvar)
        v = randn(eltype(m1.meta.x0), m1.meta.ncon)

        m2_examodel = length(var2) != m2.meta.nvar
        x2 = m2_examodel ? x : x[var2]
        y2 = m2_examodel ? y : y[con2]
        u2 = m2_examodel ? u : u[var2]
        v2 = m2_examodel ? v : v[con2]
        @test NLPModels.obj(m1, x) ≈ NLPModels.obj(m2, x2) atol = 1e-6
        @test NLPModels.cons(m1, x)[con1] ≈ NLPModels.cons(m2, x2)[con2] atol = 1e-6
        @test NLPModels.grad(m1, x)[var1] ≈ NLPModels.grad(m2, x2)[var2] atol = 1e-6
        @test NLPModels.jprod(m1, x, u)[con1] ≈ NLPModels.jprod(m2, x2, u2)[con2] atol = 1e-6
        @test NLPModels.jtprod(m1, x, v)[var1] ≈ NLPModels.jtprod(m2, x2, v2)[var2] atol = 1e-6
        @test NLPModels.hprod(m1, x, y, u)[var1] ≈ NLPModels.hprod(m2, x2, y2, u2)[var2] atol = 1e-6

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

            NLPModels.jac_coord!(m1, x, jac_buffer1)
            NLPModels.jac_coord!(m2, x, jac_buffer2)
            NLPModels.hess_coord!(m1, x, y, hess_buffer1)
            NLPModels.hess_coord!(m2, x, y, hess_buffer2)
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

function test_nlp_solution((r1, var1, con1), (r2, var2, con2))
    @testset "solution test" begin
        @test r1.status == r2.status
        @test r1.solution[var1] ≈ r2.solution[var2]
        @test r1.multipliers[con1] ≈ r2.multipliers[con2]
        @test r1.multipliers_L ≈ r2.multipliers_L
        @test r1.multipliers_U ≈ r2.multipliers_U
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

fake_set_to_ind(s, len) = sort([x for x in 1:len if !(x in s)])

function runtests()
    @testset "NLP test" begin
        for backend in BACKENDS
            @testset "$backend" begin
                for (name, args) in NLP_TEST_ARGUMENTS
                    @testset "$name $args" begin

                        exa_model = getfield(@__MODULE__, Symbol("_exa_$(name)_model"))
                        jump_model = getfield(@__MODULE__, Symbol("_jump_$(name)_model"))

                        m, vars0, cons0 = exa_model(nothing, args)
                        conind0 = fake_set_to_ind(m.fake_con_inds, m.meta.ncon)
                        varind0 = fake_set_to_ind(m.fake_var_inds, m.meta.nvar)
                        m0 = WrapperNLPModel(m)

                        m, vars2, cons2 = jump_model(nothing, args)
                        m2 = MathOptNLPModel(m)
                        conind2 = fake_set_to_ind(Set(), m2.meta.ncon)
                        varind2 = fake_set_to_ind(Set(), m2.meta.nvar)

                        set_optimizer(m, MadNLP.Optimizer)
                        set_optimizer_attribute(m, "print_level", MadNLP.ERROR)
                        optimize!(m)

                        m, vars1, cons1 = exa_model(backend, args)
                        conind1 = fake_set_to_ind(m.fake_con_inds, m.meta.ncon)
                        varind1 = fake_set_to_ind(m.fake_var_inds, m.meta.nvar)
                        m1 = WrapperNLPModel(m)

                        @testset "Backend test" begin
                            test_nlp((m0, varind0, conind0), (m1, varind1, conind1); full = true)
                            test_nlp_callbacks(m0.meta.x0, m0.meta.y0, (m0, varind0, conind0), (m1, varind1, conind1); full = true)
                        end
                        @info "backend done"
                        @testset "Comparison to JuMP" begin
                            test_nlp((m1, varind1, conind1), (m2, varind2, conind2); full = false)
                            for (sname, solver) in SOLVERS
                                if (name, sname) in EXCLUDE1 || (sname, backend) in EXCLUDE2
                                    continue
                                end

                                r1 = solver(m1)
                                r2 = solver(m2)

                                @testset "$sname" begin
                                    test_nlp_solution((r1, varind1, conind1), (r2, varind2, conind2))
                                end
                                if sname == SOLVERS[1][1]
                                    test_nlp((m1, varind1, conind1), (m2, varind2, conind2); full = false)
                                    test_nlp_callbacks(r1.solution, r1.multipliers, (m1, varind1, conind1), (m2, varind2, conind2); full = false)
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
