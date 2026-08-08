module RecorderTest

using Test, Random
using ExaModels
using NLPModels
using SparseArrays
import JuMP, PowerModels, Downloads
import LuksanVlcekBenchmark as LVB
import COPSBenchmark
import JuliaC, MadNLP, CNLPModels

const TMPDIR = mktempdir()

# Direct (non-recorder) reference builders, reused from the repo's own tests.
include("../NLPTest/luksan.jl")   # luksan_vlcek_* helpers + _exa_luksan_vlcek_model
include("../NLPTest/power.jl")    # parse_ac_power_data + __exa_ac_power_model

include("recorded_models.jl")
include("aot_libraries.jl")

# The canonical 1-D LuksanVlcek example (docs/src/gpu.jl), kept from the PoC:
# helpers shared, paths independent.

function direct_model(N; T = Float64)
    c = ExaCore(T; concrete = Val(true))
    @add_var(c, x, N; start = (LVB.rosenrock_start(i) for i = 1:N))
    @add_con(c, LVB.rosenrock_constraint(x, i) for i = 1:N-2)
    @add_obj(c, LVB.rosenrock_objective(x, i) for i = 1:N-1)
    return ExaModel(c)
end

lv_tape() = let data = DataTracer((; N = 4)), c = ExaTape()
    @add_var(c, x, data.N; start = (LVB.rosenrock_start(i) for i = 1:data.N))
    @add_con(c, LVB.rosenrock_constraint(x, i) for i = 1:data.N-2)
    @add_obj(c, LVB.rosenrock_objective(x, i) for i = 1:data.N-1)
    c
end

_sp_jac(m, x) = sparse(jac_structure(m)..., jac_coord(m, x), m.meta.ncon, m.meta.nvar)
_sp_hess(m, x, y) = sparse(hess_structure(m)..., hess_coord(m, x, y), m.meta.nvar, m.meta.nvar)

function compare_models(m_rec, m_ref; dense = false)
    rng = Random.MersenneTwister(42)

    @test m_rec.meta.nvar == m_ref.meta.nvar
    @test m_rec.meta.ncon == m_ref.meta.ncon
    @test m_rec.meta.minimize == m_ref.meta.minimize
    @test m_rec.meta.x0 ≈ m_ref.meta.x0 rtol = 1e-14
    @test m_rec.meta.lvar == m_ref.meta.lvar
    @test m_rec.meta.uvar == m_ref.meta.uvar
    @test m_rec.meta.lcon == m_ref.meta.lcon
    @test m_rec.meta.ucon == m_ref.meta.ucon

    x = randn(rng, m_ref.meta.nvar)
    y = randn(rng, m_ref.meta.ncon)

    @test obj(m_rec, x) ≈ obj(m_ref, x) rtol = 1e-12
    @test grad(m_rec, x) ≈ grad(m_ref, x) rtol = 1e-12
    @test cons(m_rec, x) ≈ cons(m_ref, x) rtol = 1e-12
    if dense
        # Used where recorded and direct models legitimately differ in tree
        # representation (e.g. parameter vs baked constant): sparse() sums
        # duplicate coordinates, so this compares the assembled operators.
        @test _sp_jac(m_rec, x) ≈ _sp_jac(m_ref, x) rtol = 1e-12
        @test _sp_hess(m_rec, x, y) ≈ _sp_hess(m_ref, x, y) rtol = 1e-12
    else
        @test jac_structure(m_rec) == jac_structure(m_ref)
        @test hess_structure(m_rec) == hess_structure(m_ref)
        @test jac_coord(m_rec, x) ≈ jac_coord(m_ref, x) rtol = 1e-12
        @test hess_coord(m_rec, x, y) ≈ hess_coord(m_ref, x, y) rtol = 1e-12
    end
end

function runtests()
    @testset "Recorder" begin
        tape = lv_tape()

        @testset "replay matches direct build (N = $N)" for N in (10, 500)
            m_rec = ExaModel(tape, (; N = N))
            m_ref = direct_model(N)
            compare_models(m_rec, m_ref)
        end

        @testset "replay is type-stable" begin
            core = @inferred ExaModels.replay(tape, (; N = 10))
            @test core isa ExaCore{Float64}
        end

        @testset "element type is a replay-time choice" begin
            m = ExaModel(tape, (; N = 10); T = Float32)
            @test obj(m, ones(Float32, 10)) isa Float32
        end

        @testset "built models are independent of later replays" begin
            m10 = ExaModel(tape, (; N = 10))
            x = ones(10)
            o1 = obj(m10, x)
            ExaModels.replay(tape, (; N = 100))  # rebinds the tape's variable refs
            @test obj(m10, x) == o1
        end

        @testset "tree-built tape matches closure-built tape" begin
            # The Python-facing path: expressions arrive as pre-built Node
            # trees with unbound-handle sentinels, no Julia closures anywhere.
            data = DataTracer((; N = 4))
            ct = ExaTape()
            ct, xt = add_var(ct, data.N; start = -0.5)
            i = ExaModels.DataSource()
            ct, _ = add_con(ct,
                3xt[i+1]^3 + 2xt[i+2] - 5 + sin(xt[i+1] - xt[i+2]) * sin(xt[i+1] + xt[i+2]) +
                4xt[i+1] - xt[i] * exp(xt[i] - xt[i+1]) - 3,
                1:data.N-2)
            ct, _ = add_obj(ct, 100 * (xt[i-1]^2 - xt[i])^2 + (xt[i-1] - 1)^2, 2:data.N)

            cc = let data = DataTracer((; N = 4)), c = ExaTape()
                @add_var(c, x, data.N; start = -0.5)
                @add_con(c, 3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) +
                            4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3 for i = 1:data.N-2)
                @add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:data.N)
                c
            end

            for N in (10, 200)
                mt = ExaModel(ct, (; N = N))
                mc = ExaModel(cc, (; N = N))
                rng = Random.MersenneTwister(3)
                xr = randn(rng, N)
                yr = randn(rng, N - 2)
                @test obj(mt, xr) == obj(mc, xr)
                @test grad(mt, xr) == grad(mc, xr)
                @test cons(mt, xr) == cons(mc, xr)
                @test jac_structure(mt) == jac_structure(mc)
                @test jac_coord(mt, xr) == jac_coord(mc, xr)
                @test hess_structure(mt) == hess_structure(mc)
                @test hess_coord(mt, xr, yr) == hess_coord(mc, xr, yr)
            end
            core = @inferred ExaModels.replay(ct, (; N = 10))
            @test core isa ExaCore{Float64}
        end

        @testset "structure guardrails" begin
            @test_throws RecorderStructureError let data = DataTracer((; N = 4)), c = ExaTape()
                data.N > 5 && error("unreachable")
                c
            end
            @test_throws RecorderStructureError let data = DataTracer((; N = 4)), c = ExaTape()
                total = 0
                for i in 1:data.N
                    total += i
                end
                c
            end
        end

        @testset "LuksanVlcek set: $(case.name)" for case in LV_CASES
            t = case.tape()
            dense = haskey(case, :dense) && case.dense
            for N in case.sizes
                m_rec = ExaModel(t, (; N = N))
                m_ref = lv_direct(case.name, N)
                compare_models(m_rec, m_ref; dense = dense)
            end
            core = @inferred ExaModels.replay(t, (; N = case.sizes[1]))
            @test core isa ExaCore{Float64}
        end

        @testset "2-D Luksan with product generators" begin
            t = luksan2d_tape()
            for (N, M) in ((10, 3), (30, 1))
                m_rec = ExaModel(t, (; N = N, M = M); prod = true)
                m_ref, _, _ = _exa_luksan_vlcek_model(nothing, N; M = M)
                compare_models(m_rec, m_ref)
            end
            core = @inferred ExaModels.replay(t, (; N = 6, M = 2))
            @test core isa ExaCore{Float64}
        end

        @testset "COPS: $name" for (name, sizes) in (
            ("chain", (64, 200)),
            ("camshape", (50, 200)),
        )
            t = getfield(COPSBenchmark, Symbol(name, :_tape))()
            for n in sizes
                m_rec = ExaModel(t, (; n = n))
                m_ref = getfield(COPSBenchmark, Symbol(name, :_model))(
                    COPSBenchmark.ExaModelsBackend(), n)
                compare_models(m_rec, m_ref; dense = true)
            end
            core = @inferred ExaModels.replay(t, (; n = sizes[1]))
            @test core isa ExaCore{Float64}
        end

        @testset "AC power flow: one tape, several grids" begin
            data3 = parse_ac_power_data(get_power_case("pglib_opf_case3_lmbd.m"))
            data14 = parse_ac_power_data(get_power_case("pglib_opf_case14_ieee.m"))
            t = opf_build(ExaTape(), DataTracer(data3))
            for data in (data3, data14)
                m_rec = ExaModel(t, data; prod = true)
                m_ref, _, _ = __exa_ac_power_model(nothing, data)
                compare_models(m_rec, m_ref)
            end
            core = @inferred ExaModels.replay(t, data3)
            @test core isa ExaCore{Float64}
        end

        aot_library_tests()
    end
end

end # module RecorderTest
