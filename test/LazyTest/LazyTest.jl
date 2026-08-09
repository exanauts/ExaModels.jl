module LazyTest

using Test, ExaModels, NLPModels
import MadNLP
import Downloads, JuMP, PowerModels
import LuksanVlcekBenchmark
import COPSBenchmark

const LV = LuksanVlcekBenchmark
const TMPDIR = mktempdir()
include("../NLPTest/power.jl")
include("opf_model.jl")

# ── Lazy-core API semantics ──────────────────────────────────────────────────
function api_tests()
    @testset "lazy API" begin
        args = ExaModels.ArgTracer()

        # arg nodes resolve by name, or bare for the direct tracer
        @test ExaModels.resolve(args.N - 2, (; N = 10)) == 8
        @test ExaModels.resolve(ExaModels.ArgTracer() - 2, 10) == 8
        # a field reference refuses a bare value: binding is by name
        @test_throws ArgumentError ExaModels.resolve(args.N, 10)
        # structure may not depend on argument values
        @test_throws ExaModels.RecorderStructureError args.N > 5

        # sentinel-free cores materialize as a no-op: today's API verbatim
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, 4; start = 0.5)
        c, _ = add_obj(c, (x[i] - 1)^2 for i in 1:4)
        @test NLPModels.get_nvar(ExaModel(c)) == 4

        # deferred dims: one core, materialized at any size — and repeatedly
        d = ExaCore(concrete = Val(true))
        d, y = add_var(d, args.N; start = 0.0)
        d, _ = add_con(d, y[i] + y[i+1] for i in 1:args.N-1)
        d, _ = add_obj(d, (y[i] - 1)^2 for i in 1:args.N)
        @test !(d.nvar isa Int)     # the value slot is not materialized
        m10 = ExaModel(d, (; N = 10))
        m20 = ExaModel(d, (; N = 20))
        @test NLPModels.get_nvar(m10) == 10
        @test NLPModels.get_ncon(m20) == 19
        # re-materialization at the first size is stable (no state advanced)
        m10b = ExaModel(d, (; N = 10))
        @test NLPModels.get_nvar(m10b) == 10
        @test m10b.meta.x0 == m10.meta.x0

        # a field-built core refuses bare-value materialization…
        @test_throws ArgumentError ExaModel(d, 10)
        # …and refuses materialization with no args at all
        @test_throws ArgumentError ExaModel(d)

        # the tracer used directly IS the bare-value form
        e = ExaCore(concrete = Val(true))
        e, w = add_var(e, ExaModels.ArgTracer(); start = 0.0)
        @test NLPModels.get_nvar(ExaModel(e, 7)) == 7
    end
end

# ── Shared parity check ──────────────────────────────────────────────────────
# The lazy builders call the same expressions in the same add_* order as the
# direct constructors, so compressed coordinate order matches and coordinate
# vectors compare directly.
function check_parity(m, md)
    nvar = NLPModels.get_nvar(md)
    ncon = NLPModels.get_ncon(md)
    @test NLPModels.get_nvar(m) == nvar
    @test NLPModels.get_ncon(m) == ncon
    @test m.meta.minimize == md.meta.minimize
    @test m.meta.x0 ≈ md.meta.x0 atol = 1e-12
    @test NLPModels.get_lvar(m) == NLPModels.get_lvar(md)
    @test NLPModels.get_uvar(m) == NLPModels.get_uvar(md)
    @test NLPModels.get_lcon(m) == NLPModels.get_lcon(md)
    @test NLPModels.get_ucon(m) == NLPModels.get_ucon(md)
    xt = md.meta.x0 .+ 0.001 .* sin.(1:nvar)
    y0 = ones(ncon)
    @test NLPModels.obj(m, xt) ≈ NLPModels.obj(md, xt) atol = 1e-8 rtol = 1e-10
    @test NLPModels.grad(m, xt) ≈ NLPModels.grad(md, xt) atol = 1e-8 rtol = 1e-10
    if ncon > 0
        @test NLPModels.cons(m, xt) ≈ NLPModels.cons(md, xt) atol = 1e-8 rtol = 1e-10
        @test NLPModels.jac_coord(m, xt) ≈ NLPModels.jac_coord(md, xt) atol = 1e-8 rtol = 1e-10
    end
    @test NLPModels.hess_coord(m, xt, y0) ≈ NLPModels.hess_coord(md, xt, y0) atol = 1e-8 rtol = 1e-10
end

# ── Benchmark-repo core builders (consumption) ───────────────────────────────
# Full coverage parity lives in each benchmark repo's own testset; here every
# builder is consumed the way a user would, against the direct constructor.
function lv_tests()
    @testset "LuksanVlcekBenchmark cores" begin
        for name in LV.CORE_NAMES
            @testset "$name" begin
                model_f = getfield(LV, Symbol(replace(string(name), r"core$" => "model")))
                m = ExaModels.ExaModel(getfield(LV, name)(), 100)
                md = model_f(LV.ExaModelsBackend(), 100)
                check_parity(m, md)
            end
        end
    end
end

# Representatives spanning the deferred-value vocabulary; the full 30-instance
# sweep runs in COPSBenchmark's own testset.
const COPS_CASES = [
    (:camshape, (n = 50,)),          # scalar constraints, arg-dependent bounds, minimize = false
    (:catmix,   (nh = 10,)),         # deferred product iterators
    (:channel,  (nh = 20,)),         # value-loop collocation data via Deferred pars
    (:elec,     (np = 25,)),         # seeded random starts
    (:gasoil,   (nh = 10,)),         # collocation family, nested deferred comprehensions
    (:torsion,  (nx = 10, ny = 10)), # 2-D grid, deferred indexing
]

function cops_tests()
    @testset "COPSBenchmark cores" begin
        for (nm, s) in COPS_CASES
            @testset "$nm" begin
                m = ExaModels.ExaModel(getfield(COPSBenchmark, Symbol(nm, :_core))(), s)
                md = getfield(COPSBenchmark, Symbol(nm, :_model))(
                    COPSBenchmark.ExaModelsBackend(), values(s)...)
                check_parity(m, md)
            end
        end
    end
end

# ── Structured args: one core over record vectors, any grid ──────────────────
function opf_tests()
    @testset "AC-OPF structured args" begin
        data = parse_ac_power_data(get_power_case("pglib_opf_case3_lmbd.m"))
        m = ExaModels.ExaModel(opf_lazy(ExaModels.ArgTracer()), data)
        md, _, _ = __exa_ac_power_model(nothing, data)
        check_parity(m, md)
        r = MadNLP.madnlp(m; print_level = MadNLP.ERROR)
        @test r.status == MadNLP.SOLVE_SUCCEEDED
    end
end

function runtests()
    @testset "Lazy core" begin
        api_tests()
        lv_tests()
        cops_tests()
        opf_tests()
    end
end

end # module
