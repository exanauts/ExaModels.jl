# THE AOT test of the recorder: each model's tape is compiled into an
# evaluation-only shared library (ExaModelsJuliaC's `compile_library` — no
# solver trimmed in), consumed through the C ABI by CNLPModels.jl, and solved
# by a HOST solver (MadNLP here; any NLPModels solver works, since the host is
# not trimmed). Each case runs in its own subprocess: several privatized
# runtimes cannot coexist in one long-lived process. Covers both
# `compile_library` input forms — a model file (`build`/`make_data` contract)
# and a tree-built tape (the non-Julia-frontend path).

const _SKIP_AOT = get(ENV, "EXAMODELS_SKIP_AOT", "") != ""
const _HAS_JULIAC_API = isdefined(JuliaC, :ImageRecipe)

# Script tail shared by every case. Expects `r` (compile_library result) and
# `m_ref` (in-process reference model at n = 40) to be in scope. The whole
# callback surface is exercised through the ABI and compared against the
# reference at 1e-14: the library's model is the same tape instantiated, so
# structure and values must agree to rounding.
function _lib_consume(; solve::Bool)
    s = """
        m = CNLPModel(CNLPModels.load(r.libpath); prefix = "m", args = 40)
        x0 = copy(m.meta.x0)
        ok_meta = m.meta.nvar == m_ref.meta.nvar && m.meta.ncon == m_ref.meta.ncon &&
                  x0 == m_ref.meta.x0
        println("LIB_META : ", ok_meta ? 0 : 1)
        y = ones(m.meta.ncon)
        ok_eval = isapprox(NLPModels.obj(m, x0), NLPModels.obj(m_ref, x0); rtol = 1e-14) &&
                  isapprox(NLPModels.grad(m, x0), NLPModels.grad(m_ref, x0); rtol = 1e-14) &&
                  (m.meta.ncon == 0 ||
                   isapprox(NLPModels.cons(m, x0), NLPModels.cons(m_ref, x0); rtol = 1e-14) &&
                   isapprox(NLPModels.jac_coord(m, x0), NLPModels.jac_coord(m_ref, x0); rtol = 1e-14)) &&
                  isapprox(NLPModels.hess_coord(m, x0, y), NLPModels.hess_coord(m_ref, x0, y); rtol = 1e-14)
        println("LIB_EVAL : ", ok_eval ? 0 : 1)
        o_before = NLPModels.obj(m, x0)
        m2 = CNLPModel(CNLPModels.load(r.libpath); prefix = "m", args = 20)
        ok_iso = NLPModels.obj(m, x0) == o_before && m2.meta.nvar != m.meta.nvar
        println("LIB_ISO : ", ok_iso ? 0 : 1)
        """
    if solve
        s *= """
            res = MadNLP.madnlp(m; print_level = MadNLP.ERROR)
            res_ref = MadNLP.madnlp(m_ref; print_level = MadNLP.ERROR)
            ok_solve = res.status == MadNLP.SOLVE_SUCCEEDED &&
                       res_ref.status == MadNLP.SOLVE_SUCCEEDED &&
                       isapprox(res.objective, res_ref.objective; rtol = 1e-8)
            println("LIB_SOLVE : ", ok_solve ? 0 : 1)
            """
    end
    return s
end

function _file_case(mname::String)
    path = repr(joinpath(@__DIR__, "models", mname * ".jl"))
    script = """
        using ExaModels, ExaModelC, MadNLP, CNLPModels, NLPModels
        r = ExaModelC.compile_library(
            $path; prefix = "m", out = mktempdir(), template_n = 8)
        mod = Module(:MRef)
        Core.eval(mod, :(using ExaModels))
        Base.include(mod, $path)
        # The file's methods (and the traced closures inside the reference
        # model) are newer than this frame's world age.
        m_ref = Base.invokelatest() do
            core = mod.build(ExaModels.ExaCore(concrete = Val(true)),
                             ExaModels.ArgTracer())
            ExaModels.ExaModel(core, mod.make_data(40))
        end
        """ * _lib_consume(solve = true)
    markers = ("LIB_META", "LIB_EVAL", "LIB_ISO", "LIB_SOLVE")
    return (name = mname, script = script, markers = markers)
end

# Core-input form: the model is synthetic (no solver reference values), so the
# solve-parity leg is skipped; the evaluation surface is the property
# compile_library guarantees.
function _tree_case()
    script = """
        using ExaModels, ExaModelC, MadNLP, CNLPModels, NLPModels
        args = ExaModels.ArgTracer()
        c = ExaModels.ExaCore(concrete = Val(true))
        c, xv = ExaModels.add_var(c, args.N; start = -0.5)
        i = ExaModels.DataSource()
        c, _ = ExaModels.add_con(c, 3xv[i+1]^3 + 2xv[i+2] - 5, 1:args.N-2)
        c, _ = ExaModels.add_obj(c, (xv[i-1] - 1)^2, 2:args.N)
        r = ExaModelC.compile_library(c; template = (; N = 4), prefix = "m", out = mktempdir())
        m_ref = ExaModels.ExaModel(c, (; N = 40))
        """ * _lib_consume(solve = false)
    return (name = "core input", script = script, markers = ("LIB_META", "LIB_EVAL", "LIB_ISO"))
end

function aot_library_tests()
    @testset "AOT model libraries (ExaModelC)" begin
        if _SKIP_AOT
            @info "Skipping AOT library tests (EXAMODELS_SKIP_AOT is set)"
        elseif !_HAS_JULIAC_API
            @warn "JuliaC.ImageRecipe not available, skipping AOT library tests"
        else
            cases = [_file_case("rosenrock"), _file_case("wood"),
                     _file_case("two_blocks"), _tree_case()]
            @testset "$(case.name)" for case in cases
                out = IOBuffer()
                result = run(pipeline(
                    ignorestatus(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $(case.script)`);
                    stdout = out, stderr = out,
                ))
                txt = String(take!(out))
                @test success(result)
                for mk in case.markers
                    @test contains(txt, "$mk : 0")
                end
            end
        end
    end
end
