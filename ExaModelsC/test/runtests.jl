module ExaModelsCTest

using Test
using ExaModels, ExaModelsC
import CNLPModels
import NLPModels
import NLPModelsIpopt

# A model with a real objective and a real constraint, sized by `arg.N`:
#
#   min  Σ (x[i] - 2)^2      s.t.   x[i] + x[i+1] >= 3,   i = 1 … N-1
#
# written exactly as it would be without `arg`, except that the size is left
# open.  `build` is called twice below — once to compile, once to produce the
# in-Julia reference the library is checked against.
function build(cn = ExaCore(concrete = Val(true), nargs = Val(1)))
    c, N = cn
    @add_var(c, x, N; start = 1.0)
    @add_obj(c, (x[i] - 2.0)^2 for i in 1:N)
    @add_con(c, x[i] + x[i + 1] for i in 1:(N - 1); lcon = 3.0, ucon = Inf)
    return c
end

const OUT = get(ENV, "EXAMODELSC_TEST_OUT", joinpath(tempdir(), "examodelsc_test"))

function runtests()
    @testset "ExaModelsC" begin

        @testset "the example arg is read, not guessed" begin
            c = build()
            # A shape `P_new(n)` cannot carry must be refused up front, with a
            # reason — not discovered as a missing symbol at load time.
            @test_throws ArgumentError compile_library(OUT, c, (N = 4, v = [1.0]))
            @test_throws ArgumentError compile_library(OUT, c, (x = 1.5,))
            # And a prefix that is not a C identifier is caught before juliac.
            @test_throws ArgumentError compile_library(OUT, c, 4; prefix = "lib-a")
            @test_throws ArgumentError compile_library(OUT, c, 4; prefix = "2fast")
        end

        # Compiling takes minutes, so the library is built once and every
        # behavioural test below reads that one artifact.
        # Compiling takes minutes, so the library is built once and every
        # behavioural test below reads that one artifact.
        r = compile_library(joinpath(OUT, "rosen"), build(), 4)

        @testset "the library exists and loads" begin
            @test isfile(r.libpath)
            @test r.prefix == "rosen"
        end

        lib = CNLPModels.load(r.libpath)

        @testset "instantiated at a size the compile never saw" begin
            # Compiled with the example N = 4; instantiated at 25.  If the size
            # had been baked in rather than deferred, this is where it shows.
            N = 25
            m = CNLPModels.CNLPModel(lib; prefix = r.prefix, args = N)
            ref = ExaModel(build(), N)

            @test m.meta.nvar == ref.meta.nvar == N
            @test m.meta.ncon == ref.meta.ncon == N - 1
            @test m.meta.nnzj == ref.meta.nnzj
            @test m.meta.nnzh == ref.meta.nnzh
            @test m.meta.x0 ≈ ref.meta.x0
            @test m.meta.lcon ≈ ref.meta.lcon
            @test m.meta.ucon ≈ ref.meta.ucon

            x = collect(range(0.5, 3.0; length = N))
            y = collect(range(-1.0, 1.0; length = N - 1))

            @test NLPModels.obj(m, x) ≈ NLPModels.obj(ref, x)
            @test NLPModels.grad(m, x) ≈ NLPModels.grad(ref, x)
            @test NLPModels.cons(m, x) ≈ NLPModels.cons(ref, x)

            jr, jc = NLPModels.jac_structure(m)
            rr, rc = NLPModels.jac_structure(ref)
            @test jr == rr && jc == rc
            @test NLPModels.jac_coord(m, x) ≈ NLPModels.jac_coord(ref, x)

            hr, hc = NLPModels.hess_structure(m)
            sr, sc = NLPModels.hess_structure(ref)
            @test hr == sr && hc == sc
            @test NLPModels.hess_coord(m, x, y; obj_weight = 0.5) ≈
                  NLPModels.hess_coord(ref, x, y; obj_weight = 0.5)
        end

        @testset "instances are independent" begin
            m1 = CNLPModels.CNLPModel(lib; prefix = r.prefix, args = 6)
            m2 = CNLPModels.CNLPModel(lib; prefix = r.prefix, args = 11)
            @test m1.meta.nvar == 6
            @test m2.meta.nvar == 11
            # The second instantiation must not have disturbed the first.
            @test NLPModels.obj(m1, fill(1.0, 6)) ≈ 6.0
            @test NLPModels.obj(m2, fill(1.0, 11)) ≈ 11.0
        end

        @testset "bundle = false is a single file, for Python and C callers" begin
            # One ~2 MB library rather than an 80 MB directory, linked against
            # the installed Julia.  This is the form to hand a Python or C
            # caller; it cannot be loaded from Julia — see `compile_library`'s
            # docstring for why — so it is checked structurally here and the
            # Python leg below exercises the behaviour.
            u = compile_library(joinpath(OUT, "flat"), build(), 4; bundle = false)
            @test isfile(u.libpath)
            @test u.libpath == joinpath(
                u.outdir, "lib" * u.prefix * "." * Base.BinaryPlatforms.platform_dlext())
            @test !isdir(joinpath(u.outdir, "lib", "julia"))     # not a bundle
            @test filesize(u.libpath) < 20_000_000
            @test filesize(u.libpath) < filesize(r.libpath) * 10 # far smaller than bundled
        end

        @testset "the Python consumer reads the same model" begin
            # cnlpmodels is an unrelated package (https://github.com/MadNLP/cnlpmodels-py),
            # so this leg is skipped rather than failed when it is not present.
            # It is pure Python over ctypes+numpy, hence usable straight from a
            # source checkout — point CNLPMODELS_PY at one.
            pysrc = get(
                ENV, "CNLPMODELS_PY",
                joinpath(homedir(), "git", "pkg", "cnlpmodels-py", "src"),
            )
            script = joinpath(@__DIR__, "cnlpmodels_check.py")
            # `python3` on unix, `python` on Windows — probe for one that both
            # exists and has numpy, rather than assuming either.
            py = nothing
            for cand in ("python3", "python")
                ok = try
                    success(pipeline(`$cand -c "import numpy"`; stdout = devnull, stderr = devnull))
                catch
                    false
                end
                ok && (py = cand; break)
            end

            if py === nothing || !isdir(pysrc)
                @info "skipping the Python leg" python = py pysrc_exists = isdir(pysrc)
                @test_skip false
            else
                N = 12
                outfile = joinpath(mktempdir(), "py.txt")
                env = copy(ENV)
                sep = Sys.iswindows() ? ";" : ":"     # PATH separator, not ':' everywhere
                env["PYTHONPATH"] =
                    pysrc * (haskey(env, "PYTHONPATH") ? sep * env["PYTHONPATH"] : "")
                run(setenv(`$py $script $(r.libpath) $(r.prefix) $N $outfile`, env))

                vals = Dict{String,Vector{Float64}}()
                for line in eachline(outfile)
                    parts = split(line)
                    vals[parts[1]] = parse.(Float64, parts[2:end])
                end

                ref = ExaModel(build(), N)
                x = collect(range(0.5, 3.0; length = N))
                y = collect(range(-1.0, 1.0; length = N - 1))

                @test only(vals["nvar"]) == ref.meta.nvar
                @test only(vals["ncon"]) == ref.meta.ncon
                @test only(vals["nnzj"]) == ref.meta.nnzj
                @test only(vals["nnzh"]) == ref.meta.nnzh
                @test vals["x0"] ≈ ref.meta.x0
                @test vals["lvar"] ≈ ref.meta.lvar
                @test vals["uvar"] ≈ ref.meta.uvar
                @test vals["lcon"] ≈ ref.meta.lcon
                @test vals["ucon"] ≈ ref.meta.ucon
                @test only(vals["obj"]) ≈ NLPModels.obj(ref, x)
                @test vals["grad"] ≈ NLPModels.grad(ref, x)
                @test vals["cons"] ≈ NLPModels.cons(ref, x)

                jr, jc = NLPModels.jac_structure(ref)
                @test Int.(vals["jac_rows"]) == jr
                @test Int.(vals["jac_cols"]) == jc
                @test vals["jac"] ≈ NLPModels.jac_coord(ref, x)

                hr, hc = NLPModels.hess_structure(ref)
                @test Int.(vals["hess_rows"]) == hr
                @test Int.(vals["hess_cols"]) == hc
                @test vals["hess"] ≈ NLPModels.hess_coord(ref, x, y; obj_weight = 0.5)
            end
        end

        @testset "solving through the library agrees with solving in Julia" begin
            N = 20
            m = CNLPModels.CNLPModel(lib; prefix = r.prefix, args = N)
            ref = ExaModel(build(), N)

            res = NLPModelsIpopt.ipopt(m; print_level = 0)
            ref_res = NLPModelsIpopt.ipopt(ref; print_level = 0)

            @test res.status == ref_res.status
            @test res.objective ≈ ref_res.objective atol = 1e-6
            @test res.solution ≈ ref_res.solution atol = 1e-5
            # x[i] = 2 satisfies x[i] + x[i+1] = 4 >= 3, so the constraint is
            # inactive and the unconstrained optimum stands.
            @test res.objective ≈ 0.0 atol = 1e-6
            @test res.solution ≈ fill(2.0, N) atol = 1e-5
        end
    end
end

end # module

ExaModelsCTest.runtests()
