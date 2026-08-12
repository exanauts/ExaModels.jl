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
# in-Julia reference the library is checked against.  The default core is the
# default (non-concrete) mode on purpose: compiling from it is what proves
# `_concretize` hands juliac the same artifact a `Val(true)` core produces
# (the `Val(true)` compile path is covered by the main suite's app tests).
function build(cn = ExaCore(nargs = Val(1)))
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

        @testset "out is @name on the search path, or a literal path" begin
            # The same convention the consumers apply to their string spec:
            # any string without the sigil is a local path, exactly as written.
            @test ExaModelsC._resolve_out("rosen", true) == abspath("rosen")
            @test ExaModelsC._resolve_out("./rosen", true) == abspath("./rosen")
            withenv("CNLPMODELS_PATH" => nothing) do
                @test_throws ArgumentError ExaModelsC._resolve_out("@rosen", true)
            end
            withenv("CNLPMODELS_PATH" => "/tmp/models") do
                @test ExaModelsC._resolve_out("@rosen", true) == joinpath("/tmp/models", "rosen")
                @test ExaModelsC._resolve_out("@rosen", false) == "/tmp/models"
            end
            # The default prefix keeps no sigil.
            @test ExaModelsC._default_out_prefix("@rosen") == "rosen"
            @test_throws ArgumentError ExaModelsC._resolve_out("@", true)
        end

        # Compiling takes minutes, so the library is built once and every
        # behavioural test below reads that one artifact.  Bundled explicitly:
        # this is the bundle-path coverage (the default — unbundled — has its
        # own testset below), and the bundle is the form CNLPModels.jl can
        # load on every OS.
        r = compile_library(joinpath(OUT, "rosen"), build(), 4; bundle = true)

        @testset "the library exists and loads" begin
            @test isfile(r.libpath)
            @test r.prefix == "rosen"
        end

        lib = CNLPModels.load(r.libpath)

        @testset "instantiated at a size the compile never saw" begin
            # Compiled with the example N = 4; instantiated at 25.  If the size
            # had been baked in rather than deferred, this is where it shows.
            N = 25
            m = CNLPModels.CNLPModel(lib, N; prefix = r.prefix)
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
            m1 = CNLPModels.CNLPModel(lib, 6; prefix = r.prefix)
            m2 = CNLPModels.CNLPModel(lib, 11; prefix = r.prefix)
            @test m1.meta.nvar == 6
            @test m2.meta.nvar == 11
            # The second instantiation must not have disturbed the first.
            @test NLPModels.obj(m1, fill(1.0, 6)) ≈ 6.0
            @test NLPModels.obj(m2, fill(1.0, 11)) ≈ 11.0
        end

        @testset "the recipe library declares its arity" begin
            @test ccall(
                CNLPModels.Libdl.dlsym(lib.handle, Symbol(r.prefix, "_nargs")),
                Cint, ()) == 1
        end

        @testset "a fixed core — no placeholders — compiles with no args" begin
            # A declared-arity recipe without examples is refused up front,
            # naming what is missing. Pinned to the guard's own message: the
            # instantiation probe would also throw an ArgumentError, so a bare
            # type match could not tell the guard from its backstop.
            @test_throws "declared 1 placeholder" compile_library(OUT, build())

            # A complete model, built in the default (non-concrete) mode and
            # compiled without example values.
            n = 7
            c = ExaCore()
            @add_var(c, x, n; start = 1.0)
            @add_obj(c, (x[i] - 2.0)^2 for i in 1:n)
            @add_con(c, x[i] + x[i + 1] for i in 1:(n - 1); lcon = 3.0, ucon = Inf)
            ref = ExaModel(c)

            # Compiled through the `@name` spelling: installs on the search
            # path, with the prefix defaulting to the name.  Bundled, so the
            # in-Julia consumption below works on every OS — and a second
            # bundle in one process is the salt-collision regression case.
            f = withenv("CNLPMODELS_PATH" => OUT) do
                compile_library("@fixed", c; bundle = true)
            end
            @test f.outdir == joinpath(OUT, "fixed")
            @test f.prefix == "fixed"
            @test isfile(f.libpath)
            flib = CNLPModels.load(f.libpath)

            # The library says it consumes no instantiation arguments.
            @test ccall(
                CNLPModels.Libdl.dlsym(flib.handle, Symbol(f.prefix, "_nargs")),
                Cint, ()) == 0

            # The whole chain with no instance data anywhere: CNLPModels
            # consults `_nargs` and instantiates from nothing but the handle.
            m3 = CNLPModels.CNLPModel(flib; prefix = f.prefix)
            @test m3.meta.nvar == n

            # `_new` keeps its one-integer C signature and ignores the value:
            # any integer instantiates the same fixed model.
            m = CNLPModels.CNLPModel(flib, 0; prefix = f.prefix)
            @test m.meta.nvar == ref.meta.nvar == n
            @test m.meta.ncon == ref.meta.ncon == n - 1
            @test m.meta.x0 ≈ ref.meta.x0

            x = collect(range(0.5, 3.0; length = n))
            @test NLPModels.obj(m, x) ≈ NLPModels.obj(ref, x)
            @test NLPModels.grad(m, x) ≈ NLPModels.grad(ref, x)
            @test NLPModels.cons(m, x) ≈ NLPModels.cons(ref, x)

            m2 = CNLPModels.CNLPModel(flib, 999; prefix = f.prefix)
            @test m2.meta.nvar == n
        end

        @testset "the default — unbundled — is a single file, loadable everywhere" begin
            # No `bundle` argument: this is what `compile_library` emits by
            # default — one ~2 MB library rather than an 80 MB directory,
            # linked against the installed Julia.
            u = compile_library(joinpath(OUT, "flat"), build(), 4)
            @test isfile(u.libpath)
            @test u.libpath == joinpath(
                u.outdir, "lib" * u.prefix * "." * Base.BinaryPlatforms.platform_dlext())
            @test !isdir(joinpath(u.outdir, "lib", "julia"))     # not a bundle
            @test filesize(u.libpath) < 20_000_000
            @test filesize(u.libpath) < filesize(r.libpath) * 10 # far smaller than bundled

            if Sys.islinux()
                # In-process consumption: CNLPModels detects the standard
                # libjulia NEEDED and provisions a load-time private runtime
                # (loading this library as-is would abort the process, so
                # this block passing IS the mechanism working). The bundled
                # `rosen` runtime above is already resident: the two must
                # coexist.
                ulib = CNLPModels.load(u.libpath)
                N = 17
                um = CNLPModels.CNLPModel(ulib, N; prefix = u.prefix)
                ref = ExaModel(build(), N)
                x = collect(range(0.5, 3.0; length = N))
                @test um.meta.nvar == ref.meta.nvar == N
                @test NLPModels.obj(um, x) ≈ NLPModels.obj(ref, x)
                @test NLPModels.grad(um, x) ≈ NLPModels.grad(ref, x)
                @test NLPModels.cons(um, x) ≈ NLPModels.cons(ref, x)
                res = NLPModelsIpopt.ipopt(um; print_level = 0)
                @test res.solution ≈ fill(2.0, N) atol = 1e-5
                # And the bundled library is still alive next to it.
                @test NLPModels.obj(
                    CNLPModels.CNLPModel(lib, 5; prefix = r.prefix),
                    fill(1.0, 5)) ≈ 5.0
            else
                # The Python leg below is the consumer for this form here.
                @info "in-process unbundled loading is Linux-only; skipped"
            end
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

            # Both forms go through the same script: the bundled library and
            # the unbundled single file, which Python loads as-is.
            flatlib = joinpath(
                OUT, "flat", "libflat." * Base.BinaryPlatforms.platform_dlext())
            pylibs = [(r.libpath, r.prefix)]
            isfile(flatlib) && push!(pylibs, (flatlib, "flat"))
            if py === nothing || !isdir(pysrc)
                @info "skipping the Python leg" python = py pysrc_exists = isdir(pysrc)
                @test_skip false
            else
                for (pylib, pyprefix) in pylibs
                N = 12
                outfile = joinpath(mktempdir(), "py.txt")
                env = copy(ENV)
                sep = Sys.iswindows() ? ";" : ":"     # PATH separator, not ':' everywhere
                env["PYTHONPATH"] =
                    pysrc * (haskey(env, "PYTHONPATH") ? sep * env["PYTHONPATH"] : "")
                run(setenv(`$py $script $(pylib) $(pyprefix) $N $outfile`, env))

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
        end

        @testset "solving through the library agrees with solving in Julia" begin
            N = 20
            m = CNLPModels.CNLPModel(lib, N; prefix = r.prefix)
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
