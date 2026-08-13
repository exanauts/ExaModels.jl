module ExaModelsCTest

using Test
using ExaModels, ExaModelsC
import Pkg
import RecipeKernels
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

# A three-placeholder recipe — a bare size, a NamedTuple carrying a start and
# a bound, and a table — the data-defined shape the builder ABI exists for.
function sbuild()
    c, sz, dat, tab = ExaCore(nargs = Val(3))
    @add_var(c, x, sz; start = dat.v0, lvar = dat.lo)
    @add_obj(c, t.w * (x[t.i] - t.s)^2 for t in tab)
    @add_con(c, x[i] + x[i+1] for i in 1:(sz - 1); lcon = -100.0, ucon = 100.0)
    return c
end

# The example values `sbuild` is compiled against, and the instantiation data
# the compile never sees (different size, different rows).
const S_EXTAB = [(i = 1, w = 2.0, s = 1.0), (i = 3, w = 1.0, s = 0.5)]
const S_EXARGS = (4, (v0 = fill(0.5, 4), lo = fill(-10.0, 4)), S_EXTAB)
const S_N = 6
const S_V0 = collect(range(0.1, 0.6; length = S_N))
const S_LO = fill(-5.0, S_N)
const S_TAB = [(i = 2, w = 1.5, s = 2.0), (i = 5, w = 3.0, s = -1.0), (i = 6, w = 0.5, s = 0.0)]

function runtests()
    @testset "ExaModelsC" begin

        @testset "the example arg is read, not guessed" begin
            c = build()
            # `build`'s core reads its one placeholder as a bare size, so a
            # NamedTuple example cannot instantiate it — refused up front by
            # the probe, with a reason, not discovered inside juliac.
            @test_throws ArgumentError compile_library(OUT, c, (N = 4, v = [1.0]))
            @test_throws ArgumentError compile_library(OUT, c, (x = 1.5,))
            # And a prefix that is not a C identifier is caught before juliac.
            @test_throws ArgumentError compile_library(OUT, c, 4; prefix = "lib-a")
            @test_throws ArgumentError compile_library(OUT, c, 4; prefix = "2fast")
        end

        @testset "builder examples are read, not guessed" begin
            c = build()
            # Types that cannot cross the boundary stay refused with the
            # builder there.
            @test_throws ArgumentError compile_library(OUT, c, "c")
            @test_throws ArgumentError compile_library(OUT, c, 4, "c", [1.0, 2.0])
            # Builder storage is the example's type EXACTLY — looser numeric
            # types are named here, not discovered inside the compiled library.
            @test_throws "Int64/Float64" compile_library(OUT, c, Int32(4), [1.0])
            @test_throws "Int64/Float64" compile_library(OUT, c, 4, Float32[1.0])
            @test_throws "Int64/Float64" compile_library(OUT, c, 4, 1:3)
            # Flattened field names must be distinct across all placeholders.
            @test_throws "both be named" compile_library(OUT, c, (n = 1,), (n = 2.0,))
            # An empty example, and rows with no columns, carry no types.
            @test_throws ArgumentError compile_library(OUT, c, [1.0], Float64[])
            @test_throws "no columns" compile_library(OUT, c, [1.0], [(;), (;)])

            # A one-key integer NamedTuple keeps the `P_new(n)` fast path —
            # the generated constructor rebuilds the key.
            cnt, _ = ExaCore(nargs = Val(1))
            s = ExaModelsC._model_spec("nt", cnt, ((N = 4,),))
            @test s.field === :N
            ntsrc = ExaModelsC._module_source("M", [s])
            @test occursin("(; N = Int(n))", ntsrc)
            @test occursin("nt_new(", ntsrc)
            @test !occursin("function nt_data_begin", ntsrc)   # the comment may NAME it

            # Anything else flattens to the builder: bare values by position,
            # NamedTuple entries by key, and NO `P_new` — the surfaces are
            # disjoint, which is how a consumer routes a lone integer.
            b = ExaModelsC._model_spec(
                "bs", cnt, (4, (v0 = [1.0], lo = [2.0]), [(i = 1, w = 0.5)]),
            )
            @test b.field isa ExaModelsC.BuilderModel
            @test [f.name for f in b.field.fields] == ["arg1", "v0", "lo", "arg3"]
            bsrc = ExaModelsC._module_source("M", [b])
            @test !occursin("function bs_new(", bsrc)   # `_argkind`'s comment names it
            for sym in (
                "bs_schema", "bs_data_begin", "bs_set_scalar_i64",
                "bs_set_scalar_f64", "bs_set_array_i64", "bs_set_array_f64",
                "bs_set_col_i64", "bs_set_col_f64", "bs_data_ready",
                "bs_new_from_data", "bs_nargs",
            )
                @test occursin(sym, bsrc)
            end
        end

        @testset "a recipe's own package travels into the generated app" begin
            # A modelling library's own function inside the core is the ordinary
            # case, not an exotic one: a per-index start and a size-dependent
            # index set both have to be deferred, and what gets deferred is that
            # library's code.  The core then names a `RecipeKernels` type.
            c, N = ExaCore(nargs = Val(1))
            @add_var(c, x, N; start = Base.Generator(RecipeKernels.alternating, 1:N))
            @add_obj(c, (x[i] - 2.0)^2 for i in 1:N)
            @add_con(c, x[l+1] + x[l+2] for l in ExaModels.ArgNode1(RecipeKernels.offsets, N))

            pkgs = ExaModelsC._core_packages(ExaModelsC._concretize(c))
            @test length(pkgs) == 1
            @test only(pkgs).name == "RecipeKernels"
            @test isdir(only(pkgs).dir)

            # A core with nothing but ExaModels in it must not drag anything in:
            # the generated project has to stay exactly as it was for every model
            # that does not need this.
            plain, M = ExaCore(nargs = Val(1))
            @add_var(plain, z, M; start = 1.0)
            @add_obj(plain, (z[i] - 2.0)^2 for i in 1:M)
            @test isempty(ExaModelsC._core_packages(ExaModelsC._concretize(plain)))

            # A package the CALLER is developing — RecipeKernels is path-
            # tracked in this test environment — is pinned in the generated
            # project as a dependency + path source even when no core
            # references it, so the app compiles the code the caller is
            # actually running rather than the registry copy. It is NOT
            # imported: only a core's own packages need that.
            @test any(p -> p.name == "RecipeKernels", ExaModelsC._developed_packages())
            pdir = ExaModelsC._generate_app(
                [ExaModelsC._model_spec("pl", ExaModelsC._concretize(plain), (4,))],
                "pl",
            )
            pproj = read(joinpath(pdir, "Project.toml"), String)
            psrc = read(joinpath(pdir, "src", "ExaLib_pl.jl"), String)
            @test occursin("RecipeKernels = {path =", pproj)
            @test !occursin("import RecipeKernels", psrc)

            # Both halves are needed and neither alone is enough: a dependency
            # the app never imports resolves no better than one it never had, so
            # the generated files are checked for the dependency AND the import.
            appdir = ExaModelsC._generate_app(
                [ExaModelsC._model_spec("rk", ExaModelsC._concretize(c), (8,))], "rk",
            )
            proj = read(joinpath(appdir, "Project.toml"), String)
            src = read(joinpath(appdir, "src", "ExaLib_rk.jl"), String)
            @test occursin("RecipeKernels = \"5c1e4a77", proj)
            @test occursin("RecipeKernels = {path =", proj)
            @test occursin("import RecipeKernels", src)

            # And the property those two exist for: a *different* process, whose
            # only knowledge of RecipeKernels is what the generated project says,
            # can read the core back and build a model from it.  This is the step
            # that failed before, with `KeyError: PkgId(... "RecipeKernels")`.
            probe = joinpath(appdir, "probe.jl")
            answer = joinpath(appdir, "answer.txt")
            write(probe, """
                import Pkg
                Pkg.instantiate(; io = devnull)
                import ExaModels, Serialization, RecipeKernels
                core = Serialization.deserialize(joinpath(@__DIR__, "src", "core_rk.jls"))
                m = ExaModels.ExaModel(core, 12; check = Val(false))
                write(
                    joinpath(@__DIR__, "answer.txt"),
                    string(m.meta.nvar, ",", m.meta.ncon, ",", m.meta.x0[1], ",", m.meta.x0[2]),
                )
                """)
            # Keep the child's output: without it a failure here reads as a bare
            # `false`, and the whole point of this assertion is the reason.
            log = IOBuffer()
            ran = success(pipeline(
                `$(Base.julia_cmd()) --startup-file=no --project=$appdir $probe`;
                stdout = log, stderr = log,
            ))
            ran || @error "generated app failed to load its core" output =
                String(take!(log))
            @test ran
            @test isfile(answer) && read(answer, String) == "12,5,-1.2,1.0"

            # An extension is the natural home for a recipe's ExaModels-facing
            # parts, and a type it owns cannot be depended on by name — there is
            # no `RecipeKernelsExaModels` to add to a project.  It has to resolve
            # to the package that carries it, which the app then imports;
            # ExaModels is already imported, so Julia loads the extension itself.
            ext = Base.get_extension(RecipeKernels, :RecipeKernelsExaModels)
            @test ext !== nothing
            e, K = ExaCore(nargs = Val(1))
            @add_var(e, w, K; start = Base.Generator(ext.ramp, 1:K))
            @add_obj(e, (w[i] - 1.0)^2 for i in 1:K)
            epkgs = ExaModelsC._core_packages(ExaModelsC._concretize(e))
            @test length(epkgs) == 1
            @test only(epkgs).name == "RecipeKernels"       # the parent, not the ext
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

        @testset "several models in one library" begin
            # One library carrying three models: the same recipe under two
            # names — so the two are known to be genuinely separate, not one
            # model aliased twice — and a fixed model taking no instantiation
            # data at all.  Compiling takes minutes, so this is the suite's
            # only multi-model build.
            n = 5
            fc = ExaCore()
            @add_var(fc, z, n; start = 1.0)
            @add_obj(fc, (z[i] - 4.0)^2 for i in 1:n)
            fref = ExaModel(fc)

            g = compile_library(
                joinpath(OUT, "grid"),
                :sized => (build(), 4),
                :other => (build(), 6),
                :flat => fc,
            )

            @test g.prefixes == ["sized", "other", "flat"]
            @test isfile(g.libpath)
            # The FILE is named after `out`, not after any one model. This is
            # what separates one multi-model library from three single-model
            # ones, and it is what a consumer resolves `@grid` against.
            @test basename(g.libpath) ==
                  "libgrid." * Base.BinaryPlatforms.platform_dlext()

            glib = CNLPModels.load(g.libpath)

            # Each model declares its own arity, out of the one library.
            @test ccall(
                CNLPModels.Libdl.dlsym(glib.handle, :sized_nargs), Cint, ()) == 1
            @test ccall(
                CNLPModels.Libdl.dlsym(glib.handle, :other_nargs), Cint, ()) == 1
            @test ccall(
                CNLPModels.Libdl.dlsym(glib.handle, :flat_nargs), Cint, ()) == 0

            # Instances of three different models, from one library, at once —
            # and at sizes neither compile saw (the examples were 4 and 6).
            a = CNLPModels.CNLPModel(glib, 9; prefix = "sized")
            b = CNLPModels.CNLPModel(glib, 13; prefix = "other")
            c = CNLPModels.CNLPModel(glib; prefix = "flat")
            @test a.meta.nvar == 9
            @test b.meta.nvar == 13
            @test c.meta.nvar == n

            # Separate instance tables, not one shared counter: each model's
            # first instance is id 1.  Sharing a table would number these
            # 1, 2, 3, so this is what actually distinguishes the two designs.
            @test a.id == b.id == c.id == 1

            # And no model's instantiation disturbed another's.
            @test NLPModels.obj(a, fill(1.0, 9)) ≈ 9.0
            @test NLPModels.obj(b, fill(1.0, 13)) ≈ 13.0
            @test NLPModels.obj(c, fill(1.0, n)) ≈ NLPModels.obj(fref, fill(1.0, n))

            # Each still evaluates as its own core, against an in-Julia
            # reference — the derivatives too, not just the sizes.
            aref = ExaModel(build(), 9)
            x = collect(range(0.5, 3.0; length = 9))
            y = collect(range(-1.0, 1.0; length = 8))
            @test NLPModels.obj(a, x) ≈ NLPModels.obj(aref, x)
            @test NLPModels.grad(a, x) ≈ NLPModels.grad(aref, x)
            @test NLPModels.cons(a, x) ≈ NLPModels.cons(aref, x)
            @test NLPModels.jac_coord(a, x) ≈ NLPModels.jac_coord(aref, x)
            @test NLPModels.hess_coord(a, x, y; obj_weight = 0.5) ≈
                  NLPModels.hess_coord(aref, x, y; obj_weight = 0.5)

            # The consumer's own spelling for this selection is
            # `CNLPModel(glib, :sized, 9)` — the same prefix, named as a
            # symbol; it is exercised in CNLPModels' own suite.
        end

        @testset "a multi-model library is checked before it is compiled" begin
            # Every one of these is refused in this process, before any code is
            # generated and long before juliac runs.
            c = build()
            @test_throws ArgumentError compile_library(OUT)          # no models
            @test_throws "is not a model" compile_library(OUT, :a => 5)
            @test_throws "names no core" compile_library(OUT, :a => ())
            @test_throws "must begin with an `ExaCore`" compile_library(OUT, :a => (4, c))
            @test_throws "both named `a`" compile_library(OUT, :a => (c, 4), :a => (c, 5))
            @test_throws "must be a C identifier" compile_library(
                OUT, Symbol("2bad") => (c, 4))
            # The per-model checks are the single-model ones, and the message
            # names which model is at fault: two bare integers flatten to a
            # perfectly valid two-field schema, but this core declared ONE
            # placeholder, so the probe refuses before juliac spends minutes.
            @test_throws "`b`: the example values do not instantiate" compile_library(
                OUT, :a => (c, 4), :b => (c, 4, 5))
            @test_throws "`b`: this core declared 1 placeholder" compile_library(
                OUT, :a => (c, 4), :b => c)
            # `prefix =` has no meaning when the names supply the prefixes.
            @test_throws MethodError compile_library(OUT, :a => (c, 4); prefix = "z")
        end

        @testset "an argument function is checked before it is compiled" begin
            c = build()
            # Examples are the FUNCTION's arguments; none means nothing to pin.
            @test_throws "the examples are the arguments" compile_library(
                OUT, c; argfun = RecipeKernels.doubled_args)
            # One value crosses the boundary; give the function one.
            @test_throws "exactly one argument" compile_library(
                OUT, c, 4, 5; argfun = RecipeKernels.doubled_args)
            # A float is neither of the two shapes the boundary carries.
            @test_throws "one string or one 64-bit integer" compile_library(
                OUT, c, 1.5; argfun = RecipeKernels.doubled_args)
            # The function must return the argument tuple the core takes.
            @test_throws "must return the argument TUPLE" compile_library(
                OUT, c, 4; argfun = RecipeKernels.alternating)
            # Anonymous functions have no name for the library to call.
            @test_throws "named function" compile_library(OUT, c, 4; argfun = n -> (n,))
            # The pair spelling: a Function in second position is the argfun —
            # nothing callable can ever be a model argument.
            co, fn, rest = ExaModelsC._core_and_args(:m, (c, RecipeKernels.doubled_args, 4))
            @test fn === RecipeKernels.doubled_args && rest == (4,)
        end

        @testset "a structured model instantiates through the builder" begin
            # One compile carries every surface — builder, one-knob, and both
            # argument-function kinds — in one library: the surface is per
            # prefix, not per file.
            sb = compile_library(
                joinpath(OUT, "structs"),
                :structm => (sbuild(), S_EXARGS...),
                :knob => (build(), 4),
                :dbl => (build(), RecipeKernels.doubled_args, 4),
                :strd => (build(), RecipeKernels.parsed_args, "6"),
            )
            @test sb.prefixes == ["structm", "knob", "dbl", "strd"]
            slib = CNLPModels.load(sb.libpath)

            # The builder model exports no one-integer constructor; the knob
            # model exports no builder. Disjoint, as the consumers assume.
            dl(s) = CNLPModels.Libdl.dlsym(slib.handle, s; throw_error = false)
            @test dl(:structm_new) === nothing
            @test dl(:structm_data_begin) !== nothing
            @test dl(:knob_new) !== nothing
            @test dl(:knob_data_begin) === nothing
            @test ccall(dl(:structm_nargs), Cint, ()) == 4

            # The published schema is the flattened example: bare values by
            # position, NamedTuple entries by key, the table with its columns.
            sj = CNLPModels.schema_json(slib; prefix = "structm")
            for needle in (
                "\"arg1\"", "\"v0\"", "\"lo\"",
                """{"name":"arg3","kind":"table","columns":[{"name":"i","type":"i64"},{"name":"w","type":"f64"},{"name":"s","type":"f64"}]}""",
            )
                @test occursin(needle, sj)
            end

            # Instantiated at a size and data the compile never saw, through
            # the consumer's positional spelling — one value per schema field.
            m = CNLPModels.CNLPModel(slib, S_N, S_V0, S_LO, S_TAB; prefix = "structm")
            ref = ExaModel(sbuild(), S_N, (v0 = S_V0, lo = S_LO), S_TAB)

            @test m.meta.nvar == ref.meta.nvar == S_N
            @test m.meta.ncon == ref.meta.ncon == S_N - 1
            @test m.meta.x0 ≈ ref.meta.x0
            @test m.meta.lvar ≈ ref.meta.lvar

            x = collect(range(0.5, 3.0; length = S_N))
            y = collect(range(-1.0, 1.0; length = S_N - 1))
            @test NLPModels.obj(m, x) ≈ NLPModels.obj(ref, x)
            @test NLPModels.grad(m, x) ≈ NLPModels.grad(ref, x)
            @test NLPModels.cons(m, x) ≈ NLPModels.cons(ref, x)
            @test NLPModels.jac_coord(m, x) ≈ NLPModels.jac_coord(ref, x)
            @test NLPModels.hess_coord(m, x, y; obj_weight = 0.5) ≈
                  NLPModels.hess_coord(ref, x, y; obj_weight = 0.5)

            # A second builder instance and the sibling knob model, with the
            # first instance undisturbed.
            m2 = CNLPModels.CNLPModel(
                slib, 4, fill(0.5, 4), fill(-10.0, 4), S_EXTAB; prefix = "structm",
            )
            @test m2.meta.nvar == 4
            @test NLPModels.obj(m, x) ≈ NLPModels.obj(ref, x)
            k = CNLPModels.CNLPModel(slib, 7; prefix = "knob")
            @test k.meta.nvar == 7

            # The third surface: argument functions of both kinds, in the same
            # library. `_argkind` declares every model's shape — 0 fixed,
            # 1 `_new(n)`, 2 `_new_str`, 3 builder — so a consumer routes on
            # it rather than probing symbols.
            ak(s) = ccall(dl(Symbol(s, :_argkind)), Cint, ())
            @test ak("knob") == 1 && ak("structm") == 3
            @test ak("dbl") == 1 && ak("strd") == 2
            # :int kind — `P_new(n)` hands n to the function (size 2n here).
            dm = CNLPModels.CNLPModel(slib, 5; prefix = "dbl")
            @test dm.meta.nvar == 10
            # :str kind — `P_new_str` instantiates; `P_new` is not its entry
            # point and returns the documented failure value.
            sid = ccall(dl(:strd_new_str), Cint, (Cstring,), "6")
            @test sid > 0
            @test ccall(dl(:strd_nvar), Cint, (Cint,), sid) == 6
            @test ccall(dl(:strd_new), Cint, (Cint,), 4) == 0

            # Wrong-arity and incomplete data are the consumer's errors, not
            # aborts: the library reports, the consumer explains.
            @test_throws ErrorException CNLPModels.CNLPModel(
                slib, S_N, S_V0; prefix = "structm")

            # And a solve through the builder-instantiated model.
            res = NLPModelsIpopt.ipopt(m; print_level = 0)
            refres = NLPModelsIpopt.ipopt(ref; print_level = 0)
            @test res.status == refres.status
            @test res.objective ≈ refres.objective atol = 1e-6
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

        @testset "the Python consumer drives the builder" begin
            # Same probing as the leg above; additionally needs the structured
            # library the builder testset compiled.
            pysrc = get(
                ENV, "CNLPMODELS_PY",
                joinpath(homedir(), "git", "pkg", "cnlpmodels-py", "src"),
            )
            py = nothing
            for cand in ("python3", "python")
                ok = try
                    success(pipeline(`$cand -c "import numpy"`; stdout = devnull, stderr = devnull))
                catch
                    false
                end
                ok && (py = cand; break)
            end
            slibpath = joinpath(
                OUT, "structs", "libstructs." * Base.BinaryPlatforms.platform_dlext(),
            )
            if py === nothing || !isdir(pysrc) || !isfile(slibpath)
                @info "skipping the Python builder leg" python = py lib = isfile(slibpath)
                @test_skip false
            else
                script = joinpath(@__DIR__, "builder_check.py")
                outfile = joinpath(mktempdir(), "py.txt")
                env = copy(ENV)
                sep = Sys.iswindows() ? ";" : ":"
                env["PYTHONPATH"] =
                    pysrc * (haskey(env, "PYTHONPATH") ? sep * env["PYTHONPATH"] : "")
                run(setenv(`$py $script $slibpath structm $S_N $outfile`, env))

                vals = Dict{String,Vector{Float64}}()
                for line in eachline(outfile)
                    parts = split(line)
                    vals[parts[1]] = parse.(Float64, parts[2:end])
                end
                # The script's inputs mirror S_V0/S_LO/S_TAB — the reference
                # is the same in-Julia model the in-process leg checked.
                ref = ExaModel(sbuild(), S_N, (v0 = S_V0, lo = S_LO), S_TAB)
                x = collect(range(0.5, 3.0; length = S_N))
                @test only(vals["nvar"]) == ref.meta.nvar
                @test only(vals["ncon"]) == ref.meta.ncon
                @test only(vals["obj"]) ≈ NLPModels.obj(ref, x)
                @test vals["grad"] ≈ NLPModels.grad(ref, x)
                @test vals["cons"] ≈ NLPModels.cons(ref, x)
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
