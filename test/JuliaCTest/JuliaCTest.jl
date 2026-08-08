module JuliaCTest

using Test, JuliaC
using Libdl, LinearAlgebra
import ExaModels
import NLPModels
import MadNLP

const LUKSANVLCEK_APP_DIR = abspath(joinpath(@__DIR__, "..", "LuksanVlcekApp.jl"))
const COPS_APP_DIR        = abspath(joinpath(@__DIR__, "..", "COPSApp.jl"))
const ORACLE_APP_DIR = abspath(joinpath(@__DIR__, "..", "OracleApp.jl"))
const RECORDER_APP_DIR = abspath(joinpath(@__DIR__, "..", "RecorderApp.jl"))

# Compile app_dir into an executable at exe_path using the JuliaC programmatic API.
# Returns true on success, false if JuliaC API is not available.
const _HAS_JULIAC_API = isdefined(JuliaC, :ImageRecipe)

# Skip AOT tests when EXAMODELS_SKIP_AOT is set (e.g. on Julia LTS or less critical CI legs)
const _SKIP_AOT = get(ENV, "EXAMODELS_SKIP_AOT", "") != ""

# JuliaC.compile_products copies the app dir into a fresh mktempdir() before
# Pkg.instantiate(), which breaks any relative `path = "../..."` entries in
# Project.toml / Manifest.toml because the copy is made into a sibling temp
# dir, not the original location. Stage the app into a temp dir with those
# relative paths rewritten to absolute paths so they survive JuliaC's copy.
function _stage_app(app_dir::String)
    stage = mktempdir()
    staged_app = joinpath(stage, basename(app_dir))
    cp(app_dir, staged_app)
    chmod(staged_app, 0o755; recursive = true)
    for fname in ("Project.toml", "Manifest.toml", "JuliaProject.toml", "JuliaManifest.toml")
        fpath = joinpath(staged_app, fname)
        isfile(fpath) || continue
        text = read(fpath, String)
        # Match path = "..." entries with relative paths and absolutize them
        # relative to the *original* app_dir.
        text = replace(
            text, r"path\s*=\s*\"([^\"]+)\""m =>
                s -> begin
                m = match(r"path\s*=\s*\"([^\"]+)\"", s)
                rel = m.captures[1]
                isabspath(rel) && return s
                # Normalize to forward slashes so the path is a valid TOML
                # basic-string on Windows (backslashes are escape characters).
                abs_path = replace(abspath(joinpath(app_dir, rel)), '\\' => '/')
                "path = \"$abs_path\""
            end,
        )
        write(fpath, text)
    end
    return staged_app
end

# Minimal host-side NLPModels wrapper over the generated C ABI (the full
# consumer lives in CNLPModels.jl; this test-local copy keeps the suite free
# of that dependency). Handle-based: <prefix>_new(n) -> id, id-first calls.
struct _CLibModel <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    h::Ptr{Cvoid}
    p::String
    id::Cint
end

function _CLibModel(h::Ptr{Cvoid}, prefix::AbstractString, n::Integer)
    f(s) = Libdl.dlsym(h, Symbol(prefix, "_", s))
    id = ccall(f(:new), Cint, (Cint,), Cint(n))
    id > 0 || error("$(prefix)_new($n) failed")
    nvar = Int(ccall(f(:nvar), Cint, (Cint,), id))
    ncon = Int(ccall(f(:ncon), Cint, (Cint,), id))
    nnzj = Int(ccall(f(:nnzj), Cint, (Cint,), id))
    nnzh = Int(ccall(f(:nnzh), Cint, (Cint,), id))
    x0 = zeros(nvar); lvar = zeros(nvar); uvar = zeros(nvar)
    lcon = zeros(ncon); ucon = zeros(ncon)
    ccall(f(:meta), Cint,
        (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        id, x0, lvar, uvar, lcon, ucon) == 0 || error("meta failed")
    meta = NLPModels.NLPModelMeta(
        nvar; ncon, nnzj, nnzh, x0, lvar, uvar, lcon, ucon, minimize = true)
    return _CLibModel(meta, NLPModels.Counters(), h, String(prefix), id)
end

_f(m::_CLibModel, s) = Libdl.dlsym(m.h, Symbol(m.p, "_", s))

function NLPModels.obj(m::_CLibModel, x::AbstractVector{Float64})
    out = Ref{Cdouble}(0.0)
    ccall(_f(m, :obj), Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, out) == 0 ||
        error("obj failed")
    return out[]
end
function NLPModels.grad!(m::_CLibModel, x::AbstractVector{Float64}, g::AbstractVector{Float64})
    ccall(_f(m, :grad), Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, g) == 0 ||
        error("grad failed")
    return g
end
function NLPModels.cons!(m::_CLibModel, x::AbstractVector{Float64}, c::AbstractVector{Float64})
    ccall(_f(m, :cons), Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, c) == 0 ||
        error("cons failed")
    return c
end
function NLPModels.jac_structure!(m::_CLibModel, rows::AbstractVector, cols::AbstractVector)
    r = Vector{Cint}(undef, m.meta.nnzj); c = Vector{Cint}(undef, m.meta.nnzj)
    ccall(_f(m, :jac_structure), Cint, (Cint, Ptr{Cint}, Ptr{Cint}), m.id, r, c) == 0 ||
        error("jac_structure failed")
    copyto!(rows, r); copyto!(cols, c)
    return rows, cols
end
function NLPModels.jac_coord!(m::_CLibModel, x::AbstractVector{Float64}, v::AbstractVector{Float64})
    ccall(_f(m, :jac), Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, v) == 0 ||
        error("jac failed")
    return v
end
function NLPModels.hess_structure!(m::_CLibModel, rows::AbstractVector, cols::AbstractVector)
    r = Vector{Cint}(undef, m.meta.nnzh); c = Vector{Cint}(undef, m.meta.nnzh)
    ccall(_f(m, :hess_structure), Cint, (Cint, Ptr{Cint}, Ptr{Cint}), m.id, r, c) == 0 ||
        error("hess_structure failed")
    copyto!(rows, r); copyto!(cols, c)
    return rows, cols
end
function NLPModels.hess_coord!(
    m::_CLibModel, x::AbstractVector{Float64}, y::AbstractVector{Float64},
    v::AbstractVector{Float64}; obj_weight::Real = 1.0,
)
    ccall(_f(m, :hess), Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ptr{Cdouble}),
        m.id, x, y, Cdouble(obj_weight), v) == 0 || error("hess failed")
    return v
end

function _compile_exe(app_dir::String, exe_path::String)
    if !_HAS_JULIAC_API
        @warn "JuliaC.ImageRecipe not available, skipping AOT compilation test"
        return false
    end

    staged_app = _stage_app(app_dir)

    img = JuliaC.ImageRecipe(
        file = staged_app,
        output_type = "--output-exe",
        trim_mode   = "safe",
        julia_args  = ["--experimental"],
    )
    JuliaC.compile_products(img)

    link = JuliaC.LinkRecipe(image_recipe = img, outname = exe_path)
    JuliaC.link_products(link)
    return true
end

function _run_app_tests(exe_path, cases)
    isfile(exe_path) || return
    try
        for (model, n, broken) in cases
            @testset "AOT exe: $model N=$n" begin
                out = IOBuffer()
                result = run(pipeline(
                    ignorestatus(`$exe_path $model $n`);
                    stdout = out, stderr = out,
                ))
                @test success(result) broken=broken
                @test contains(String(take!(out)), "Ipopt status : 0") broken=broken
            end
        end
    finally
        rm(exe_path; force = true)
    end
end

function runtests()
    if _SKIP_AOT
        @info "Skipping AOT tests (EXAMODELS_SKIP_AOT is set)"
        @testset "AOT compilation (juliac)" begin end
        return
    end
    @testset "AOT compilation (juliac)" begin

        # ── LuksanVlcek ──────────────────────────────────────────────────────────
        @testset "LuksanVlcekApp" begin
            exe_path = joinpath(tempdir(), "luksanvlcek_test" * (Sys.iswindows() ? ".exe" : ""))

            compiled = false
            @testset "juliac compiles LuksanVlcekApp" begin
                compiled = _compile_exe(LUKSANVLCEK_APP_DIR, exe_path)
                @test compiled skip=!_HAS_JULIAC_API
                compiled && @test isfile(exe_path)
            end

            compiled && _run_app_tests(exe_path, [
                ("rosenrock",            10, false),
                ("augmented_lagrangian", 20, false),
                ("broyden_tridiagonal",  10, false),
            ])
        end

        # ── COPSBenchmark ─────────────────────────────────────────────────────────
        @testset "COPSApp" begin
            exe_path = joinpath(tempdir(), "cops_test" * (Sys.iswindows() ? ".exe" : ""))

            compiled = false
            @testset "juliac compiles COPSApp" begin
                compiled = _compile_exe(COPS_APP_DIR, exe_path)
                @test compiled skip=!_HAS_JULIAC_API
                compiled && @test isfile(exe_path)
            end

            # Run a representative subset at runtime — compilation of all 28 models
            # is already verified by the juliac compile step above.
            compiled && _run_app_tests(exe_path, [
                ("camshape",  50, false),
                ("bearing",   10, false),
                ("catmix",    10, false),
                ("chain",     20, false),
                ("gasoil",    10, false),
                ("glider",    20, false),
                ("marine",    10, false),
                ("minsurf",   10, false),
                ("pinene",    10, false),
                ("robot",     20, false),
                ("rocket",    20, false),
                ("steering",  20, false),
                ("torsion",   10, false),
                ("elec",      10, false),
                ("channel",   10, false),
            ])
        end

        # ── OracleApp ─────────────────────────────────────────────────────────────
        # Verifies that VectorNonlinearOracle with hand-written (named function)
        # callbacks survives juliac AOT compilation and produces the correct result.
        @testset "OracleApp" begin
            exe_path = joinpath(tempdir(), "oracle_aot_test" * (Sys.iswindows() ? ".exe" : ""))

            compiled = false
            @testset "juliac compiles OracleApp" begin
                compiled = _compile_exe(ORACLE_APP_DIR, exe_path)
                @test compiled skip = !_HAS_JULIAC_API
                compiled && @test isfile(exe_path)
            end

            if compiled
                @testset "AOT exe: oracle NLP" begin
                    out = IOBuffer()
                    result = run(pipeline(ignorestatus(`$exe_path`); stdout = out, stderr = out))
                    @test success(result)
                    @test contains(String(take!(out)), "Ipopt status : 0")
                end
                rm(exe_path; force = true)
            end
        end

        # ── RecorderApp ───────────────────────────────────────────────────────────
        # Verifies that a tape recorded at precompile time replays, builds, and
        # solves inside a trimmed binary — the recorder's core AOT property.
        # The replay sizes deliberately differ from the recording template (N = 4).
        @testset "RecorderApp" begin
            exe_path = joinpath(tempdir(), "recorder_aot_test" * (Sys.iswindows() ? ".exe" : ""))

            compiled = false
            @testset "juliac compiles RecorderApp" begin
                compiled = _compile_exe(RECORDER_APP_DIR, exe_path)
                @test compiled skip = !_HAS_JULIAC_API
                compiled && @test isfile(exe_path)
            end

            if compiled
                try
                    @testset "AOT exe: replay at N=$n" for n in (10, 1000)
                        out = IOBuffer()
                        result = run(pipeline(
                            ignorestatus(`$exe_path $n`);
                            stdout = out, stderr = out,
                        ))
                        @test success(result)
                        @test contains(String(take!(out)), "Ipopt status : 0")
                    end
                finally
                    rm(exe_path; force = true)
                end
            end
        end

        # ── individually compiled models, host-side solving ───────────────────
        # One evaluation-only library per model (compile_library: tape at the
        # generated package's precompile, no solver compiled in), consumed
        # through the C ABI and solved by a HOST solver (MadNLP here — any
        # NLPModels solver works, since the host is not trimmed). Each model
        # runs in its own subprocess: loading several privatized runtimes into
        # one long-lived driver is exactly the interaction this avoids.
        @testset "individual model libraries: $mname" for mname in
                ("rosenrock", "wood", "two_blocks")
            script = """
                using ExaModels, JuliaC, MadNLP, Libdl, LinearAlgebra
                include($(repr(joinpath(@__DIR__, "JuliaCTest.jl"))))
                r = ExaModels.compile_library(
                    $(repr(joinpath(@__DIR__, "models", mname * ".jl")));
                    prefix = "m", out = mktempdir(), template_n = 8)
                blas = [l.libname for l in BLAS.get_config().loaded_libs]
                h = Libdl.dlopen(r.libpath, Libdl.RTLD_LOCAL | Libdl.RTLD_DEEPBIND)
                m = JuliaCTest._CLibModel(h, "m", 40)
                for (i, l) in enumerate(blas)
                    BLAS.lbt_forward(l; clear = (i == 1))
                end
                res = MadNLP.madnlp(m; print_level = MadNLP.ERROR)
                mod = Module(:MRef)
                Core.eval(mod, :(using ExaModels))
                Base.include(mod, $(repr(joinpath(@__DIR__, "models", mname * ".jl"))))
                m_ref = Base.invokelatest() do
                    tape = ExaModels.record(mod.build, mod.make_data(8))
                    ExaModels.ExaModel(tape, mod.make_data(40))
                end
                res_ref = MadNLP.madnlp(m_ref; print_level = MadNLP.ERROR)
                ok = res.status == MadNLP.SOLVE_SUCCEEDED &&
                     res_ref.status == MadNLP.SOLVE_SUCCEEDED &&
                     isapprox(res.objective, res_ref.objective; rtol = 1e-8)
                println("MODEL_LIB_RESULT : ", ok ? 0 : 1)
                """
            out = IOBuffer()
            result = run(pipeline(
                ignorestatus(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`);
                stdout = out, stderr = out,
            ))
            txt = String(take!(out))
            @test success(result)
            @test contains(txt, "MODEL_LIB_RESULT : 0")
        end

        # ── compile_library ───────────────────────────────────────────────────
        # One-command model-file → shared-library path (ExaModelsJuliaC ext).
        # The library is consumed here through plain Libdl, and its obj value
        # is checked against the same tape replayed in-process.
        @testset "compile_library" begin
            if _HAS_JULIAC_API
                r = ExaModels.compile_library(
                    joinpath(@__DIR__, "recorder_lib_model.jl");
                    prefix = "lv", out = mktempdir(),
                )
                @test isfile(r.libpath)

                # The library's lazy runtime init re-forwards the process-global
                # libblastrampoline; snapshot and restore so later testsets keep
                # a working BLAS. (The library itself stays loaded; a Julia
                # runtime cannot be safely unloaded.)
                blas_libs = [l.libname for l in BLAS.get_config().loaded_libs]
                h = Libdl.dlopen(r.libpath, Libdl.RTLD_LOCAL | Libdl.RTLD_DEEPBIND)
                f(s) = Libdl.dlsym(h, s)
                id = ccall(f(:lv_new), Cint, (Cint,), Cint(10))
                @test id > 0
                for (i, l) in enumerate(blas_libs)
                    BLAS.lbt_forward(l; clear = (i == 1))
                end

                @test Int(ccall(f(:lv_nvar), Cint, (Cint,), id)) == 10
                @test Int(ccall(f(:lv_ncon), Cint, (Cint,), id)) == 8

                x0 = zeros(10); lv = zeros(10); uv = zeros(10)
                lc = zeros(8); uc = zeros(8)
                @test ccall(f(:lv_meta), Cint,
                    (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                    id, x0, lv, uv, lc, uc) == 0

                out = Ref{Cdouble}(0.0)
                @test ccall(f(:lv_obj), Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), id, x0, out) == 0

                # A second instance must not disturb the first.
                id2 = ccall(f(:lv_new), Cint, (Cint,), Cint(20))
                @test id2 > 0 && id2 != id
                out2 = Ref{Cdouble}(0.0)
                @test ccall(f(:lv_obj), Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), id, x0, out2) == 0
                @test out2[] == out[]

                # In-process reference from the same model file.
                mod = Module(:RecorderLibRef)
                Core.eval(mod, :(using ExaModels))
                Base.include(mod, joinpath(@__DIR__, "recorder_lib_model.jl"))
                # The file's methods are newer than this frame's world age, and
                # so are the traced closures inside the reference model:
                # construct and evaluate in the latest world.
                m_ref = Base.invokelatest() do
                    tape = ExaModels.record(mod.build, mod.make_data(4))
                    ExaModels.ExaModel(ExaModels.replay(tape, mod.make_data(10)))
                end
                @test x0 == m_ref.meta.x0
                @test out[] ≈ Base.invokelatest(NLPModels.obj, m_ref, x0) rtol = 1e-14

                # ── tape-input form: a tree-built tape, no model file ─────────
                data = ExaModels.DataTracer((; N = 4))
                t = ExaModels.ExaTape()
                t, xv = ExaModels.add_var(t, data.N; start = -0.5)
                i = ExaModels.DataSource()
                t, _ = ExaModels.add_con(t, 3xv[i+1]^3 + 2xv[i+2] - 5, 1:data.N-2)
                t, _ = ExaModels.add_obj(t, (xv[i-1] - 1)^2, 2:data.N)
                r2 = ExaModels.compile_library(
                    t; template = (; N = 4), prefix = "tt", out = mktempdir(),
                )
                @test isfile(r2.libpath)
                blas_libs2 = [l.libname for l in BLAS.get_config().loaded_libs]
                h2 = Libdl.dlopen(r2.libpath, Libdl.RTLD_LOCAL | Libdl.RTLD_DEEPBIND)
                f2(s) = Libdl.dlsym(h2, s)
                id2t = ccall(f2(:tt_new), Cint, (Cint,), Cint(12))
                @test id2t > 0
                for (k, l) in enumerate(blas_libs2)
                    BLAS.lbt_forward(l; clear = (k == 1))
                end
                @test Int(ccall(f2(:tt_nvar), Cint, (Cint,), id2t)) == 12
                x0t = zeros(12); lvt = zeros(12); uvt = zeros(12)
                lct = zeros(10); uct = zeros(10)
                @test ccall(f2(:tt_meta), Cint,
                    (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                    id2t, x0t, lvt, uvt, lct, uct) == 0
                outt = Ref{Cdouble}(0.0)
                @test ccall(f2(:tt_obj), Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), id2t, x0t, outt) == 0
                # The same tape replayed in-process is the reference (tree
                # tapes contain no runtime-included methods: no world-age gap).
                m_t = ExaModels.ExaModel(ExaModels.replay(t, (; N = 12)))
                @test x0t == m_t.meta.x0
                @test outt[] ≈ NLPModels.obj(m_t, x0t) rtol = 1e-14
            else
                @warn "JuliaC.ImageRecipe not available, skipping compile_library test"
            end
        end

    end
end

end # module JuliaCTest
