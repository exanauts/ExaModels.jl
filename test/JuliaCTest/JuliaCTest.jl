module JuliaCTest

using Test, JuliaC

const LUKSANVLCEK_APP_DIR = abspath(joinpath(@__DIR__, "..", "LuksanVlcekApp.jl"))
const COPS_APP_DIR        = abspath(joinpath(@__DIR__, "..", "COPSApp.jl"))

# Compile app_dir into an executable at exe_path using the JuliaC programmatic API.
# Returns true on success, false if JuliaC API is not available.
const _HAS_JULIAC_API = isdefined(JuliaC, :ImageRecipe)

# Skip AOT tests when EXAMODELS_SKIP_AOT is set (e.g. on Julia LTS or less critical CI legs)
const _SKIP_AOT = get(ENV, "EXAMODELS_SKIP_AOT", "") != ""

function _compile_exe(app_dir::String, exe_path::String)
    if !_HAS_JULIAC_API
        @warn "JuliaC.ImageRecipe not available, skipping AOT compilation test"
        return false
    end

    img = JuliaC.ImageRecipe(
        file        = app_dir,
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
                ("glider",    20, Sys.iswindows()),
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

    end
end

end # module JuliaCTest
