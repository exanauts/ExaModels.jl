module JuliaCTest

using Test

const LUKSANVLCEK_APP_DIR = abspath(joinpath(@__DIR__, "..", "LuksanVlcekApp.jl"))
const COPS_APP_DIR        = abspath(joinpath(@__DIR__, "..", "COPSApp.jl"))

const _JuliaC = try
    m = Base.require(Base.PkgId(Base.UUID("acedd4c2-ced6-4a15-accc-2607eb759ba2"), "JuliaC"))
    isdefined(m, :ImageRecipe) ? m : nothing
catch
    nothing
end

# Compile app_dir into an executable at exe_path using the JuliaC programmatic API.
# Returns true on success, false if JuliaC is not available.
function _compile_exe(app_dir::String, exe_path::String)
    if _JuliaC === nothing
        @warn "JuliaC not available (or incompatible version), skipping AOT compilation test"
        return false
    end

    img = _JuliaC.ImageRecipe(
        file        = app_dir,
        output_type = "--output-exe",
        trim_mode   = "safe",
        julia_args  = ["--experimental"],
    )
    _JuliaC.compile_products(img)

    link = _JuliaC.LinkRecipe(image_recipe = img, outname = exe_path)
    _JuliaC.link_products(link)
    return true
end

function _run_app_tests(exe_path, cases)
    isfile(exe_path) || return
    try
        for (model, n) in cases
            @testset "AOT exe: $model N=$n" begin
                out = IOBuffer()
                result = run(pipeline(
                    ignorestatus(`$exe_path $model $n`);
                    stdout = out, stderr = out,
                ))
                @test success(result)
                @test contains(String(take!(out)), "Ipopt status : 0")
            end
        end
    finally
        rm(exe_path; force = true)
    end
end

function runtests()
    @testset "AOT compilation (juliac)" begin

        # ── LuksanVlcek ──────────────────────────────────────────────────────────
        @testset "LuksanVlcekApp" begin
            exe_path = joinpath(tempdir(), "luksanvlcek_test")

            compiled = false
            @testset "juliac compiles LuksanVlcekApp" begin
                compiled = _compile_exe(LUKSANVLCEK_APP_DIR, exe_path)
                @test compiled
                compiled && @test isfile(exe_path)
            end

            compiled && _run_app_tests(exe_path, [
                ("rosenrock",            10),
                ("augmented_lagrangian", 20),
                ("broyden_tridiagonal",  10),
            ])
        end

        # ── COPSBenchmark ─────────────────────────────────────────────────────────
        @testset "COPSApp" begin
            exe_path = joinpath(tempdir(), "cops_test")

            compiled = false
            @testset "juliac compiles COPSApp" begin
                compiled = _compile_exe(COPS_APP_DIR, exe_path)
                @test compiled
                compiled && @test isfile(exe_path)
            end

            compiled && _run_app_tests(exe_path, [
                ("camshape",  50),
                ("bearing",   10),
                ("catmix",    10),
                ("chain",     20),
                ("gasoil",    10),
                ("glider",    20),
                ("marine",    10),
                ("minsurf",   10),
                ("pinene",    10),
                ("robot",     20),
                ("rocket",    20),
                ("steering",  20),
                ("torsion",   10),
            ])
        end

    end
end

end # module JuliaCTest
