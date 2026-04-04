module COPSApp

using ExaModels, COPSBenchmark, NLPModelsIpoptLite

const MODELS = [
    "camshape",
    "bearing",
    "catmix",
    "chain",
    "glider",
    "gasoil",
    "marine",
    "methanol",
    "minsurf",
    "pinene",
    "robot",
    "rocket",
    "steering",
    "torsion",
]

# @inline wrappers with explicit T = Float64 so juliac --trim=safe can resolve
# the concrete ExaCore type without needing to propagate default keyword args.
@inline _camshape(N)  = COPSBenchmark.camshape_model(COPSBenchmark.ExaModelsBackend(), N;    T = Float64)
@inline _bearing(N)   = COPSBenchmark.bearing_model(COPSBenchmark.ExaModelsBackend(),  N, N; T = Float64)
@inline _catmix(N)    = COPSBenchmark.catmix_model(COPSBenchmark.ExaModelsBackend(),   N;    T = Float64)
@inline _chain(N)     = COPSBenchmark.chain_model(COPSBenchmark.ExaModelsBackend(),    N;    T = Float64)
@inline _glider(N)    = COPSBenchmark.glider_model(COPSBenchmark.ExaModelsBackend(),   N;    T = Float64)
@inline _gasoil(N)    = COPSBenchmark.gasoil_model(COPSBenchmark.ExaModelsBackend(),   N;    T = Float64)
@inline _marine(N)    = COPSBenchmark.marine_model(COPSBenchmark.ExaModelsBackend(),   N;    T = Float64)
@inline _methanol(N)  = COPSBenchmark.methanol_model(COPSBenchmark.ExaModelsBackend(), N;    T = Float64)
@inline _minsurf(N)   = COPSBenchmark.minsurf_model(COPSBenchmark.ExaModelsBackend(),  N, N; T = Float64)
@inline _pinene(N)    = COPSBenchmark.pinene_model(COPSBenchmark.ExaModelsBackend(),   N;    T = Float64)
@inline _robot(N)     = COPSBenchmark.robot_model(COPSBenchmark.ExaModelsBackend(),    N;    T = Float64)
@inline _rocket(N)    = COPSBenchmark.rocket_model(COPSBenchmark.ExaModelsBackend(),   N;    T = Float64)
@inline _steering(N)  = COPSBenchmark.steering_model(COPSBenchmark.ExaModelsBackend(), N;    T = Float64)
@inline _torsion(N)   = COPSBenchmark.torsion_model(COPSBenchmark.ExaModelsBackend(),  N, N; T = Float64)
@inline _elec(N)       = COPSBenchmark.elec_model(COPSBenchmark.ExaModelsBackend(), N; T = Float64)
@inline _lane_emden(N) = COPSBenchmark.lane_emden_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _polygon(N)    = COPSBenchmark.polygon_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _channel(N)   = COPSBenchmark.channel_model(COPSBenchmark.ExaModelsBackend(),  N; T = Float64)
@inline _dirichlet(N)  = COPSBenchmark.dirichlet_model(COPSBenchmark.ExaModelsBackend(), N; T = Float64)
@inline _henon(N)     = COPSBenchmark.henon_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _tetra_duct12(N) = COPSBenchmark.tetra_duct12_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _tetra_duct15(N) = COPSBenchmark.tetra_duct15_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _tetra_duct20(N) = COPSBenchmark.tetra_duct20_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _tetra_foam5(N)  = COPSBenchmark.tetra_foam5_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _tetra_gear(N)   = COPSBenchmark.tetra_gear_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _tetra_hook(N)  = COPSBenchmark.tetra_hook_model(COPSBenchmark.ExaModelsBackend(), N;  T = Float64)
@inline _triangle_deer(N)   = COPSBenchmark.triangle_deer_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)                                                          
@inline _triangle_pacman(N) = COPSBenchmark.triangle_pacman_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)
@inline _triangle_turtle(N) = COPSBenchmark.triangle_turtle_model(COPSBenchmark.ExaModelsBackend(), N;   T = Float64)

function (@main)(ARGS)
    if length(ARGS) < 2
        println(Core.stdout, "Usage: cops model N")
        println(Core.stdout, "")
        println(Core.stdout, "Available models:")
        for name in MODELS
            println(Core.stdout, "  ", name)
        end
        return 1
    end
    name = ARGS[1]
    N    = parse(Int, ARGS[2])
    println(Core.stdout, "Solving $name (N=$N) with Ipopt...")
 
    if name == "camshape"
        result = ipopt(_camshape(N); print_level = 5)
    elseif name == "bearing"
        result = ipopt(_bearing(N); print_level = 5)
    elseif name == "catmix"
        result = ipopt(_catmix(N); print_level = 5)
    elseif name == "chain"
        result = ipopt(_chain(N); print_level = 5)
    elseif name == "glider"
        result = ipopt(_glider(N); print_level = 5)
    elseif name == "gasoil"
        result = ipopt(_gasoil(N); print_level = 5)
    elseif name == "marine"
        result = ipopt(_marine(N); print_level = 5)
    elseif name == "methanol"
        result = ipopt(_methanol(N); print_level = 5)
    elseif name == "minsurf"
        result = ipopt(_minsurf(N); print_level = 5)
    elseif name == "pinene"
        result = ipopt(_pinene(N); print_level = 5)
    elseif name == "robot"
        result = ipopt(_robot(N); print_level = 5)
    elseif name == "rocket"
        result = ipopt(_rocket(N); print_level = 5)
    elseif name == "steering"
        result = ipopt(_steering(N); print_level = 5)
    elseif name == "torsion"
        result = ipopt(_torsion(N); print_level = 5)
    elseif name == "elec"
        result = ipopt(_elec(N); print_level = 5)
    # elseif name == "lane_emden"
    #     result = ipopt(_lane_emden(N); print_level = 5)
    elseif name == "polygon"
        result = ipopt(_polygon(N); print_level = 5)
    elseif name == "channel"
        result = ipopt(_channel(N); print_level = 5)
    # elseif name == "dirichlet"
    #     result = ipopt(_dirichlet(N); print_level = 5)
    # elseif name == "henon"
    #     result = ipopt(_henon(N); print_level = 5)
    # elseif name == "tetra_duct12"
    #     result = ipopt(_tetra_duct12(N); print_level = 5)
    # elseif name == "tetra_duct15"
    #     result = ipopt(_tetra_duct15(N); print_level = 5)
    # elseif name == "tetra_duct20"
    #     result = ipopt(_tetra_duct20(N); print_level = 5)
    # elseif name == "tetra_foam5"
    #     result = ipopt(_tetra_foam5(N); print_level = 5)
    # elseif name == "tetra_gear"
    #     result = ipopt(_tetra_gear(N); print_level = 5)
    # elseif name == "tetra_hook"
    #     result = ipopt(_tetra_hook(N); print_level = 5)
    # elseif name == "triangle_deer"
    #     result = ipopt(_triangle_deer(N); print_level = 5)
    # elseif name == "triangle_pacman"
    #     result = ipopt(_triangle_pacman(N); print_level = 5)
    # elseif name == "triangle_turtle"
    #     result = ipopt(_triangle_turtle(N); print_level = 5) 
    else
        println(Core.stdout, "Unknown model: $name")
        return 1
    end

    println(Core.stdout, "Ipopt status : ", result.status)
    return result.status == 0 ? 0 : 1
end

end # module COPSApp
