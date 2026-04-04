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
    else
        println(Core.stdout, "Unknown model: $name")
        return 1
    end

    println(Core.stdout, "Ipopt status : ", result.status)
    return result.status <= 1 ? 0 : 1
end

end # module COPSApp
