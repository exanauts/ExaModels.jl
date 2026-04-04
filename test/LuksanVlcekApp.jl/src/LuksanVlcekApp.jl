module LuksanVlcekApp

using ExaModels, NLPModelsIpoptLite
import LuksanVlcekBenchmark as LV

const MODELS = [
    "rosenrock",
    "wood",
    "augmented_lagrangian",
    "broyden_banded",
    "broyden_tridiagonal",
    "chained_powell",
    "cragg_levy",
    "generalized_brown",
    "modified_brown",
    "trigo_tridiagonal",
    "Chained_HS46",
    "Chained_HS47",
    "Chained_HS48",
    "Chained_HS49",
    "Chained_HS50",
    "Chained_HS51",
    "Chained_HS52",
    "Chained_HS53",
]

function (@main)(ARGS)
    if length(ARGS) < 2
        println(Core.stdout, "Usage: luksanvlcek model N")
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
    
    if name == "rosenrock"
        m = LV.rosenrock_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)
        
    elseif name == "wood"
        m = LV.wood_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)

    elseif name == "augmented_lagrangian"
        m = LV.augmented_lagrangian_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)

    # elseif name == "broyden_banded"
    #     m = LV.broyden_banded_model(LV.ExaModelsBackend(), N)
    #     result = ipopt(m; print_level = 5)
        
    elseif name == "broyden_tridiagonal"
        m = LV.broyden_tridiagonal_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)
        
    # elseif name == "chained_powell"
    #     m = LV.chained_powell_model(LV.ExaModelsBackend(), N)
    #     result = ipopt(m; print_level = 5)
        
    # elseif name == "cragg_levy"
    #     m = LV.cragg_levy_model(LV.ExaModelsBackend(), N)
    #     result = ipopt(m; print_level = 5)
        
    # elseif name == "generalized_brown"
    #     m = LV.generalized_brown_model(LV.ExaModelsBackend(), N)
    #     result = ipopt(m; print_level = 5)
        
    # elseif name == "modified_brown"
    #     m = LV.modified_brown_model(LV.ExaModelsBackend(), N)
    #     result = ipopt(m; print_level = 5)
        
    # elseif name == "trigo_tridiagonal"
    #     m = LV.trigo_tridiagonal_model(LV.ExaModelsBackend(), N)
    #     result = ipopt(m; print_level = 5)
        
    elseif name == "Chained_HS46"
        m = LV.Chained_HS46_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)

    elseif name == "Chained_HS47"
        m = LV.Chained_HS47_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)

    elseif name == "Chained_HS48"
        m = LV.Chained_HS48_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)
        
    elseif name == "Chained_HS49"
        m = LV.Chained_HS49_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)

    elseif name == "Chained_HS50"
        m = LV.Chained_HS50_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)

    elseif name == "Chained_HS51"
        m = LV.Chained_HS51_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)

    elseif name == "Chained_HS52"
        m = LV.Chained_HS52_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)

    elseif name == "Chained_HS53"
        m = LV.Chained_HS53_model(LV.ExaModelsBackend(), N)
        result = ipopt(m; print_level = 5)
    else
        println(Core.stdout, "Unknown model: $name")
        return 1
    end
    
    println(Core.stdout, "Ipopt status : ", result.status)
    return result.status <= 1 ? 0 : 1
end

end # module LuksanVlcekApp
