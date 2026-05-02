# ============================================================================
# Minimal benchmark cases (fast turnaround for paper results)
# ============================================================================

using LuksanVlcekBenchmark
using COPSBenchmark
using ExaModelsPower

const LV_CASES = [
    (name = "rosenrock",            model = LuksanVlcekBenchmark.rosenrock_model,            sizes = [1000, 10000]),
    (name = "wood",                 model = LuksanVlcekBenchmark.wood_model,                 sizes = [1000, 10000]),
    (name = "chained_powell",       model = LuksanVlcekBenchmark.chained_powell_model,       sizes = [1000, 10000]),
    (name = "cragg_levy",           model = LuksanVlcekBenchmark.cragg_levy_model,           sizes = [1000, 10000]),
    (name = "broyden_tridiagonal",  model = LuksanVlcekBenchmark.broyden_tridiagonal_model,  sizes = [1000, 10000]),
    (name = "broyden_banded",       model = LuksanVlcekBenchmark.broyden_banded_model,       sizes = [1000, 10000]),
    (name = "trigo_tridiagonal",    model = LuksanVlcekBenchmark.trigo_tridiagonal_model,    sizes = [1000, 10000]),
    (name = "augmented_lagrangian", model = LuksanVlcekBenchmark.augmented_lagrangian_model, sizes = [1000, 10000]),
    (name = "modified_brown",       model = LuksanVlcekBenchmark.modified_brown_model,       sizes = [1000, 10000]),
    (name = "generalized_brown",    model = LuksanVlcekBenchmark.generalized_brown_model,    sizes = [1000, 10000]),
    (name = "chained_hs46",         model = LuksanVlcekBenchmark.Chained_HS46_model,         sizes = [1000, 10000]),
    (name = "chained_hs47",         model = LuksanVlcekBenchmark.Chained_HS47_model,         sizes = [1000, 10000]),
    (name = "chained_hs48",         model = LuksanVlcekBenchmark.Chained_HS48_model,         sizes = [1000, 10000]),
    (name = "chained_hs49",         model = LuksanVlcekBenchmark.Chained_HS49_model,         sizes = [1000, 10000]),
    (name = "chained_hs50",         model = LuksanVlcekBenchmark.Chained_HS50_model,         sizes = [1000, 10000]),
    (name = "chained_hs51",         model = LuksanVlcekBenchmark.Chained_HS51_model,         sizes = [1000, 10000]),
    (name = "chained_hs52",         model = LuksanVlcekBenchmark.Chained_HS52_model,         sizes = [1000, 10000]),
    (name = "chained_hs53",         model = LuksanVlcekBenchmark.Chained_HS53_model,         sizes = [1000, 10000]),
]

const COPS_CASES = [
    (name = "bearing",     model = COPSBenchmark.bearing_model,     sizes = [(50,50)]),
    (name = "camshape",    model = COPSBenchmark.camshape_model,    sizes = [3200]),
    (name = "catmix",      model = COPSBenchmark.catmix_model,      sizes = [1600]),
    (name = "chain",       model = COPSBenchmark.chain_model,       sizes = [3200]),
    (name = "channel",     model = COPSBenchmark.channel_model,     sizes = [3200]),
    (name = "elec",        model = COPSBenchmark.elec_model,        sizes = [200]),
    (name = "gasoil",      model = COPSBenchmark.gasoil_model,      sizes = [1600]),
    (name = "glider",      model = COPSBenchmark.glider_model,      sizes = [1600]),
    (name = "marine",      model = COPSBenchmark.marine_model,      sizes = [800]),
    (name = "methanol",    model = COPSBenchmark.methanol_model,    sizes = [1600]),
    (name = "minsurf",     model = COPSBenchmark.minsurf_model,     sizes = [(50,50)]),
    (name = "pinene",      model = COPSBenchmark.pinene_model,      sizes = [1600]),
    (name = "polygon",     model = COPSBenchmark.polygon_model,     sizes = [1600]),
    (name = "robot",       model = COPSBenchmark.robot_model,       sizes = [800]),
    (name = "rocket",      model = COPSBenchmark.rocket_model,      sizes = [6400]),
    (name = "steering",    model = COPSBenchmark.steering_model,    sizes = [6400]),
    (name = "torsion",     model = COPSBenchmark.torsion_model,     sizes = [(50,50)]),
    (name = "dirichlet",   model = COPSBenchmark.dirichlet_model,   sizes = [20]),
]

const OPF_CASES = [
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case1354_pegase.m",
]

const OPF_FORMS = [:polar, :rect]
