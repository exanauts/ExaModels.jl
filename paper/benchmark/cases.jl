# ============================================================================
# Benchmark problem definitions
# ============================================================================
#
# Sizes target nnzh ≈ 1e2 (small), 1e4 (medium), 1e6 (large) per problem.

using LuksanVlcekBenchmark
using COPSBenchmark
using ExaModelsPower

# --- Lukšan–Vlček problems (scalable equality-constrained) ------------------
# nnzh scales roughly as c·N where c ∈ [2, 10] depending on problem.
# N ∈ {20, 2000, 200000} gives nnzh ~ {40-200, 4k-20k, 400k-2M}.
const LV_CASES = [
    (name = "rosenrock",            model = LuksanVlcekBenchmark.rosenrock_model,            sizes = [20, 2000, 200000]),
    (name = "wood",                 model = LuksanVlcekBenchmark.wood_model,                 sizes = [20, 2000, 200000]),
    (name = "chained_powell",       model = LuksanVlcekBenchmark.chained_powell_model,       sizes = [20, 2000, 200000]),
    (name = "cragg_levy",           model = LuksanVlcekBenchmark.cragg_levy_model,           sizes = [20, 2000, 200000]),
    (name = "broyden_tridiagonal",  model = LuksanVlcekBenchmark.broyden_tridiagonal_model,  sizes = [20, 2000, 200000]),
    (name = "broyden_banded",       model = LuksanVlcekBenchmark.broyden_banded_model,       sizes = [20, 2000, 200000]),
    (name = "trigo_tridiagonal",    model = LuksanVlcekBenchmark.trigo_tridiagonal_model,    sizes = [20, 2000, 200000]),
    (name = "augmented_lagrangian", model = LuksanVlcekBenchmark.augmented_lagrangian_model, sizes = [20, 2000, 200000]),
    (name = "modified_brown",       model = LuksanVlcekBenchmark.modified_brown_model,       sizes = [20, 2000, 200000]),
    (name = "generalized_brown",    model = LuksanVlcekBenchmark.generalized_brown_model,    sizes = [20, 2000, 200000]),
    (name = "chained_hs46",         model = LuksanVlcekBenchmark.Chained_HS46_model,         sizes = [20, 2000, 200000]),
    (name = "chained_hs47",         model = LuksanVlcekBenchmark.Chained_HS47_model,         sizes = [20, 2000, 200000]),
    (name = "chained_hs48",         model = LuksanVlcekBenchmark.Chained_HS48_model,         sizes = [20, 2000, 200000]),
    (name = "chained_hs49",         model = LuksanVlcekBenchmark.Chained_HS49_model,         sizes = [20, 2000, 200000]),
    (name = "chained_hs50",         model = LuksanVlcekBenchmark.Chained_HS50_model,         sizes = [20, 2000, 200000]),
    (name = "chained_hs51",         model = LuksanVlcekBenchmark.Chained_HS51_model,         sizes = [20, 2000, 200000]),
    (name = "chained_hs52",         model = LuksanVlcekBenchmark.Chained_HS52_model,         sizes = [20, 2000, 200000]),
    (name = "chained_hs53",         model = LuksanVlcekBenchmark.Chained_HS53_model,         sizes = [20, 2000, 200000]),
]

# --- COPS problems (scalable discretized optimal control / PDE) -------------
# Sizes per problem chosen to target nnzh ≈ 1e2, 1e4, 1e6.
const COPS_CASES = [
    # bearing: nnzh ≈ 10n². (5,5)→250, (35,35)→12k, (320,320)→1M
    (name = "bearing",     model = COPSBenchmark.bearing_model,     sizes = [(5,5), (35,35), (320,320)]),
    # camshape: nnzh ≈ 6N. 20→120, 2000→12k, 170000→1M
    (name = "camshape",    model = COPSBenchmark.camshape_model,    sizes = [20, 2000, 170000]),
    # catmix: nnzh ≈ 51N. 3→153, 200→10k, 20000→1M
    (name = "catmix",      model = COPSBenchmark.catmix_model,      sizes = [3, 200, 20000]),
    # chain: nnzh ≈ 2N. 50→100, 5000→10k, 500000→1M
    (name = "chain",       model = COPSBenchmark.chain_model,       sizes = [50, 5000, 500000]),
    # channel: nnzh ≈ 24N. 5→120, 400→10k, 42000→1M
    (name = "channel",     model = COPSBenchmark.channel_model,     sizes = [5, 400, 42000]),
    # dirichlet: nnzh starts at 71k for N=5. Only medium/large possible.
    (name = "dirichlet",   model = COPSBenchmark.dirichlet_model,   sizes = [5, 20, 100]),
    # elec: nnzh ≈ O(N³). 5→120, 50→26k, 200→418k
    (name = "elec",        model = COPSBenchmark.elec_model,        sizes = [5, 50, 200]),
    # gasoil: nnzh ≈ 56N. 2→112, 200→11k, 18000→1M
    (name = "gasoil",      model = COPSBenchmark.gasoil_model,      sizes = [2, 200, 18000]),
    # glider: nnzh ≈ 126N. 2→252, 80→10k, 8000→1M
    (name = "glider",      model = COPSBenchmark.glider_model,      sizes = [2, 80, 8000]),
    # marine: nnzh ≈ 74N. 2→148, 150→11k, 14000→1M
    (name = "marine",      model = COPSBenchmark.marine_model,      sizes = [2, 150, 14000]),
    # methanol: nnzh ≈ 246N. 2→492, 50→12k, 4000→984k
    (name = "methanol",    model = COPSBenchmark.methanol_model,    sizes = [2, 50, 4000]),
    # minsurf: nnzh ≈ 13n². (5,5)→325, (30,30)→12k, (280,280)→1M
    (name = "minsurf",     model = COPSBenchmark.minsurf_model,     sizes = [(5,5), (30,30), (280,280)]),
    # pinene: nnzh ≈ 105N. 2→210, 100→10k, 10000→1M
    (name = "pinene",      model = COPSBenchmark.pinene_model,      sizes = [2, 100, 10000]),
    # polygon: nnzh ≈ O(N³). 10→140, 100→13k, 800→802k
    (name = "polygon",     model = COPSBenchmark.polygon_model,     sizes = [10, 100, 800]),
    # robot: nnzh ≈ 67N. 2→134, 150→10k, 15000→1M
    (name = "robot",       model = COPSBenchmark.robot_model,       sizes = [2, 150, 15000]),
    # rocket: nnzh ≈ 59N. 2→118, 200→12k, 17000→1M
    (name = "rocket",      model = COPSBenchmark.rocket_model,      sizes = [2, 200, 17000]),
    # steering: nnzh ≈ 24N. 5→120, 400→10k, 42000→1M
    (name = "steering",    model = COPSBenchmark.steering_model,    sizes = [5, 400, 42000]),
    # torsion: nnzh ≈ 10n². (5,5)→250, (35,35)→12k, (320,320)→1M
    (name = "torsion",     model = COPSBenchmark.torsion_model,     sizes = [(5,5), (35,35), (320,320)]),
]

# --- OPF problems (all PGLIB instances, ACP and ACR) ----------------------
const OPF_CASES = [
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case300_ieee.m",
    "pglib_opf_case1354_pegase.m",
    "pglib_opf_case1888_rte.m",
    "pglib_opf_case2383wp_k.m",
    "pglib_opf_case2736sp_k.m",
    "pglib_opf_case2853_sdet.m",
    "pglib_opf_case2868_rte.m",
    "pglib_opf_case2869_pegase.m",
    "pglib_opf_case3012wp_k.m",
    "pglib_opf_case3120sp_k.m",
    "pglib_opf_case3375wp_k.m",
    "pglib_opf_case4661_sdet.m",
    "pglib_opf_case6468_rte.m",
    "pglib_opf_case6470_rte.m",
    "pglib_opf_case6495_rte.m",
    "pglib_opf_case6515_rte.m",
    "pglib_opf_case8387_pegase.m",
    "pglib_opf_case9241_pegase.m",
    "pglib_opf_case10000_goc.m",
    "pglib_opf_case13659_pegase.m",
    "pglib_opf_case19402_goc.m",
    "pglib_opf_case24464_goc.m",
    "pglib_opf_case30000_goc.m",
    "pglib_opf_case78484_epigrids.m",
]

const OPF_FORMS = [:polar, :rect]
