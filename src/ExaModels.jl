"""
    ExaModels

An algebraic modeling and automatic differentiation tool in Julia Language, specialized for SIMD abstraction of nonlinear programs.

For more information, please visit https://github.com/exanauts/ExaModels.jl
"""
module ExaModels

import NLPModels:
    NLPModels,
    obj,
    cons!,
    grad!,
    jac_coord!,
    hess_coord!,
    jprod!,
    jtprod!,
    hprod!,
    jac_structure!,
    hess_structure!,
    cons_nln!,
    jtprod_nln!,
    jprod_nln!
import SolverCore
import Printf

include("templates.jl")
include("graph.jl")
include("register.jl")
include("specialization.jl")
include("functionlist.jl")
include("simdfunction.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")
include("nlp.jl")
include("utils.jl")
include("two-stage.jl")

export ExaModel,
    ExaCore,
    ExaModelsBackend,
    Subexpr,
    ReducedSubexpr,
    ParameterSubexpr,
    data,
    variable,
    subexpr,
    parameter,
    set_parameter!,
    objective,
    constraint,
    constraint!,
    solution,
    multipliers,
    multipliers_L,
    multipliers_U,
    @register_univariate,
    @register_bivariate,
    # Stochastic optimization exports
    TwoStageExaModel,
    num_scenarios,
    num_recourse_vars,
    num_design_vars,
    num_constraints_per_scenario,
    total_vars,
    total_cons,
    set_scenario_parameters!,
    set_all_scenario_parameters!,
    # Index range functions (GPU-friendly)
    recourse_var_indices,
    design_var_indices,
    cons_block_indices,
    grad_recourse_indices,
    grad_design_indices,
    # In-place extraction (zero allocation)
    extract_recourse_vars!,
    extract_design_vars!,
    extract_cons_block!,
    extract_grad_block!,
    # Index mapping
    global_var_index,
    global_con_index,
    recourse_var_index,
    design_var_index,
    get_model

end # module ExaModels
