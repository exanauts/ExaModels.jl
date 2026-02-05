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
include("two_stage.jl")
include("batch.jl")

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
    TwoStageExaModel,
    BatchExaModel

end # module ExaModels
