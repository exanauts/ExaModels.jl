"""
    ExaModels

An algebraic modeling and automatic differentiation tool in Julia Language, specialized for SIMD abstraction of nonlinear programs.

For more information, please visit https://github.com/exanauts/ExaModels.jl
"""
module ExaModels

import Adapt: adapt
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
include("tags.jl")
include("utils.jl")

export ExaModel,
    ExaCore,
    TwoStageExaCore,
    Expression,
    add_var,
    add_par,
    add_con,
    add_con!,
    add_obj,
    add_expr,
    @var,
    @expr,
    @par,
    @obj,
    @con,
    @con!,
    set_parameter!,
    solution,
    multipliers,
    multipliers_L,
    multipliers_U,
    Const,
    SumNode,
    ProdNode,
    exa_sum,
    exa_prod,
    @register_univariate,
    @register_bivariate

end # module ExaModels
