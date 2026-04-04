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
include("aot.jl")

export ExaModel,
    ExaCore,
    ExaModelsBackend,
    Subexpr,
    ReducedSubexpr,
    ParameterSubexpr,
    data,
    add_var,
    subsubexp,
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
    @register_univariate,
    @register_bivariate,
    warmup,
    precompile_model

end # module ExaModels
