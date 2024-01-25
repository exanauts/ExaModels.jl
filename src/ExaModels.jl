"""
    ExaModels

An algebraic modeling and automatic differentiation tool in Julia Language, specialized for SIMD abstraction of nonlinear programs.

For more information, please visit https://github.com/sshin23/ExaModels.jl
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
include("wrapper.jl")

export ExaModel,
    ExaCore,
    data,
    variable,
    objective,
    constraint,
    constraint!,
    solution,
    multipliers,
    multipliers_L,
    multipliers_U,
    WrapperNLPModel,
    @register_univariate,
    @register_bivariate

end # module ExaModels
