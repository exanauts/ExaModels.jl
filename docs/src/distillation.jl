# # [Example: Distillation Column](@id distillation)
#
# This example demonstrates the use of `subexpr` to simplify complex models.
# We show three versions of a distillation column model comparing lifted vs reduced subexpressions.
#
# ## Subexpressions in ExaModels
#
# ExaModels provides the `subexpr` function to define reusable expressions in two forms:
#
# **Lifted subexpressions** (default): Creates auxiliary variables with defining equality constraints.
# ```julia
# s = subexpr(c, x[i]^2 for i in 1:n)  # adds n variables + n constraints
# ```
#
# **Reduced subexpressions**: Inlines the expression directly, no extra variables or constraints.
# ```julia
# s = subexpr(c, x[i]^2 for i in 1:n; reduced=true)  # no extra vars/cons
# ```
#
# ## Trade-offs: Lifted vs Reduced
#
# | Aspect | Lifted | Reduced |
# |--------|--------|---------|
# | **Problem size** | Adds auxiliary vars/cons | No extra vars/cons |
# | **Expression complexity** | Simple variable references | Inlined expressions |
# | **Derivative code** | Generated once per pattern | Regenerated at each use |
#
# ## Performance Comparison (T=10)
#
# | Model | Variables | Constraints | Iterations |
# |-------|-----------|-------------|------------|
# | Original (no subexpr) | 737 | 726 | 7 |
# | Lifted | 1398 | 1387 | 7 |
# | Reduced | 737 | 726 | 7 |
#
# All three models converge to the same optimal solution. Reduced subexpressions maintain
# the original problem size while providing the code clarity benefits of subexpressions.

# ## Original Model (without subexpressions)
# This is the original formulation where expressions like `(xA[t, i] - xA[t-1, i]) / dt`
# are repeated in multiple constraints.

function distillation_column_model(T = 3; backend = nothing)

    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D = 0.2
    F = 0.4
    ybar = 0.8958
    ubar = 2.0
    alpha = 1.6
    dt = 10 / T
    xAf = 0.5
    xA0s = ExaModels.convert_array([(i, 0.5) for i in 0:(NT + 1)], backend)

    itr0 = ExaModels.convert_array(collect(Iterators.product(1:T, 1:(FT - 1))), backend)
    itr1 = ExaModels.convert_array(collect(Iterators.product(1:T, (FT + 1):NT)), backend)
    itr2 = ExaModels.convert_array(collect(Iterators.product(0:T, 0:(NT + 1))), backend)

    c = ExaCore(backend)

    xA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    yA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    u = variable(c, 0:T; start = 1.0)
    V = variable(c, 0:T; start = 1.0)
    L2 = variable(c, 0:T; start = 1.0)

    objective(c, (yA[t, 1] - ybar)^2 for t in 0:T)
    objective(c, (u[t] - ubar)^2 for t in 0:T)

    constraint(c, xA[0, i] - xA0 for (i, xA0) in xA0s)
    constraint(
        c,
        (xA[t, 0] - xA[t - 1, 0]) / dt - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t in 1:T
    )
    constraint(
        c,
        (xA[t, i] - xA[t - 1, i]) / dt -
            (1 / At) * (u[t] * D * (yA[t, i - 1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i + 1])) for
            (t, i) in itr0
    )
    constraint(
        c,
        (xA[t, FT] - xA[t - 1, FT]) / dt -
            (1 / At) * (
                F * xAf + u[t] * D * xA[t, FT - 1] - L2[t] * xA[t, FT] -
                V[t] * (yA[t, FT] - yA[t, FT + 1])
            ) for t in 1:T
    )
    constraint(
        c,
        (xA[t, i] - xA[t - 1, i]) / dt -
            (1 / At) * (L2[t] * (yA[t, i - 1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i + 1])) for
            (t, i) in itr1
    )
    constraint(
        c,
        (xA[t, NT + 1] - xA[t - 1, NT + 1]) / dt -
            (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT + 1] - V[t] * yA[t, NT + 1]) for
            t in 1:T
    )
    constraint(c, V[t] - u[t] * D - D for t in 0:T)
    constraint(c, L2[t] - u[t] * D - F for t in 0:T)
    constraint(
        c,
        yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr2
    )

    return ExaModel(c)
end

# ## Model with Lifted Subexpressions
# Uses subexpressions for time derivatives and vapor differences.
# This adds auxiliary variables and constraints but makes the model more readable.

function distillation_column_model_with_subexpr(T = 3; backend = nothing)

    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D = 0.2
    F = 0.4
    ybar = 0.8958
    ubar = 2.0
    alpha = 1.6
    dt = 10 / T
    xAf = 0.5
    xA0s = ExaModels.convert_array([(i, 0.5) for i in 0:(NT + 1)], backend)

    c = ExaCore(backend)

    ## Decision variables
    xA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    yA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    u = variable(c, 0:T; start = 1.0)
    V = variable(c, 0:T; start = 1.0)
    L2 = variable(c, 0:T; start = 1.0)

    ## Subexpressions - define common terms once
    dxA = subexpr(c, (xA[t, i] - xA[t - 1, i]) / dt for t in 1:T, i in 0:(NT + 1))
    dyA = subexpr(c, yA[t, i] - yA[t, i + 1] for t in 0:T, i in 0:NT)

    ## Objectives
    objective(c, (yA[t, 1] - ybar)^2 for t in 0:T)
    objective(c, (u[t] - ubar)^2 for t in 0:T)

    ## Initial conditions
    constraint(c, xA[0, i] - xA0 for (i, xA0) in xA0s)

    ## Condenser - now using dxA subexpression
    constraint(c, dxA[t, 0] - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t in 1:T)

    ## Rectifying section - cleaner with dxA and dyA
    itr_rect = ExaModels.convert_array(collect(Iterators.product(1:T, 1:(FT - 1))), backend)
    constraint(c, dxA[t, i] - (1 / At) * (u[t] * D * (yA[t, i - 1] - xA[t, i]) - V[t] * dyA[t, i]) for (t, i) in itr_rect)

    ## Feed tray
    constraint(c, dxA[t, FT] - (1 / At) * (F * xAf + u[t] * D * xA[t, FT - 1] - L2[t] * xA[t, FT] - V[t] * dyA[t, FT]) for t in 1:T)

    ## Stripping section
    itr_strip = ExaModels.convert_array(collect(Iterators.product(1:T, (FT + 1):NT)), backend)
    constraint(c, dxA[t, i] - (1 / At) * (L2[t] * (yA[t, i - 1] - xA[t, i]) - V[t] * dyA[t, i]) for (t, i) in itr_strip)

    ## Reboiler
    constraint(c, dxA[t, NT + 1] - (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT + 1] - V[t] * yA[t, NT + 1]) for t in 1:T)

    ## Flow relationships
    constraint(c, V[t] - u[t] * D - D for t in 0:T)
    constraint(c, L2[t] - u[t] * D - F for t in 0:T)

    ## VLE
    itr_vle = ExaModels.convert_array(collect(Iterators.product(0:T, 0:(NT + 1))), backend)
    constraint(c, yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr_vle)

    return ExaModel(c)
end

# ## Model with Reduced Subexpressions
#
# Uses reduced subexpressions for time derivatives and vapor differences.
# Unlike lifted subexpressions, reduced subexpressions do NOT add auxiliary variables
# or constraints - they inline the expression directly where used.
#
# This gives the same problem size as the original model but with cleaner code.

function distillation_column_model_reduced(T = 3; backend = nothing)

    NT = 30
    FT = 17
    Ac = 0.5
    At = 0.25
    Ar = 1.0
    D = 0.2
    F = 0.4
    ybar = 0.8958
    ubar = 2.0
    alpha = 1.6
    dt = 10 / T
    xAf = 0.5
    xA0s = ExaModels.convert_array([(i, 0.5) for i in 0:(NT + 1)], backend)

    c = ExaCore(backend)

    ## Decision variables (same as original)
    xA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    yA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    u = variable(c, 0:T; start = 1.0)
    V = variable(c, 0:T; start = 1.0)
    L2 = variable(c, 0:T; start = 1.0)

    ## Reduced subexpressions - NO extra variables or constraints!
    ## These inline the expression wherever they're used
    dxA = subexpr(c, (xA[t, i] - xA[t - 1, i]) / dt for t in 1:T, i in 0:(NT + 1); reduced = true)
    dyA = subexpr(c, yA[t, i] - yA[t, i + 1] for t in 0:T, i in 0:NT; reduced = true)

    ## Objectives
    objective(c, (yA[t, 1] - ybar)^2 for t in 0:T)
    objective(c, (u[t] - ubar)^2 for t in 0:T)

    ## Initial conditions
    constraint(c, xA[0, i] - xA0 for (i, xA0) in xA0s)

    ## Condenser - using dxA reduced subexpression (inlines the derivative)
    constraint(c, dxA[t, 0] - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t in 1:T)

    ## Rectifying section
    itr_rect = ExaModels.convert_array(collect(Iterators.product(1:T, 1:(FT - 1))), backend)
    constraint(c, dxA[t, i] - (1 / At) * (u[t] * D * (yA[t, i - 1] - xA[t, i]) - V[t] * dyA[t, i]) for (t, i) in itr_rect)

    ## Feed tray
    constraint(c, dxA[t, FT] - (1 / At) * (F * xAf + u[t] * D * xA[t, FT - 1] - L2[t] * xA[t, FT] - V[t] * dyA[t, FT]) for t in 1:T)

    ## Stripping section
    itr_strip = ExaModels.convert_array(collect(Iterators.product(1:T, (FT + 1):NT)), backend)
    constraint(c, dxA[t, i] - (1 / At) * (L2[t] * (yA[t, i - 1] - xA[t, i]) - V[t] * dyA[t, i]) for (t, i) in itr_strip)

    ## Reboiler
    constraint(c, dxA[t, NT + 1] - (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT + 1] - V[t] * yA[t, NT + 1]) for t in 1:T)

    ## Flow relationships (still needed as these are decision variables)
    constraint(c, V[t] - u[t] * D - D for t in 0:T)
    constraint(c, L2[t] - u[t] * D - F for t in 0:T)

    ## VLE
    itr_vle = ExaModels.convert_array(collect(Iterators.product(0:T, 0:(NT + 1))), backend)
    constraint(c, yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr_vle)

    return ExaModel(c)
end

#-

# ## Running the Models
#
# Let's compare all three model variants and verify they converge to the same solution.

using ExaModels, NLPModelsIpopt

T_val = 10

# 1. Original model (no subexpressions)
m_orig = distillation_column_model(T_val)
result_orig = ipopt(m_orig; print_level = 0)

#-

# 2. Lifted subexpressions
m_lifted = distillation_column_model_with_subexpr(T_val)
result_lifted = ipopt(m_lifted; print_level = 0)

#-

# 3. Reduced subexpressions
m_reduced = distillation_column_model_reduced(T_val)
result_reduced = ipopt(m_reduced; print_level = 0)

#-

# ## Comparison Results
#
# Print comparison table:

println("="^70)
println("Distillation Column Model Comparison (T=$T_val)")
println("="^70)
println()
println("| Model                  | Variables | Constraints | Iterations | Objective |")
println("|------------------------|-----------|-------------|------------|-----------|")
println("| Original (no subexpr)  | $(lpad(m_orig.meta.nvar, 9)) | $(lpad(m_orig.meta.ncon, 11)) | $(lpad(result_orig.iter, 10)) | $(round(result_orig.objective, digits = 6)) |")
println("| Lifted                 | $(lpad(m_lifted.meta.nvar, 9)) | $(lpad(m_lifted.meta.ncon, 11)) | $(lpad(result_lifted.iter, 10)) | $(round(result_lifted.objective, digits = 6)) |")
println("| Reduced                | $(lpad(m_reduced.meta.nvar, 9)) | $(lpad(m_reduced.meta.ncon, 11)) | $(lpad(result_reduced.iter, 10)) | $(round(result_reduced.objective, digits = 6)) |")
println()

#-

# Verify all models converge to the same solution:

all_same = all(
    [
        isapprox(result_orig.objective, result_lifted.objective, rtol = 1.0e-4),
        isapprox(result_orig.objective, result_reduced.objective, rtol = 1.0e-4),
    ]
)

println("All models converge to same objective: $all_same")

# Key observations:
# - Lifted subexpressions ADD variables/constraints (auxiliary vars + defining constraints)
# - Reduced subexpressions maintain the original problem size (no aux vars)
# - All models reach the same optimal solution
