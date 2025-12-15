# # [Example: Distillation Column](@id distillation)
#
# This example demonstrates the use of `subexpr` to simplify complex models.
# We show five versions of a distillation column model comparing lifted vs reduced subexpressions.
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
# | **Nesting** | Robust | Works with fixed indexing bug |
# | **Convergence** | May need more iterations | Same as original |
#
# ## Performance Comparison (T=10)
#
# | Model | Variables | Constraints | Iterations |
# |-------|-----------|-------------|------------|
# | Original (no subexpr) | 737 | 726 | 7 |
# | Moderate lifted | 1398 | 1387 | 7 |
# | Aggressive lifted | 1409 | 1398 | 14 |
# | Moderate reduced | 737 | 726 | 7 |
# | Aggressive reduced | 715 | 704 | 7 |
#
# All five models converge to the same optimal solution. Reduced subexpressions maintain
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
    xA0s = ExaModels.convert_array([(i, 0.5) for i = 0:(NT+1)], backend)

    itr0 = ExaModels.convert_array(collect(Iterators.product(1:T, 1:(FT-1))), backend)
    itr1 = ExaModels.convert_array(collect(Iterators.product(1:T, (FT+1):NT)), backend)
    itr2 = ExaModels.convert_array(collect(Iterators.product(0:T, 0:(NT+1))), backend)

    c = ExaCore(backend)

    xA = variable(c, 0:T, 0:(NT+1); start = 0.5)
    yA = variable(c, 0:T, 0:(NT+1); start = 0.5)
    u = variable(c, 0:T; start = 1.0)
    V = variable(c, 0:T; start = 1.0)
    L2 = variable(c, 0:T; start = 1.0)

    objective(c, (yA[t, 1] - ybar)^2 for t = 0:T)
    objective(c, (u[t] - ubar)^2 for t = 0:T)

    constraint(c, xA[0, i] - xA0 for (i, xA0) in xA0s)
    constraint(
        c,
        (xA[t, 0] - xA[t-1, 0]) / dt - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t = 1:T
    )
    constraint(
        c,
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (u[t] * D * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) for
        (t, i) in itr0
    )
    constraint(
        c,
        (xA[t, FT] - xA[t-1, FT]) / dt -
        (1 / At) * (
            F * xAf + u[t] * D * xA[t, FT-1] - L2[t] * xA[t, FT] -
            V[t] * (yA[t, FT] - yA[t, FT+1])
        ) for t = 1:T
    )
    constraint(
        c,
        (xA[t, i] - xA[t-1, i]) / dt -
        (1 / At) * (L2[t] * (yA[t, i-1] - xA[t, i]) - V[t] * (yA[t, i] - yA[t, i+1])) for
        (t, i) in itr1
    )
    constraint(
        c,
        (xA[t, NT+1] - xA[t-1, NT+1]) / dt -
        (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT+1] - V[t] * yA[t, NT+1]) for
        t = 1:T
    )
    constraint(c, V[t] - u[t] * D - D for t = 0:T)
    constraint(c, L2[t] - u[t] * D - F for t = 0:T)
    constraint(
        c,
        yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr2
    )

    return ExaModel(c)
end

# ## Model with Subexpressions (Moderate)
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

# ## Compact Model with Aggressive Subexpressions
#
# This version uses subexpressions aggressively to maximize code clarity:
# - `V[t]` and `L2[t]` become subexpressions of `u[t]` (eliminates flow relationship constraints)
# - `dxA[t,i]` captures time derivatives
# - `VdyA[t,i]` captures vapor transport terms
# - `uD[t]` captures the reflux flow term
#
# **Note:** This model takes ~2x more iterations to converge (14 vs 7) due to the
# nested subexpression structure. All models reach the same optimal solution.

function distillation_column_model_compact(T = 3; backend = nothing)

    NT, FT = 30, 17
    Ac, At, Ar = 0.5, 0.25, 1.0
    D, F = 0.2, 0.4
    ybar, ubar, alpha = 0.8958, 2.0, 1.6
    dt, xAf = 10 / T, 0.5

    c = ExaCore(backend)

    ## Primary decision variables (V and L2 eliminated!)
    xA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    yA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    u = variable(c, 0:T; start = 1.0)

    ## Flow subexpressions (replaces V and L2 decision variables + constraints)
    V = subexpr(c, u[t] * D + D for t in 0:T)      ## Vapor flow = D(u+1)
    L2 = subexpr(c, u[t] * D + F for t in 0:T)     ## Liquid flow below feed

    ## Time derivative of liquid composition
    dxA = subexpr(c, (xA[t, i] - xA[t - 1, i]) / dt for t in 1:T, i in 0:(NT + 1))

    ## Vapor transport term V*(yA[t,i] - yA[t,i+1])
    VdyA = subexpr(c, V[t] * (yA[t, i] - yA[t, i + 1]) for t in 0:T, i in 0:NT)

    ## Reflux term u*D
    uD = subexpr(c, u[t] * D for t in 0:T)

    ## Objectives
    objective(c, (yA[t, 1] - ybar)^2 for t in 0:T)
    objective(c, (u[t] - ubar)^2 for t in 0:T)

    ## Initial conditions
    xA0s = ExaModels.convert_array([(i, 0.5) for i in 0:(NT + 1)], backend)
    constraint(c, xA[0, i] - xA0 for (i, xA0) in xA0s)

    ## Condenser (i=0): dxA/dt = (1/Ac)(yA[1] - xA[0])
    constraint(c, dxA[t, 0] - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t in 1:T)

    ## Rectifying section (i=1 to FT-1): dxA/dt = (1/At)(uD*(yA[i-1]-xA[i]) - V*dyA)
    itr_rect = ExaModels.convert_array(collect(Iterators.product(1:T, 1:(FT - 1))), backend)
    constraint(c, dxA[t, i] - (1 / At) * (uD[t] * (yA[t, i - 1] - xA[t, i]) - VdyA[t, i]) for (t, i) in itr_rect)

    ## Feed tray (i=FT)
    constraint(c, dxA[t, FT] - (1 / At) * (F * xAf + uD[t] * xA[t, FT - 1] - L2[t] * xA[t, FT] - VdyA[t, FT]) for t in 1:T)

    ## Stripping section (i=FT+1 to NT)
    itr_strip = ExaModels.convert_array(collect(Iterators.product(1:T, (FT + 1):NT)), backend)
    constraint(c, dxA[t, i] - (1 / At) * (L2[t] * (yA[t, i - 1] - xA[t, i]) - VdyA[t, i]) for (t, i) in itr_strip)

    ## Reboiler (i=NT+1)
    constraint(c, dxA[t, NT + 1] - (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT + 1] - V[t] * yA[t, NT + 1]) for t in 1:T)

    ## VLE: yA(1-xA) = α*xA(1-yA)
    itr_vle = ExaModels.convert_array(collect(Iterators.product(0:T, 0:(NT + 1))), backend)
    constraint(c, yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr_vle)

    return ExaModel(c)
end

# ## Model with Reduced Subexpressions (Moderate)
#
# Uses **reduced** subexpressions for time derivatives and vapor differences.
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

# ## Compact Model with Aggressive Reduced Subexpressions
#
# This version uses **reduced** subexpressions aggressively:
# - `V[t]` and `L2[t]` become reduced subexpressions of `u[t]` (eliminates V and L2 as decision variables!)
# - `dxA[t,i]` captures time derivatives (reduced)
# - `VdyA[t,i]` captures vapor transport terms (reduced, nests V)
# - `uD[t]` captures the reflux flow term (reduced)
#
# This results in FEWER variables than the original (no V, L2 decision variables)
# while keeping the same convergence behavior.

function distillation_column_model_compact_reduced(T = 3; backend = nothing)

    NT, FT = 30, 17
    Ac, At, Ar = 0.5, 0.25, 1.0
    D, F = 0.2, 0.4
    ybar, ubar, alpha = 0.8958, 2.0, 1.6
    dt, xAf = 10 / T, 0.5

    c = ExaCore(backend)

    ## Primary decision variables (V and L2 eliminated via reduced subexpressions!)
    xA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    yA = variable(c, 0:T, 0:(NT + 1); start = 0.5)
    u = variable(c, 0:T; start = 1.0)

    ## Reduced flow subexpressions (replaces V and L2 decision variables + their constraints!)
    V = subexpr(c, u[t] * D + D for t in 0:T; reduced = true)      ## Vapor flow = D(u+1)
    L2 = subexpr(c, u[t] * D + F for t in 0:T; reduced = true)     ## Liquid flow below feed

    ## Reduced time derivative
    dxA = subexpr(c, (xA[t, i] - xA[t - 1, i]) / dt for t in 1:T, i in 0:(NT + 1); reduced = true)

    ## Reduced vapor transport term V*(yA[t,i] - yA[t,i+1]) - nests V!
    VdyA = subexpr(c, V[t] * (yA[t, i] - yA[t, i + 1]) for t in 0:T, i in 0:NT; reduced = true)

    ## Reduced reflux term u*D
    uD = subexpr(c, u[t] * D for t in 0:T; reduced = true)

    ## Objectives
    objective(c, (yA[t, 1] - ybar)^2 for t in 0:T)
    objective(c, (u[t] - ubar)^2 for t in 0:T)

    ## Initial conditions
    xA0s = ExaModels.convert_array([(i, 0.5) for i in 0:(NT + 1)], backend)
    constraint(c, xA[0, i] - xA0 for (i, xA0) in xA0s)

    ## Condenser (i=0)
    constraint(c, dxA[t, 0] - (1 / Ac) * (yA[t, 1] - xA[t, 0]) for t in 1:T)

    ## Rectifying section (i=1 to FT-1)
    itr_rect = ExaModels.convert_array(collect(Iterators.product(1:T, 1:(FT - 1))), backend)
    constraint(c, dxA[t, i] - (1 / At) * (uD[t] * (yA[t, i - 1] - xA[t, i]) - VdyA[t, i]) for (t, i) in itr_rect)

    ## Feed tray (i=FT)
    constraint(c, dxA[t, FT] - (1 / At) * (F * xAf + uD[t] * xA[t, FT - 1] - L2[t] * xA[t, FT] - VdyA[t, FT]) for t in 1:T)

    ## Stripping section (i=FT+1 to NT)
    itr_strip = ExaModels.convert_array(collect(Iterators.product(1:T, (FT + 1):NT)), backend)
    constraint(c, dxA[t, i] - (1 / At) * (L2[t] * (yA[t, i - 1] - xA[t, i]) - VdyA[t, i]) for (t, i) in itr_strip)

    ## Reboiler (i=NT+1)
    constraint(c, dxA[t, NT + 1] - (1 / Ar) * (L2[t] * xA[t, NT] - (F - D) * xA[t, NT + 1] - V[t] * yA[t, NT + 1]) for t in 1:T)

    ## VLE: yA(1-xA) = α*xA(1-yA)
    itr_vle = ExaModels.convert_array(collect(Iterators.product(0:T, 0:(NT + 1))), backend)
    constraint(c, yA[t, i] * (1 - xA[t, i]) - alpha * xA[t, i] * (1 - yA[t, i]) for (t, i) in itr_vle)

    return ExaModel(c)
end

#-

# ## Running the Models
#
# Let's compare all five model variants and verify they converge to the same solution.

using ExaModels, NLPModelsIpopt

T_val = 10

# 1. Original model (no subexpressions)
m_orig = distillation_column_model(T_val)
result_orig = ipopt(m_orig; print_level = 0)

#-

# 2. Moderate lifted subexpressions
m_lifted_mod = distillation_column_model_with_subexpr(T_val)
result_lifted_mod = ipopt(m_lifted_mod; print_level = 0)

#-

# 3. Aggressive lifted subexpressions
m_lifted_agg = distillation_column_model_compact(T_val)
result_lifted_agg = ipopt(m_lifted_agg; print_level = 0)

#-

# 4. Moderate reduced subexpressions
m_reduced_mod = distillation_column_model_reduced(T_val)
result_reduced_mod = ipopt(m_reduced_mod; print_level = 0)

#-

# 5. Aggressive reduced subexpressions
m_reduced_agg = distillation_column_model_compact_reduced(T_val)
result_reduced_agg = ipopt(m_reduced_agg; print_level = 0)

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
println("| Moderate lifted        | $(lpad(m_lifted_mod.meta.nvar, 9)) | $(lpad(m_lifted_mod.meta.ncon, 11)) | $(lpad(result_lifted_mod.iter, 10)) | $(round(result_lifted_mod.objective, digits = 6)) |")
println("| Aggressive lifted      | $(lpad(m_lifted_agg.meta.nvar, 9)) | $(lpad(m_lifted_agg.meta.ncon, 11)) | $(lpad(result_lifted_agg.iter, 10)) | $(round(result_lifted_agg.objective, digits = 6)) |")
println("| Moderate reduced       | $(lpad(m_reduced_mod.meta.nvar, 9)) | $(lpad(m_reduced_mod.meta.ncon, 11)) | $(lpad(result_reduced_mod.iter, 10)) | $(round(result_reduced_mod.objective, digits = 6)) |")
println("| Aggressive reduced     | $(lpad(m_reduced_agg.meta.nvar, 9)) | $(lpad(m_reduced_agg.meta.ncon, 11)) | $(lpad(result_reduced_agg.iter, 10)) | $(round(result_reduced_agg.objective, digits = 6)) |")
println()

#-

# Verify all models converge to the same solution:

all_same = all(
    [
        isapprox(result_orig.objective, result_lifted_mod.objective, rtol = 1.0e-4),
        isapprox(result_orig.objective, result_lifted_agg.objective, rtol = 1.0e-4),
        isapprox(result_orig.objective, result_reduced_mod.objective, rtol = 1.0e-4),
        isapprox(result_orig.objective, result_reduced_agg.objective, rtol = 1.0e-4),
    ]
)

println("All models converge to same objective: $all_same")

# Key observations:
# - Lifted subexpressions ADD variables/constraints (auxiliary vars + defining constraints)
# - Reduced subexpressions maintain or REDUCE problem size (no aux vars)
# - Aggressive reduced eliminates V and L2 as decision variables entirely!
# - All models reach the same optimal solution
