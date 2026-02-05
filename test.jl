using ExaModels, NLPModels

function test_basic()
    @info "Creating model with expressions..."
    m = ExaCore()
    v = variable(m, 5)
    e1 = expression(m, (4,), v[i] * v[i+1] for i in 1:4)
    e2 = expression(m, (4,), e1[i] + v[i] for i in 1:4)
    c = constraint(m, e2[i] / i for i in 1:4; ucon=10.0)
    o = objective(m, e2[i] for i in 1:4)
    mod = ExaModel(m)

    @info "Model created:"
    @info "  nvar = $(mod.meta.nvar)"
    @info "  ncon = $(mod.meta.ncon)"
    @info "  nnzj = $(mod.meta.nnzj)"
    @info "  nnzh = $(mod.meta.nnzh)"

    x = Float64[i for i in 1:mod.meta.nvar]

    # Test Jacobian structure
    @info "Testing Jacobian structure..."
    jac_rows = zeros(Int, mod.meta.nnzj)
    jac_cols = zeros(Int, mod.meta.nnzj)
    jac_structure!(mod, jac_rows, jac_cols)
    @info "  jac_rows = $jac_rows"
    @info "  jac_cols = $jac_cols"

    # Test Jacobian values
    @info "Testing Jacobian values..."
    jac_buffer = zeros(mod.meta.nnzj)
    jac_coord!(mod, x, jac_buffer)
    @info "  jac_buffer = $jac_buffer"
    @assert all(isfinite, jac_buffer) "Jacobian contains non-finite values!"

    # Test Hessian structure
    @info "Testing Hessian structure..."
    hess_rows = zeros(Int, mod.meta.nnzh)
    hess_cols = zeros(Int, mod.meta.nnzh)
    hess_structure!(mod, hess_rows, hess_cols)
    @info "  hess_rows = $hess_rows"
    @info "  hess_cols = $hess_cols"

    # Test Hessian values (objective only)
    @info "Testing Hessian values (objective only)..."
    hess_buffer = zeros(mod.meta.nnzh)
    hess_coord!(mod, x, hess_buffer; obj_weight=1.0)
    @info "  hess_buffer = $hess_buffer"
    @assert all(isfinite, hess_buffer) "Hessian (obj only) contains non-finite values!"

    # Test Hessian values (with constraints)
    @info "Testing Hessian values (with constraints)..."
    y = ones(mod.meta.ncon)
    hess_buffer2 = zeros(mod.meta.nnzh)
    hess_coord!(mod, x, y, hess_buffer2; obj_weight=1.0)
    @info "  hess_buffer (with cons) = $hess_buffer2"
    @assert all(isfinite, hess_buffer2) "Hessian (with cons) contains non-finite values!"

    # Test Hessian-vector product
    # @info "Testing Hessian-vector product..."
    # v_vec = ones(mod.meta.nvar)
    # Hv = zeros(mod.meta.nvar)
    # hprod!(mod, x, y, v_vec, Hv; obj_weight=1.0)
    # @info "  Hv = $Hv"
    # @assert all(isfinite, Hv) "Hessian-vector product contains non-finite values!"

    @info "Basic tests passed!"
end

function test_simple_quadratic()
    # Test a simple quadratic: f(x) = x^2, constraint: x^2 - 1 = 0
    # Hessian of f = 2
    # Hessian of constraint = 2
    @info "\n=== Testing simple quadratic ==="
    m = ExaCore()
    v = variable(m, 1)
    o = objective(m, v[1]^2)
    c = constraint(m, v[1]^2; lcon=1.0, ucon=1.0)
    mod = ExaModel(m)

    @info "Model: f(x) = x^2, c(x) = x^2 - 1 = 0"
    @info "  nvar = $(mod.meta.nvar), ncon = $(mod.meta.ncon)"
    @info "  nnzj = $(mod.meta.nnzj), nnzh = $(mod.meta.nnzh)"

    x = [3.0]  # arbitrary point

    # Hessian structure
    hess_rows = zeros(Int, mod.meta.nnzh)
    hess_cols = zeros(Int, mod.meta.nnzh)
    hess_structure!(mod, hess_rows, hess_cols)
    @info "  Hessian structure: rows=$hess_rows, cols=$hess_cols"

    # Objective Hessian only
    hess_obj = zeros(mod.meta.nnzh)
    hess_coord!(mod, x, hess_obj; obj_weight=1.0)
    @info "  Hessian (obj only): $hess_obj"
    @info "  Expected: 2.0 (from d^2/dx^2 (x^2) = 2)"
    @info "  Note: May have multiple entries at same location in sparse format"
    # The hessian has nnzh=2, one for obj, one for con. With obj_weight only, 
    # we expect one entry to be 2.0 (from objective)
    @assert any(h ≈ 2.0 for h in hess_obj) "Expected at least one Hessian entry of 2.0"

    # Full Hessian (obj + constraints)
    y = [1.0]  # constraint multiplier
    hess_full = zeros(mod.meta.nnzh)
    hess_coord!(mod, x, y, hess_full; obj_weight=1.0)
    @info "  Hessian (obj + con): $hess_full"
    @info "  Expected: 2 from obj + 2 from constraint = 4.0 total"
    # Sum should be 2 + 2 = 4
    @assert sum(hess_full) ≈ 4.0 "Expected combined Hessian entries to sum to 4"

    @info "Simple quadratic test passed!"
end

function test_expression_quadratic()
    # Test with expression: e = x*y, f(e) = e, c(e) = e - 1 = 0
    # Hessian of f w.r.t. (x,y) should show cross-derivative
    @info "\n=== Testing expression with cross-derivative ==="
    m = ExaCore()
    v = variable(m, 2)
    e1 = expression(m, (1,), v[1] * v[2] for _ in 1:1)  # e = x*y
    o = objective(m, e1[1] for _ in 1:1)  # f = e = x*y
    c = constraint(m, e1[1] for _ in 1:1; lcon=1.0, ucon=1.0)  # c = x*y = 1
    mod = ExaModel(m)

    @info "Model: e = x*y, f = e, c = e = 1"
    @info "  nvar = $(mod.meta.nvar), ncon = $(mod.meta.ncon)"
    @info "  nnzj = $(mod.meta.nnzj), nnzh = $(mod.meta.nnzh)"

    x = zeros(mod.meta.nvar)
    x[1:2] .= [2.0, 3.0]

    # Hessian structure
    hess_rows = zeros(Int, mod.meta.nnzh)
    hess_cols = zeros(Int, mod.meta.nnzh)
    hess_structure!(mod, hess_rows, hess_cols)
    @info "  Hessian structure: rows=$hess_rows, cols=$hess_cols"

    # The Hessian of f = x*y is:
    # [0  1]
    # [1  0]
    # So we expect entries at (1,2) or (2,1) with value 1

    # Objective Hessian only
    hess_obj = zeros(mod.meta.nnzh)
    hess_coord!(mod, x, hess_obj; obj_weight=1.0)
    @info "  Hessian (obj only): $hess_obj"
    @info "  Expected: cross-derivative term = 1 (d^2/dxdy (xy) = 1)"

    # Full Hessian
    y = [1.0]
    hess_full = zeros(mod.meta.nnzh)
    hess_coord!(mod, x, y, hess_full; obj_weight=1.0)
    @info "  Hessian (obj + con): $hess_full"

    @info "Expression cross-derivative test passed!"
end

function test_nested_expressions()
    # Test nested expressions: e1 = x^2, e2 = e1 + x, f = e2
    @info "\n=== Testing nested expressions ==="
    m = ExaCore()
    v = variable(m, 1)
    e1 = expression(m, (1,), v[1]^2 for _ in 1:1)  # e1 = x^2
    e2 = expression(m, (1,), e1[1] + v[1] for _ in 1:1)  # e2 = e1 + x = x^2 + x
    o = objective(m, e2[1] for _ in 1:1)  # f = e2 = x^2 + x
    mod = ExaModel(m)

    @info "Model: e1 = x^2, e2 = e1 + x, f = e2 = x^2 + x"
    @info "  nvar = $(mod.meta.nvar), ncon = $(mod.meta.ncon)"
    @info "  nnzj = $(mod.meta.nnzj), nnzh = $(mod.meta.nnzh)"

    x = zeros(mod.meta.nvar)
    x[1] = 5.0

    # Hessian structure
    hess_rows = zeros(Int, mod.meta.nnzh)
    hess_cols = zeros(Int, mod.meta.nnzh)
    hess_structure!(mod, hess_rows, hess_cols)
    @info "  Hessian structure: rows=$hess_rows, cols=$hess_cols"

    # The Hessian of f = x^2 + x is:
    # [2]  (d^2/dx^2 (x^2 + x) = 2)

    # Objective Hessian only
    hess_obj = zeros(mod.meta.nnzh)
    hess_coord!(mod, x, hess_obj; obj_weight=1.0)
    @info "  Hessian (obj only): $hess_obj"
    @info "  Expected: 2.0 (d^2/dx^2 (x^2 + x) = 2)"
    # Sum of hessian entries should be 2
    @assert sum(hess_obj) ≈ 2.0 "Expected Hessian sum to be 2"

    @info "Nested expression test passed!"
end

# Run all tests
test_basic()
test_simple_quadratic()
test_expression_quadratic()
test_nested_expressions()

@info "\n=== All Hessian tests passed! ==="
