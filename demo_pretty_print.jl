using ExaModels

println("=" ^ 60)
println("  ExaModels Pretty Print Demo")
println("=" ^ 60)

# ══════════════════════════════════════════════════════════
# 1. Named variables
# ══════════════════════════════════════════════════════════

println("\n── Named Variables ───────────────────────────────────\n")

c = ExaCore()

# Using @var macro — name flows into display
@var(c, x, 5)
println("@var(c, x, 5):")
display(x)

@var(c, y, 3)
println("@var(c, y, 3):")
display(y)

# Unnamed variable defaults to 'x'
c, z = add_var(c, 4)
println("add_var(c, 4):")
display(z)

# Named via keyword
c, w = add_var(c, 2; name = Val(:w))
println("add_var(c, 2; name = Val(:w)):")
display(w)

# ══════════════════════════════════════════════════════════
# 2. Objectives — expression tree is shown
# ══════════════════════════════════════════════════════════

println("\n── Objectives ────────────────────────────────────────\n")

println("add_obj(c, x[i]^2 for i=1:5):")
c, obj1 = add_obj(c, x[i]^2 for i=1:5)
display(obj1)

println("\nadd_obj(c, sin(x[i]) * y[j] for (i,j) in [(1,1),(2,2),(3,3)]):")
c, obj2 = add_obj(c, sin(x[i]) * y[j] for (i,j) in [(1,1),(2,2),(3,3)])
display(obj2)

# ══════════════════════════════════════════════════════════
# 3. Constraints — expression tree is shown
# ══════════════════════════════════════════════════════════

println("\n── Constraints ───────────────────────────────────────\n")

println("add_con(c, sin(x[i]) + x[i+1] for i=1:4):")
c, con1 = add_con(c, sin(x[i]) + x[i+1] for i=1:4; lcon = -1.0, ucon = 1.0)
display(con1)

println("\nadd_con(c, x[i] - y[j] for (i,j) in [(1,1),(2,2)]):")
c, con2 = add_con(c, x[i] - y[j] for (i,j) in [(1,1),(2,2)]; lcon = 0.0, ucon = 0.0)
display(con2)

# ══════════════════════════════════════════════════════════
# 4. Constraint augmentation
# ══════════════════════════════════════════════════════════

println("\n── Constraint Augmentation ───────────────────────────\n")

println("add_con!(c, con1, i => cos(x[i]) for i=2:3):")
c, con3 = add_con!(c, con1, i => cos(x[i]) for i=2:3)
display(con3)

# ══════════════════════════════════════════════════════════
# 5. Subexpressions
# ══════════════════════════════════════════════════════════

println("\n── Subexpressions ────────────────────────────────────\n")

println("add_expr(c, x[i]^3 + 1.0 for i in 1:5):")
c, s = add_expr(c, x[i]^3 + 1.0 for i in 1:5)
display(s)

# ══════════════════════════════════════════════════════════
# 6. Low-level nodes (direct construction)
# ══════════════════════════════════════════════════════════

println("\n── Raw Nodes ─────────────────────────────────────────\n")

v = ExaModels.Var(1)
println("Var(1):         ", v)
println("sin(x[1]):      ", sin(v))
println("x[1] + x[5]:    ", ExaModels.Node2(+, v, ExaModels.Var(5)))
println("x[1] ^ 2:       ", v ^ 2)

# Show identity simplification
p = ExaModels.ParSource()
println("\nParSource (iteration variable):")
println("  i:            ", p)
println("  i.cost:       ", ExaModels.ParIndexed(p, :cost))
println("  i.cost.sub:   ", ExaModels.ParIndexed(ExaModels.ParIndexed(p, :cost), :sub))

# ══════════════════════════════════════════════════════════
# 7. Full model
# ══════════════════════════════════════════════════════════

println("\n── Full Model ────────────────────────────────────────\n")
m = ExaModel(c)
display(m)

println("\n\n", "=" ^ 60)
println("  Done!")
println("=" ^ 60)
