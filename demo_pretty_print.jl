using ExaModels

println("=" ^ 60)
println("  ExaModels Node Pretty Print Demo")
println("=" ^ 60)

# ── Symbolic Expression Nodes ──────────────────────────────

println("\n── Symbolic Expression Nodes ─────────────────────────\n")

# Null nodes
n0 = ExaModels.Null()
n1 = ExaModels.Null(3.14)
println("Null():       ", n0)
println("Null(3.14):   ", n1)

# Variable nodes
v1 = ExaModels.Var(1)
v2 = ExaModels.Var(5)
println("Var(1):       ", v1)
println("Var(5):       ", v2)

# Parameter nodes
p1 = ExaModels.ParameterNode(2)
println("ParameterNode(2): ", p1)

# Par nodes
ps = ExaModels.ParSource()
pi = ExaModels.ParIndexed(ps, :cost)
pi2 = ExaModels.ParIndexed(pi, :sub)
println("ParSource():       ", ps)
println("ParIndexed(:cost): ", pi)
println("Nested ParIndexed: ", pi2)

# Unary operation nodes
sin_node = ExaModels.Node1(sin, v1)
cos_node = ExaModels.Node1(cos, v2)
neg_node = ExaModels.Node1(-, v1)
println("sin(x[1]):    ", sin_node)
println("cos(x[5]):    ", cos_node)
println("-(x[1]):      ", neg_node)

# Binary operation nodes
add_node = ExaModels.Node2(+, v1, v2)
mul_node = ExaModels.Node2(*, v1, n1)
pow_node = ExaModels.Node2(^, v2, ExaModels.Null(2))
println("x[1] + x[5]:    ", add_node)
println("x[1] * 3.14:    ", mul_node)
println("x[5] ^ 2:       ", pow_node)

# Nested expressions
nested = ExaModels.Node2(+, sin_node, mul_node)
println("sin(x[1]) + x[1]*3.14: ", nested)

deep = ExaModels.Node2(^, nested, ExaModels.Null(2))
println("(...) ^ 2:             ", deep)

par_expr = ExaModels.Node2(+, sin_node, pi)
println("sin(x[1]) + p.cost:    ", par_expr)

# ── Detailed REPL display ─────────────────────────────────

println("\n── Detailed display (text/plain) ─────────────────────\n")
println("display(sin(x[1]) + x[1]*3.14):")
display(nested)
println("\n")

# ── First-Order Adjoint Nodes ─────────────────────────────

println("\n── First-Order Adjoint Nodes ─────────────────────────\n")

an = ExaModels.AdjointNull(1.5)
println("AdjointNull:    ", an)

av = ExaModels.AdjointNodeVar(3, 0.7)
println("AdjointNodeVar: ", av)

a1 = ExaModels.AdjointNode1(sin, 0.8415, 0.5403, an)
println("AdjointNode1:   ", a1)

a2 = ExaModels.AdjointNode2(+, 2.2, 1.0, 1.0, an, av)
println("AdjointNode2:   ", a2)

println("\ndisplay(AdjointNode1):")
display(a1)
println("\n")

println("display(AdjointNode2):")
display(a2)
println("\n")

# ── Second-Order Adjoint Nodes ────────────────────────────

println("\n── Second-Order Adjoint Nodes ────────────────────────\n")

sn = ExaModels.SecondAdjointNull(2.0)
println("SecondAdjointNull:    ", sn)

sv = ExaModels.SecondAdjointNodeVar(1, 0.5)
println("SecondAdjointNodeVar: ", sv)

s1 = ExaModels.SecondAdjointNode1(sin, 0.8415, 0.5403, -0.8415, sn)
println("SecondAdjointNode1:   ", s1)

s2 = ExaModels.SecondAdjointNode2(*, 3.0, 1.5, 2.0, 0.0, 1.0, 0.0, sn, sv)
println("SecondAdjointNode2:   ", s2)

println("\ndisplay(SecondAdjointNode1):")
display(s1)
println("\n")

println("display(SecondAdjointNode2):")
display(s2)
println("\n")

# ══════════════════════════════════════════════════════════
# Using pretty print through the ExaModels API
# ══════════════════════════════════════════════════════════

println("=" ^ 60)
println("  ExaModels API - Objective & Constraint Pretty Print")
println("=" ^ 60)

# ── Objective ─────────────────────────────────────────────

println("\n── Objective (x[i]^2 for i=1:5) ─────────────────────\n")

c = ExaCore()
c, x = add_var(c, 5)
c, obj = add_obj(c, x[i]^2 for i=1:5)
display(obj)

# ── Constraint ────────────────────────────────────────────

println("\n── Constraint (sin(x[i]) + x[i+1] for i=1:4) ───────\n")

c, con = add_con(c, sin(x[i]) + x[i+1] for i=1:4; lcon = -1.0, ucon = 1.0)
display(con)

# ── Constraint Augmentation ───────────────────────────────

println("\n── ConstraintAug (i => cos(x[i]) for i=2:3) ────────\n")

c, con2 = add_con!(c, con, i => cos(x[i]) for i=2:3)
display(con2)

# ── Expression (subexpression) ────────────────────────────

println("\n── Expression (x[i]^3 + 1.0 for i in 1:5) ──────────\n")

c, s = add_expr(c, x[i]^3 + 1.0 for i in 1:5)
display(s)

# ── Objective using subexpression ─────────────────────────

println("\n── Objective using subexpression (s[i]*x[i] for i=1:5)\n")

c, obj2 = add_obj(c, s[i] * x[i] for i=1:5)
display(obj2)

# ── Full model summary ───────────────────────────────────

println("\n── Full ExaModel ────────────────────────────────────\n")

m = ExaModel(c)
display(m)

println("\n\n", "=" ^ 60)
println("  Done!")
println("=" ^ 60)
