using ExaModels

# Build a complicated model that will trigger a MethodError
c = ExaCore()
@var(c, x, 10)
@var(c, y, 5)

# This creates a deeply nested node type:
#   Node2{+, Node1{sin, Var{...}}, Node2{*, Var{...}, Node1{cos, Var{...}}}}
# Try to use a string (invalid) in the expression to trigger a MethodError
# that will print the full type in the stacktrace

println("=" ^ 70)
println("  Stacktrace Demo — deeply nested expression types")
println("=" ^ 70)

println("\n── 1. What `typeof` looks like for a complex node ──────────\n")

# Build: sin(x[i]) + x[i+1] * cos(x[i+2]) - exp(y[j])
# via the generator:
gen_f = let x=x, y=y
    p -> sin(x[p.i]) + x[p.i+1] * cos(x[p.i+2]) - exp(y[p.j])
end
node = gen_f(ExaModels.ParSource())

println("Expression: ", node)
println()
println("typeof(node):")
println("  ", typeof(node))
println()
println("Full type (formatted):")
show(stdout, MIME"text/plain"(), node)
println("\n")

println("\n── 2. MethodError stacktrace ─────────────────────────────\n")

# Trigger a MethodError by trying to call an unregistered function
struct BadFunc end
try
    ExaModels.Node1(BadFunc(), ExaModels.Var(1))(1, [1.0], nothing)
catch e
    showerror(stdout, e)
    println()
    Base.show_backtrace(stdout, catch_backtrace())
end

println("\n\n── 3. TypeError from bad generator ───────────────────────\n")

# Trigger error with complex types in the message
try
    c2 = ExaCore()
    c2, x2 = add_var(c2, 10)
    # Use an expression that references out of bounds
    c2, obj = add_obj(c2,
        sin(x2[i]) * cos(x2[i+1]) + exp(x2[i+2]) * x2[i+3]^2 + tanh(x2[i+4])
        for i=1:10  # i+4 can reach 14, but x2 only has 10 elements
    )
    m = ExaModel(c2)
    # Force evaluation
    using NLPModels
    obj(m, ones(10))
catch e
    showerror(stdout, e)
    println()
    Base.show_backtrace(stdout, catch_backtrace())
end

println("\n\n── 4. What SIMDFunction.f type looks like ────────────────\n")

c3 = ExaCore()
c3, x3 = add_var(c3, 10)
c3, obj3 = add_obj(c3,
    sin(x3[i]) * cos(x3[i+1]) + x3[i+2]^3 - exp(x3[i]) / x3[i+1]
    for i=1:5
)
println("Objective expression: ", obj3)
println()
println("typeof(obj3.f.f):")
println("  ", typeof(obj3.f.f))
println()
println("But show(obj3) is clean:")
display(obj3)

println("\n\n", "=" ^ 70)
println("  Done!")
println("=" ^ 70)
