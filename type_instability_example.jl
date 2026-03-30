"""
Minimal Working Example: Type Instability in grpass (sparsity detection)

The issue: when the same variable appears 16+ times in a single expression body,
Julia's type inference "widens" the accumulated index tuple from NTuple{N,Int64}
to Tuple{Int64,...,Vararg{Int64}}, making the grpass return type non-concrete.

This happens during sparsity detection in _simdfunction (simdfunction.jl):
    a1, y1 = ExaModels.grpass(d, nothing, nothing, NaNSource{T}(), ((),()), T(NaN))
If `a1` has Vararg type, then Compressor(a1) has a non-concrete type,
and the resulting SIMDFunction is type-unstable.
"""

using ExaModels
using InteractiveUtils

# ---------------------------------------------------------------------------
# Step 1: Build expression graphs with same variable repeated N times
# ---------------------------------------------------------------------------
x_src = ExaModels.AdjointNodeSource(nothing)
v = x_src[1]  # same variable used every time

# (v+v) = 2 reads
node_2  = ExaModels.AdjointNode2(+, 0.0, 1.0, 1.0, v, v)
# (v+v)+(v+v) = 4 reads
node_4  = ExaModels.AdjointNode2(+, 0.0, 1.0, 1.0, node_2, node_2)
# ((v+v)+(v+v)) + ((v+v)+(v+v)) = 8 reads
node_8  = ExaModels.AdjointNode2(+, 0.0, 1.0, 1.0, node_4, node_4)
# ... + ... = 16 reads  <-- this triggers widening
node_16 = ExaModels.AdjointNode2(+, 0.0, 1.0, 1.0, node_8, node_8)

cnt0 = ((), ())

# ---------------------------------------------------------------------------
# Step 2: Check return type of grpass for each size
# ---------------------------------------------------------------------------
println("=== grpass return types (sparsity detection path, comp=nothing) ===\n")
for (name, node) in [("2 reads",  node_2),
                     ("4 reads",  node_4),
                     ("8 reads",  node_8),
                     ("16 reads", node_16)]
    ret = Base.return_types(
        ExaModels.grpass,
        (typeof(node), Nothing, Nothing, Nothing, typeof(cnt0), Float64)
    )
    r = ret[1]
    is_concrete = isconcretetype(r)
    has_vararg  = occursin("Vararg", string(r))
    status = is_concrete ? "STABLE  " : (has_vararg ? "UNSTABLE (Vararg widened!)" : "UNSTABLE")
    println("$name : $status")
    println("  inferred: $r\n")
end

# ---------------------------------------------------------------------------
# Step 3: Show @code_warntype for the 8-read case (last stable one)
# ---------------------------------------------------------------------------
println("\n=== @code_warntype for 8 reads (last stable) ===")
@code_warntype ExaModels.grpass(node_8, nothing, nothing, nothing, cnt0, 1.0)

# ---------------------------------------------------------------------------
# Step 4: Show @code_warntype for the 16-read case (first unstable one)
# ---------------------------------------------------------------------------
println("\n=== @code_warntype for 16 reads (UNSTABLE) ===")
@code_warntype ExaModels.grpass(node_16, nothing, nothing, nothing, cnt0, 1.0)

# ---------------------------------------------------------------------------
# Step 5: Show how it affects _simdfunction
# ---------------------------------------------------------------------------
println("\n=== Effect on _simdfunction ===")
T = Float64
p  = ExaModels.ParSource()
xs = ExaModels.VarSource()
vp = xs[p[1]]  # x[p] - same variable

# (v+v)+(v+v): 4 reads, stable
f_stable = ExaModels.Node2(+, ExaModels.Node2(+, vp, vp), ExaModels.Node2(+, vp, vp))
# Deep nesting: 16 reads via (((f_stable+f_stable)): triggers widening in practice
f_unstable_inner = ExaModels.Node2(+, f_stable, f_stable)   # 8 reads
f_unstable       = ExaModels.Node2(+, f_unstable_inner, f_unstable_inner)  # 16 reads

for (label, f) in [("f_stable (4 reads)", f_stable), ("f_unstable (16 reads)", f_unstable)]
    ret = Base.return_types(ExaModels._simdfunction, (Type{T}, typeof(f), Int, Int, Int))
    r = ret[1]
    is_concrete = isconcretetype(r)
    println("$label  => _simdfunction return concrete: $is_concrete")
    if !is_concrete
        # Extract Compressor type
        println("  type: $r")
    end
end
