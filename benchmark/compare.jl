using JLD2, Printf

main_file = joinpath(@__DIR__, "benchmark-results-main.jld2")
current_file = joinpath(@__DIR__, "benchmark-results-current.jld2")

isfile(main_file) || error("Missing $main_file — run 'make benchmark-main' first")
isfile(current_file) || error("Missing $current_file — run 'make benchmark-current' first")

main_res = JLD2.load(main_file, "results")
current_res = JLD2.load(current_file, "results")

all_names = collect(union(keys(main_res), keys(current_res)))

# Sort order for backends within each (model, size) group.
# GPU backends come first so the most interesting results appear early.
const BACKEND_ORDER = Dict("CUDA" => 0, "AMDGPU" => 1, "oneAPI" => 2, "Metal" => 3, "nothing" => 4)

# Keys are 4-tuples: (group_index, model_name, size_as_int, backend_index).
# Benchmark names follow the convention "<model>-<size>-<backend>", e.g.
# "rosenrock-1000-CUDA".  The size field is parsed as an integer so that
# 1000 < 10000 < 100000 (lexicographic order would give the wrong result).
function sort_key(name)
    parts = split(name, "-")
    # Last part is backend label, second-to-last is size (numeric), rest is model name
    backend = parts[end]
    size_str = length(parts) >= 2 ? parts[end - 1] : "0"
    size_num = something(tryparse(Int, size_str), 0)
    model = join(parts[1:end-2], "-")
    border = get(BACKEND_ORDER, backend, 99)

    startswith(name, "rosenrock") && return (1, model, size_num, border)
    startswith(name, "OPF")       && return (2, model, size_num, border)
    startswith(name, "chain")     && return (3, model, size_num, border)
    startswith(name, "elec")      && return (4, model, size_num, border)
    return (5, model, size_num, border)
end

sorted_names = sort(all_names; by = sort_key)

# Format a timing ratio as "current / main".  Values below 1.0 are
# improvements.  Returns "N/A" when either measurement is missing (NaN) or
# when main is zero (would give division-by-zero / Inf).
function ratio_str(c, m)
    (isnan(c) || isnan(m) || m == 0) && return "     N/A"
    return @sprintf("%8.3f", c / m)
end

const NAME_W = 26
const SEP = "=" ^ (NAME_W + 5 + 5 * 9)

println()
println("Relative timing: current / main  (values < 1.0 are improvements)")
println()
println(SEP)
@printf("  %-*s  | %8s %8s %8s %8s %8s\n", NAME_W, "name", "obj", "cons", "grad", "jac", "hess")
println(SEP)

# prev_group is a Ref so that assignment inside the for-loop body modifies the
# outer binding rather than creating a new loop-local variable (Julia soft-scope).
prev_group = Ref("")
for name in sorted_names
    group = split(name, "-")[1]
    if group != prev_group[] && prev_group[] != ""
        println("-" ^ length(SEP))
    end
    prev_group[] = group

    if !haskey(main_res, name)
        @printf("  %-*s  |  (no main result)\n", NAME_W, name)
        continue
    end
    if !haskey(current_res, name)
        @printf("  %-*s  |  (no current result)\n", NAME_W, name)
        continue
    end
    m = main_res[name]
    c = current_res[name]
    @printf("  %-*s  | %s %s %s %s %s\n",
        NAME_W, name,
        ratio_str(c.tobj,  m.tobj),
        ratio_str(c.tcon,  m.tcon),
        ratio_str(c.tgrad, m.tgrad),
        ratio_str(c.tjac,  m.tjac),
        ratio_str(c.thess, m.thess),
    )
end

println(SEP)
println()
