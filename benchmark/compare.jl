using JLD2, Printf

main_file    = joinpath(@__DIR__, "benchmark-results-main.jld2")
current_file = joinpath(@__DIR__, "benchmark-results-current.jld2")

isfile(main_file)    || error("Missing $main_file — run 'make benchmark-main' first")
isfile(current_file) || error("Missing $current_file — run 'make benchmark-current' first")

main_res    = JLD2.load(main_file,    "results")
current_res = JLD2.load(current_file, "results")

all_names = collect(union(keys(main_res), keys(current_res)))

# Sort order for backends (lower = first in output)
const BACKEND_ORDER = Dict("CUDA" => 0, "AMDGPU" => 1, "oneAPI" => 2, "Metal" => 3, "nothing" => 4)

function sort_key(name)
    parts = split(name, "-")
    # Last part is backend label, second-to-last is size (numeric), rest is model name
    backend = parts[end]
    size_str = length(parts) >= 2 ? parts[end-1] : "0"
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

function ratio_str(c, m)
    (isnan(c) || isnan(m) || m == 0) && return "     N/A"
    @sprintf("%8.3f", c / m)
end

const NAME_W = 26
const SEP = "=" ^ (NAME_W + 5 + 5 * 9)

println()
println("Relative timing: current / main  (values < 1.0 are improvements)")
println()
println(SEP)
@printf("  %-*s  | %8s %8s %8s %8s %8s\n", NAME_W, "name", "obj", "cons", "grad", "jac", "hess")
println(SEP)

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
