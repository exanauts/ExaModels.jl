# ============================================================================
# Collect all result CSVs into a single DataFrame
# ============================================================================
#
# Usage: julia --project=. collect.jl
#
# Reads all results/*.csv files, adds device info from companion _hw.txt,
# and writes results/combined.csv.

using CSV, DataFrames

function read_hw_info(hw_path)
    info = Dict{String,String}()
    isfile(hw_path) || return info
    for line in readlines(hw_path)
        parts = split(line, ":"; limit=2)
        length(parts) == 2 && (info[strip(parts[1])] = strip(parts[2]))
    end
    return info
end

function main()
    files = filter(f -> endswith(f, ".csv") && !startswith(basename(f), "combined"), readdir("results"; join=true))

    dfs = DataFrame[]
    for f in files
        @info "Reading $f"
        df = CSV.read(f, DataFrame)
        nrow(df) == 0 && continue

        # Read companion hardware file
        hw_path = replace(f, r"\.csv$" => "_hw.txt")
        hw = read_hw_info(hw_path)
        df.device .= get(hw, "device", "unknown")
        df.hostname .= get(hw, "hostname", "unknown")

        push!(dfs, df)
    end

    isempty(dfs) && error("No result files found")

    combined = vcat(dfs...; cols = :union)
    # Fill missing columns with 0.0
    for col in names(combined)
        if eltype(combined[!, col]) >: Missing
            combined[!, col] = coalesce.(combined[!, col], 0.0)
        end
    end
    outfile = joinpath("results", "combined.csv")
    CSV.write(outfile, combined)
    @info "Combined $(nrow(combined)) rows → $outfile"
end

main()
