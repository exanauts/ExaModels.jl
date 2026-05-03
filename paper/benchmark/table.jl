# ============================================================================
# Generate LaTeX tables from combined.csv
# ============================================================================
#
# Usage: julia --project=. table.jl
#
# Reads results/combined.csv, produces .tex files in results/tables/

using CSV, DataFrames, Printf

function load_combined()
    return CSV.read("results/combined.csv", DataFrame)
end

function fmt_time(t)
    if t == 0.0
        return "---"
    else
        return @sprintf("%.2e", t)
    end
end

function fmt_speedup(s)
    if s == Inf || isnan(s)
        return "---"
    else
        return @sprintf("%.1f\\times", s)
    end
end

function fmt_int(n)
    if n < 1_000
        return string(n)
    elseif n < 1_000_000
        return @sprintf("%dk", round(Int, n / 1_000))
    else
        return @sprintf("%.1fM", n / 1_000_000)
    end
end

function sgm(vals; shift = 1e-5)
    n = length(vals)
    n == 0 && return NaN
    return exp(sum(log.(vals .+ shift)) / n) - shift
end

# ============================================================================
# Size classification by max(nnzj, nnzh)
# ============================================================================

function classify_size(nnzj, nnzh)
    nnz = max(nnzj, nnzh)
    if nnz < 1_000
        return "Small"
    elseif nnz < 100_000
        return "Medium"
    else
        return "Large"
    end
end

# ============================================================================
# SGM summary table — NeurIPS style (small/medium/large/total)
# ============================================================================

function generate_sgm_summary_table(df; fname = joinpath("results", "tables", "sgm_summary.tex"), shift = 1e-5)
    # Classify each row
    df = copy(df)
    df.sizeclass = [classify_size(r.nnzj, r.nnzh) for r in eachrow(df)]

    callbacks = [:tobj, :tcon, :tgrad, :tjac, :thess, :tcreate]
    labels = [
        "\\texttt{obj}",
        "\\texttt{cons!}",
        "\\texttt{grad!}",
        "\\texttt{jac\\_coord!}",
        "\\texttt{hess\\_coord!}",
        "create",
    ]

    suites = sort(unique(df.suite))
    size_classes = ["Small", "Medium", "Large", "Total"]

    open(fname, "w") do io
        # Columns: suite, callback, ExaModels, JuMP, Small speedup, Medium speedup, Large speedup, Total speedup
        println(io, "\\begin{tabular}{ll rr rrrr}")
        println(io, "  \\toprule")
        println(io, "  & & & & \\multicolumn{4}{c@{}}{\\textbf{Speedup} (JuMP\\,/\\,ExaModels)} \\\\")
        println(io, "  \\cmidrule(l){5-8}")
        print(io, "  suite & callback & ExaModels & JuMP")
        print(io, " & \\shortstack{Small\\\\[-2pt]{\\tiny nnz\$<10^3\$}}")
        print(io, " & \\shortstack{Medium\\\\[-2pt]{\\tiny \$10^3{\\leq}\$nnz\${<}10^5\$}}")
        print(io, " & \\shortstack{Large\\\\[-2pt]{\\tiny \$10^5{\\leq}\$nnz}}")
        print(io, " & Total")
        println(io, " \\\\")
        println(io, "  \\midrule")

        for (si, suite) in enumerate(suites)
            sub = filter(r -> r.suite == suite, df)

            has_jump = "JuMP" in unique(sub.ams)

            first_row = true
            for (ci, cb) in enumerate(callbacks)
                # Skip callbacks not in data or all-zero
                hasproperty(sub, cb) || continue
                exa_sub = filter(r -> r.ams == "ExaModels", sub)
                if cb in (:tcon, :tjac)
                    all_zero = nrow(exa_sub) == 0 || all(r -> r[cb] == 0.0, eachrow(exa_sub))
                    all_zero && continue
                end

                suite_label = first_row ? replace(suite, "_" => "\\_") : ""
                first_row = false

                # Total ExaModels and JuMP SGM values
                exa_all = filter(r -> r.ams == "ExaModels", sub)
                jmp_all = filter(r -> r.ams == "JuMP", sub)
                exa_total = nrow(exa_all) > 0 ? max(sgm(exa_all[!, cb]; shift = shift), 0.0) : NaN
                jmp_total = nrow(jmp_all) > 0 ? max(sgm(jmp_all[!, cb]; shift = shift), 0.0) : NaN

                print(io, "  ", suite_label, " & ", labels[ci])
                print(io, " & ", isnan(exa_total) ? "---" : fmt_time(exa_total))
                print(io, " & ", isnan(jmp_total) ? "---" : fmt_time(jmp_total))

                # Speedup per size class + total
                for sc in ["Small", "Medium", "Large", "Total"]
                    if sc == "Total"
                        exa = exa_all
                        jmp = jmp_all
                    else
                        exa = filter(r -> r.ams == "ExaModels" && r.sizeclass == sc, sub)
                        jmp = filter(r -> r.ams == "JuMP" && r.sizeclass == sc, sub)
                    end

                    exa_val = nrow(exa) > 0 ? max(sgm(exa[!, cb]; shift = shift), 0.0) : NaN
                    jmp_val = nrow(jmp) > 0 ? max(sgm(jmp[!, cb]; shift = shift), 0.0) : NaN

                    if !isnan(exa_val) && !isnan(jmp_val) && exa_val > 0 && jmp_val > 0
                        sp = jmp_val / exa_val
                        print(io, " & \$", fmt_speedup(sp), "\$")
                    else
                        print(io, " & ---")
                    end
                end

                println(io, " \\\\")
            end
            si < length(suites) && println(io, "  \\midrule")
        end

        println(io, "  \\bottomrule")
        println(io, "\\end{tabular}")
    end

    @info "SGM summary table written to $fname"
end

# ============================================================================
# Full appendix tables — one per suite
# ============================================================================

function generate_appendix_table(df, suite; fname = nothing)
    sub = sort(filter(r -> r.suite == suite, df), [:problem, :size, :ams])
    isempty(sub) && return

    fname === nothing && (fname = joinpath("results", "tables", "appendix_$(suite).tex"))

    # Check if suite has constraints
    has_con = any(r -> r.ncon > 0, eachrow(sub))

    open(fname, "w") do io
        println(io, "{\\scriptsize")
        println(io, "\\setlength{\\tabcolsep}{4pt}")

        # Build column spec dynamically
        cols = "@{}ll rrr"  # instance, AMS, nvar, ncon, nnzh
        header = "instance & AMS & nvar & ncon & nnzh"

        cols *= " rrr"  # obj, grad, hess
        header *= " & obj & grad & hess"

        if has_con
            cols *= " rr"  # cons, jac
            header *= " & cons & jac"
        end

        cols *= " r@{}"  # create
        header *= " & create"

        println(io, "\\begin{longtable}{$cols}")
        println(io, "  \\toprule")
        println(io, "  $header \\\\")
        println(io, "  \\midrule")
        println(io, "  \\endfirsthead")
        println(io, "  \\toprule")
        println(io, "  $header \\\\")
        println(io, "  \\midrule")
        println(io, "  \\endhead")

        prev_key = ""
        for row in eachrow(sub)
            key = "$(row.problem)_$(row.size)"

            # Format instance name
            display_name = row.problem
            display_name = replace(display_name, "pglib_opf_" => "")
            display_name = replace(display_name, ".m" => "")

            # For OPF, size==problem so just use the short name; otherwise name-size
            if startswith(suite, "OPF")
                label = display_name
            else
                label = "$(display_name)-$(row.size)"
            end
            pname = key == prev_key ? "" : replace(label, "_" => "\\_")
            prev_key = key

            print(io, "  ", pname, " & ", row.ams)
            print(io, " & ", fmt_int(row.nvar), " & ", fmt_int(row.ncon), " & ", fmt_int(row.nnzh))
            print(io, " & ", fmt_time(row.tobj), " & ", fmt_time(row.tgrad), " & ", fmt_time(row.thess))

            if has_con
                print(io, " & ", fmt_time(row.tcon), " & ", fmt_time(row.tjac))
            end

            print(io, " & ", fmt_time(row.tcreate))
            println(io, " \\\\")
        end

        println(io, "  \\bottomrule")
        println(io, "\\end{longtable}")
        println(io, "}")
    end

    @info "Appendix table written to $fname"
end

# ============================================================================
# Main
# ============================================================================

function main()
    df = load_combined()
    mkpath(joinpath("results", "tables"))

    suites = sort(unique(df.suite))

    # SGM summary table (NeurIPS style: small/medium/large/total)
    generate_sgm_summary_table(df)

    # Appendix tables
    for suite in suites
        generate_appendix_table(df, suite)
    end

    @info "All tables saved to results/tables/"
end

main()
