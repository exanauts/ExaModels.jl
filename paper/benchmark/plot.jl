# ============================================================================
# Generate benchmark plots from combined.csv
# ============================================================================
#
# Usage: julia --project=. plot.jl
#
# Reads results/combined.csv, produces PDF figures in results/figures/

using CSV, DataFrames, CairoMakie, LaTeXStrings

function load_combined()
    df = CSV.read("results/combined.csv", DataFrame)
    return df
end

const CALLBACKS = [:tobj, :tcon, :tgrad, :tjac, :thess]
const CALLBACK_LABELS = Dict(
    :tobj  => "obj",
    :tcon  => "cons",
    :tgrad => "grad",
    :tjac  => "jac",
    :thess => "hess",
)

const MARKERS = [:circle, :utriangle, :dtriangle, :rect, :diamond,
                 :star4, :star5, :cross, :xcross, :hexagon, :pentagon]

"""
    plot_speedup(df, suite; reference="JuMP")

For each callback, plot speedup of each device/ams relative to `reference`.
Problems sorted by nvar.
"""
function plot_speedup(df, suite; reference = "JuMP")
    sub = filter(r -> startswith(r.suite, suite), df)
    isempty(sub) && return

    # Get unique device labels (ams + device)
    sub.label = sub.ams .* " / " .* sub.device

    # Reference rows
    ref = filter(r -> r.ams == reference, sub)
    isempty(ref) && (ref = filter(r -> r.label == first(sort(unique(sub.label))), sub); reference = first(sort(unique(sub.ams))))

    labels = sort(unique(sub.label))

    mkpath(joinpath("results", "figures"))

    for cb in CALLBACKS
        fig = Figure(size = (1200, 350), backgroundcolor = :transparent)

        # Build x-axis: unique (problem, size) pairs sorted by nvar
        ref_sorted = sort(ref, :nvar)
        cases = collect(zip(ref_sorted.problem, ref_sorted.size))
        nv = ref_sorted.nvar
        xticks_labels = ["$(p)\n$(s)" for (p, s) in cases]
        xs = 1:length(cases)

        ax = Axis(fig[1, 1];
            yscale = log10,
            ylabel = L"$y$× faster than %$(reference) (%$(CALLBACK_LABELS[cb]))",
            xticks = (collect(xs), xticks_labels),
            xticklabelrotation = π/4,
            xautolimitmargin = (0.02, 0.02),
            backgroundcolor = :transparent,
        )
        hlines!(ax, [1.0]; color = :gray, linestyle = :dash, linewidth = 0.5)

        for (i, lbl) in enumerate(labels)
            lsub = filter(r -> r.label == lbl, sub)
            ref_vals = Float64[]
            lbl_vals = Float64[]
            xpos = Int[]

            for (j, (p, s)) in enumerate(cases)
                rv = filter(r -> r.problem == p && r.size == s, ref)
                lv = filter(r -> r.problem == p && r.size == s, lsub)
                if nrow(rv) > 0 && nrow(lv) > 0
                    push!(ref_vals, rv[1, cb])
                    push!(lbl_vals, lv[1, cb])
                    push!(xpos, j)
                end
            end

            if !isempty(xpos)
                speedups = ref_vals ./ lbl_vals
                marker = MARKERS[mod1(i, length(MARKERS))]
                scatter!(ax, xpos, speedups; marker = marker, markersize = 12, alpha = 0.75, label = lbl)
            end
        end

        # Legend
        fig_legend = Figure(size = (500, 200), backgroundcolor = :transparent)
        Legend(fig_legend[1, 1], ax)

        save(joinpath("results", "figures", "$(suite)_$(cb).pdf"), fig)
        save(joinpath("results", "figures", "$(suite)_legend.pdf"), fig_legend)
    end
end

"""
    plot_creation(df, suite)

Bar chart of model creation times.
"""
function plot_creation(df, suite)
    sub = filter(r -> startswith(r.suite, suite), df)
    isempty(sub) && return

    sub.label = sub.ams .* " / " .* sub.device
    labels = sort(unique(sub.label))

    mkpath(joinpath("results", "figures"))

    fig = Figure(size = (1200, 350), backgroundcolor = :transparent)

    ref = sort(filter(r -> r.label == labels[1], sub), :nvar)
    cases = collect(zip(ref.problem, ref.size))
    xticks_labels = ["$(p)\n$(s)" for (p, s) in cases]
    xs = 1:length(cases)

    ax = Axis(fig[1, 1];
        yscale = log10,
        ylabel = "creation time (s)",
        xticks = (collect(xs), xticks_labels),
        xticklabelrotation = π/4,
        backgroundcolor = :transparent,
    )

    for (i, lbl) in enumerate(labels)
        lsub = filter(r -> r.label == lbl, sub)
        xpos = Int[]
        vals = Float64[]
        for (j, (p, s)) in enumerate(cases)
            row = filter(r -> r.problem == p && r.size == s, lsub)
            if nrow(row) > 0
                push!(xpos, j)
                push!(vals, row[1, :tcreate])
            end
        end
        if !isempty(xpos)
            marker = MARKERS[mod1(i, length(MARKERS))]
            scatter!(ax, xpos, vals; marker = marker, markersize = 12, alpha = 0.75, label = lbl)
        end
    end

    save(joinpath("results", "figures", "$(suite)_creation.pdf"), fig)
end

function main()
    df = load_combined()

    with_theme(theme_latexfonts()) do
        for suite in ["LV", "COPS", "OPF-polar", "OPF-rect"]
            @info "Plotting $suite"
            plot_speedup(df, suite)
            plot_creation(df, suite)
        end
    end

    @info "Figures saved to results/figures/"
end

main()
