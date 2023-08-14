using Documenter, DocumenterCitations, ExaModels, Literate

if !(@isdefined _LATEX)
    const _LATEX = true
end

if !(@isdefined _PAGES)
    const _PAGES = [
        "Introduction" => [
            "index.md",
            "highlights.md",
        ],
        "Mathematical Abstraction" => "simd.md",
        "Tutorial" => [
            "guide.md",
            "performance.md",
            "gpu.md",
            "develop.md",
            "quad.md",
            "dist.md",
            "opf.md",
        ],
        "API Manual" => "core.md",
        "References" => "ref.md"
    ]
end

if !(@isdefined _JL_FILENAMES)
    const _JL_FILENAMES = [
        "guide.jl",
        "quad.jl",
        "dist.jl",
        "opf.jl",
        "gpu.jl",
        "performance.jl"
    ]
end

for jl_filename in _JL_FILENAMES

    Literate.markdown(
        joinpath(@__DIR__, "src", jl_filename),
        joinpath(@__DIR__, "src");
        documenter = true,
        execute = true,
    )

end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

if _LATEX
    makedocs(
    bib, 
    sitename = "ExaModels",
    authors = "Sungho Shin",
    format = Documenter.LaTeX(platform = "docker"),
    pages = _PAGES,
)
end


makedocs(
    bib,
    sitename = "ExaModels.jl",
    modules = [ExaModels],
    authors = "Sungho Shin",
    format = Documenter.HTML(
        assets = [
            "assets/favicon.ico",
            "assets/citations.css"
        ],
        prettyurls = true,
        sidebar_sitename = true,
        collapselevel = 1
    ),
    pages = _PAGES,
    clean = false,
)


deploydocs(repo = "github.com/sshin23/ExaModels.jl.git")

