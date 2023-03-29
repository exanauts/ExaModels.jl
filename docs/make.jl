using Documenter, MadDiff, Literate

const _PAGES = [
    "Introduction" => "index.md",
    "Quick Start"=>"guide.md",
    "How it Works" => "tutorial.md",
    "API Manual" => [
        "MadDiffCore" => "core.md",
        "MadDiffSpecialFunctions" => "special.md",
        "MadDiffModels" => "models.md",
        "MadDiffMOI" => "moi.md",
    ]
]

const _JL_FILENAMES = [
    "guide.jl",
    "tutorial.jl"
]

for jl_filename in _JL_FILENAMES

    Literate.markdown(
        joinpath(@__DIR__,"src", jl_filename),
        joinpath(@__DIR__,"src");
        documenter = true, 
        execute = true, 
    )

end


makedocs(
    sitename = "MadDiff",
    authors = "Sungho Shin",
    format = Documenter.LaTeX(platform="docker"),
    pages = _PAGES
)

makedocs(
    sitename = "MadDiff",
    modules = [MadDiff],
    authors = "Sungho Shin",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        sidebar_sitename = true,
        collapselevel = 1,
    ),
    pages = _PAGES,
    clean = false,
)


deploydocs(
    repo = "github.com/sshin23/MadDiff.jl.git"
)

