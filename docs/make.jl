using Documenter, ExaModels, Literate

const _PAGES = [
    "Introduction" =>"index.md",
    "SIMD Abstraction"=>"simd.md",
    "Quick Start"=>"guide.md",
    "API Manual" => "core.md",
]

const _JL_FILENAMES = [
    "guide.jl",
    # "tutorial.jl"
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
    sitename = "ExaModels",
    authors = "Sungho Shin",
    format = Documenter.LaTeX(platform="docker"),
    pages = _PAGES
)

makedocs(
    sitename = "ExaModels.jl",
    modules = [ExaModels],
    authors = "Sungho Shin",
    format = Documenter.HTML(
        prettyurls = true,
        sidebar_sitename = true,
        collapselevel = 1,
    ),
    pages = _PAGES,
    clean = false,
)


deploydocs(
    repo = "github.com/sshin23/ExaModels.jl.git"
)

