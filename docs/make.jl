using Documenter, DifferenceEquations
using DocumenterInterLinks

include("pages.jl")

links = InterLinks(
    "SciMLBase" => "https://docs.sciml.ai/SciMLBase/stable/",
)

makedocs(
    sitename = "DifferenceEquations.jl",
    authors = "Various Authors",
    clean = true,
    doctest = true,
    linkcheck = true,
    checkdocs = :exports,
    modules = [DifferenceEquations],
    plugins = [links],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DifferenceEquations/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/DifferenceEquations.jl";
    push_preview = true
)
