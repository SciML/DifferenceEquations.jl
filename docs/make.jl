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
    doctest = false,
    linkcheck = true,
    checkdocs = :exports,
    warnonly = [:missing_docs, :linkcheck],
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
