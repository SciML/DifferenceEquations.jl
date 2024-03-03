using Documenter, DifferenceEquations

include("pages.jl")

makedocs(sitename = "DifferenceEquations.jl",
    authors = "Various Authors",
    clean = true, doctest = false, linkcheck = true,
    modules = [DifferenceEquations],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://DifferenceEquations.sciml.ai/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/DifferenceEquations.jl";
    push_preview = true)
