using Documenter, DifferenceEquations
using DocumenterInterLinks

# Force the GR null device. Without it, the Plots-based tutorials segfault on CI
# (exit 139, no Julia stack trace) while GR probes for a display.
ENV["GKSwstype"] = "100"

# TEMP DIAGNOSTIC: identify the page/block that segfaults on CI (revert before merge)
ENV["JULIA_DEBUG"] = "Documenter"

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
