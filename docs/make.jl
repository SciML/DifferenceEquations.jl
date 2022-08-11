using Documenter, DifferenceEquations

include("pages.jl")

makedocs(sitename = "DifferenceEquations.jl",
         authors = "Various Authors",
         clean = true,
         doctest = :fix,  # swap to "false" at some point
         modules = [DifferenceEquations],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://DifferenceEquations.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/DifferenceEquations.jl";
           push_preview = true)