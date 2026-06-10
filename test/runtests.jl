using Pkg
using Test

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    include(joinpath(@__DIR__, "qa", "qa.jl"))
end

if GROUP == "All" || GROUP == "Core"
    using DifferenceEquations
    using Distributions
    using LinearAlgebra
    using Random

    include("linear_direct_iteration.jl")
    include("kalman.jl")
    include("direct_iteration.jl")
    include("quadratic_direct_iteration.jl")
    include("static_arrays.jl")
    include("cache_reuse.jl")
    include("sciml_interfaces.jl")
    include("sensitivity_interface.jl")
    include("linear_direct_iteration_forwarddiff.jl")
    include("kalman_forwarddiff.jl")
    include("conditional_likelihood.jl")
    include("conditional_likelihood_forwarddiff.jl")
    include("save_everystep.jl")

    if get(ENV, "CI", "false") != "true"
        include("gradient_comparison.jl")
        include("linear_direct_iteration_enzyme.jl")
        include("quadratic_direct_iteration_enzyme.jl")
        include("kalman_enzyme.jl")
        include("conditional_likelihood_enzyme.jl")
    end
end
