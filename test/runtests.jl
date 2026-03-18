using Pkg
using Test
using DifferenceEquations
using Distributions
using LinearAlgebra
using Random

const GROUP = get(ENV, "GROUP", "All")

function activate_jet_env()
    Pkg.activate("jet")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    # include("matrix_vector_of_vectors.jl") # may add later to support noise inputs as vector of vectors
    include("qa.jl")
    include("explicit_imports.jl")
    include("kalman_likelihood.jl")
    include("linear_likelihood.jl")
    # include("linear_gradients.jl")  # AD tests disabled — will restore with Enzyme
    include("linear_simulations.jl")
    include("quadratic_likelihood.jl")
    include("quadratic_simulations.jl")
    include("sciml_interfaces.jl")
end

if GROUP == "JET"
    activate_jet_env()
    include("jet/jet_tests.jl")
end
