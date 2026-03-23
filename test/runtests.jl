using Pkg
using Test
using DifferenceEquations
using Distributions
using LinearAlgebra
using Random

function activate_jet_env()
    Pkg.activate("jet")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

include("qa.jl")
include("explicit_imports.jl")
include("kalman_likelihood.jl")
include("linear_likelihood.jl")
# include("linear_gradients.jl")  # AD tests disabled — will restore with Enzyme
include("linear_simulations.jl")
include("generic_simulations.jl")
include("generic_sciml.jl")
include("cache_reuse.jl")
include("static_arrays.jl")
include("sciml_interfaces.jl")
include("sensitivity_interface.jl")
include("enzyme_kalman.jl")
include("enzyme_direct_iteration.jl")

if get(ENV, "GROUP", "") == "JET"
    activate_jet_env()
    include("jet/jet_tests.jl")
end
