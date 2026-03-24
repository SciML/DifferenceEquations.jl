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
include("linear_direct_iteration.jl")
include("kalman.jl")
include("direct_iteration.jl")
include("static_arrays.jl")
include("cache_reuse.jl")
include("sciml_interfaces.jl")
include("sensitivity_interface.jl")
include("linear_direct_iteration_enzyme.jl")
include("kalman_enzyme.jl")

if get(ENV, "GROUP", "") == "JET"
    activate_jet_env()
    include("jet/jet_tests.jl")
end
