using Test
using DifferenceEquations
using Distributions
using LinearAlgebra
using Random

# At this point, the test here is just checking whether the codes can run, but not to check the validity of the results
# TODO: add unit tests

@testset "Tests based on DifferentiableStateSpaceModels" begin
    include("dssm.jl")
end
