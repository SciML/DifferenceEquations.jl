using Test
using DifferenceEquations
using Distributions
using LinearAlgebra
using Random

@testset "Linear tests" begin
    include("linear.jl")
end

@testset "Quadratic tests" begin
    include("quadratic.jl")
end
