using Test
using DifferenceEquations
using Distributions
using LinearAlgebra
using Random

include("matrix_vector_of_vectors.jl")
include("kalman_likelihood.jl")
include("linear_likelihood.jl")
include("linear_likelihood_gradients.jl")
include("linear_simulations.jl")
include("quadratic_likelihood.jl")
include("quadratic_simulations.jl")
include("sciml_interfaces.jl")
