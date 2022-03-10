module DifferenceEquations

using ChainRulesCore
using Distributions
using LinearAlgebra
using CommonSolve
using UnPack
using PDMats
using RecursiveArrayTools
using SciMLBase: SciMLBase, SciMLProblem, solve, @add_kwonly, NullParameters, promote_tspan,
                 AbstractRODESolution

include("matrix_vector_of_vectors.jl")
include("utilities.jl")
include("problems/state_space_problems.jl")
include("solutions/state_space_solutions.jl")
include("algorithms.jl")
include("algorithms/linear.jl")
include("algorithms/quadratic.jl")
include("solve.jl")

# Exports
export MatrixVectorOfArray

export StateSpaceProblem, NoiseConditionalFilter, KalmanFilter, LinearStateSpaceProblem,
       QuadraticStateSpaceProblem, solve

end # module
