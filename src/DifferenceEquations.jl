module DifferenceEquations

using ChainRulesCore
using Distributions
using LinearAlgebra
using CommonSolve
using Parameters
using PDMats
using SciMLBase: SciMLBase, SciMLProblem, solve

include("utilities.jl")
include("problems/state_space_problems.jl")
include("solutions/state_space_solutions.jl")
include("algorithms.jl")
include("algorithms/linear.jl")
include("algorithms/quadratic.jl")
include("solve.jl")

# Exports
export StateSpaceProblem, NoiseConditionalFilter, KalmanFilter, LinearStateSpaceProblem,
       QuadraticStateSpaceProblem, solve

end # module
