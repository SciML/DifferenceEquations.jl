module DifferenceEquations

using ChainRulesCore
using Distributions
using LinearAlgebra
using CommonSolve
using UnPack
using PDMats
using RecursiveArrayTools
using SciMLBase: @add_kwonly, NullParameters, promote_tspan, AbstractRODESolution
using DiffEqBase
using DiffEqBase: __solve

include("matrix_vector_of_vectors.jl")
include("utilities.jl")
include("problems/state_space_problems.jl")
include("solutions/state_space_solutions.jl")
include("solve.jl")
include("algorithms/linear.jl")
include("algorithms/quadratic.jl")

# Exports
export MatrixVectorOfArray

export AbstractStateSpaceProblem, LinearStateSpaceProblem, QuadraticStateSpaceProblem
export StateSpaceSolution, DirectIteration, KalmanFilter

export solve

end # module
