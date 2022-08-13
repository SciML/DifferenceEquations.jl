module DifferenceEquations

using ChainRulesCore
using Distributions
using LinearAlgebra
using CommonSolve
using UnPack
using PDMats
using SciMLBase: @add_kwonly, NullParameters, promote_tspan, AbstractRODESolution
using DiffEqBase
using DiffEqBase: __solve
using SciMLBase: build_solution

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
