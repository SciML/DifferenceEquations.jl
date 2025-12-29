module DifferenceEquations

using ChainRulesCore: ChainRulesCore, NoTangent, Tangent, ZeroTangent
using CommonSolve: CommonSolve, solve
using DiffEqBase: DiffEqBase
using Distributions: Distributions, Distribution, MvNormal, UnivariateDistribution,
                     ZeroMeanDiagNormal, logpdf
using LinearAlgebra: LinearAlgebra, Cholesky, Diagonal, NoPivot, Symmetric, cholesky!, dot,
                     ldiv!, lmul!, mul!, rmul!, transpose!
using PDMats: PDMats, PDMat
using SciMLBase: SciMLBase, @add_kwonly, NullParameters, promote_tspan, build_solution,
                 ODEFunction, remake
using UnPack: UnPack, @unpack

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
