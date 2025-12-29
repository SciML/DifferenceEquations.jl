module DifferenceEquations

using ChainRulesCore: ChainRulesCore, NoTangent, Tangent, ZeroTangent
using CommonSolve: CommonSolve, solve
using DiffEqBase: DiffEqBase, DEProblem, get_concrete_u0, get_concrete_p, isconcreteu0,
                  promote_u0
using Distributions: Distributions, Distribution, MvNormal, UnivariateDistribution,
                     ZeroMeanDiagNormal, logpdf
using LinearAlgebra: LinearAlgebra, Cholesky, Diagonal, NoPivot, Symmetric, cholesky!,
                     dot, ldiv!, lmul!, mul!, rmul!, transpose!
using PDMats: PDMats, PDMat
using SciMLBase: SciMLBase, @add_kwonly, NullParameters, promote_tspan, AbstractRODESolution,
                 ODEFunction, remake, ConstantInterpolation, build_solution
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
