module DifferenceEquations

using ChainRulesCore
using Distributions
using LinearAlgebra
using CommonSolve
using Zygote
using Parameters
using PDMats
#using Infiltrator # TEMP Add or remove during testing

using SciMLBase: SciMLBase, SciMLProblem, solve

struct NoiseConditionalFilter <: SciMLBase.SciMLAlgorithm end
struct KalmanFilter <: SciMLBase.SciMLAlgorithm end
abstract type DifferenceProblem <: SciMLProblem end
abstract type AbstractStateSpaceProblem{isinplace} <: DifferenceProblem end

# Wrapper struct, eventually needs to be a full cache
struct StateSpaceCache{probtype<:AbstractStateSpaceProblem,solvertype<:SciMLBase.SciMLAlgorithm}
    problem::probtype
    solver::solvertype
end

function StateSpaceCache(problem::AbstractStateSpaceProblem, solver::SciMLBase.SciMLAlgorithm)
    return StateSpaceCache(problem, solver)
end

# Unpack the cache. In future, this unwrapping should be eliminated when the cache
# actually does something more than just wrap around prob/solver.
function CommonSolve.solve!(cache::StateSpaceCache, args...; kwargs...)
    return _solve!(cache.problem, cache.solver, args...; kwargs...)
end

include("utilities.jl")
include("linear.jl")
include("quadratic.jl")
include("solution.jl")
include("nonlinear.jl")
include("kalman.jl")

# Exports
export StateSpaceProblem, NoiseConditionalFilter, KalmanFilter, LinearStateSpaceProblem,
       QuadraticStateSpaceProblem, solve

end # module
