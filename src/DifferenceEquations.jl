module DifferenceEquations

using Distributions
using DistributionsAD
using LinearAlgebra
using CommonSolve
using Zygote

import SciMLBase
import SciMLBase: SciMLProblem, solve
import StatsBase

abstract type DifferenceProblem <: SciMLProblem end
abstract type AbstractStateSpaceProblem{isinplace} <: DifferenceProblem end

# Wrapper struct, eventually needs to be a full cache
struct StateSpaceCache{
    probtype<:AbstractStateSpaceProblem, 
    solvertype<:SciMLBase.SciMLAlgorithm
}
    problem::probtype
    solver::solvertype
end 

function StateSpaceCache(problem::AbstractStateSpaceProblem, solver::SciMLBase.SciMLAlgorithm)
    return StateSpaceCache(problem, solver)
end

# Unpack the cache. In future, this unwrapping should be eliminated when the cache
# actually does something more than just wrap around prob/solver.
function CommonSolve.solve!(
    cache::StateSpaceCache,
    args...; 
    kwargs...
)
    return _solve!(cache.problem, cache.solver, args...; vectype=cache.vtype, kwargs...)
end

# Yuck hate this so much
promote_noise(x, y) = [x], [y]
promote_noise(x, y::AbstractArray) = [x], y
promote_noise(x::AbstractArray, y) = x, [y]
promote_noise(x::AbstractArray, y::AbstractArray) = x, y

include("noise.jl")
include("alg.jl")
include("linear.jl")
include("solution.jl")
include("nonlinear.jl")
include("kalman.jl")

# Exports
export 
    StateSpaceProblem,
    NoiseConditionalFilter,
    KalmanFilter,
    LinearStateSpaceProblem,
    StandardGaussian,
    DefinedNoise,
    solve

end # module
