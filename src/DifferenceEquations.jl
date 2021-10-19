module DifferenceEquations

using SciMLBase
using Distributions
using LinearAlgebra
using CommonSolve

import SciMLBase: SciMLProblem

abstract type DifferenceProblem <: SciMLProblem end
abstract type AbstractStateSpaceProblem{isinplace} <: DifferenceProblem end

# Yuck hate this so much
promote_noise(x, y) = [x], [y]
promote_noise(x, y::AbstractArray) = [x], y
promote_noise(x::AbstractArray, y) = x, [y]
promote_noise(x::AbstractArray, y::AbstractArray) = x, y

include("noise.jl")
include("alg.jl")
include("linear.jl")
include("nonlinear.jl")
include("solution.jl")

# Exports
export StateSpaceProblem,
    ConditionalGaussian,
    LinearGaussian,
    LinearStateSpaceProblem,
    StandardGaussian,
    DefinedNoise,
    solve

end # module
