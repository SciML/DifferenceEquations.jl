module DifferenceEquations

using SciMLBase
using Distributions
using LinearAlgebra

import SciMLBase: SciMLProblem

abstract type DifferenceProblem <: SciMLProblem end
abstract type AbstractStateSpaceProblem{isinplace} <: DifferenceProblem end

include("alg.jl")
include("problem.jl")
include("solution.jl")

# Exports
export StateSpaceProblem,
    ConditionalGaussian

end # module
