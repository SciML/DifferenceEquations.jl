module DifferenceEquations

using SciMLBase
using Distributions

import SciMLBase: SciMLProblem

abstract type DifferenceProblem <: SciMLProblem end
abstract type AbstractStateSpaceProblem{isinplace} <: DifferenceProblem end

include("problem.jl")
include("solution.jl")

# Exports
export StateSpaceProblem

end # module