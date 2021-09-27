module DifferenceEquations

using SciMLBase

import SciMLBase: SciMLProblem

abstract type DifferenceProblem <: SciMLProblem end
abstract type AbstractStateSpaceProblem{isinplace} <: DifferenceProblem end

include("problem.jl")
include("solution.jl")

end # module