module DifferenceEquations

# using ChainRulesCore: ChainRulesCore, NoTangent, Tangent, ZeroTangent  # AD disabled — will restore with Enzyme
using CommonSolve: CommonSolve, solve, init, solve!
using DiffEqBase: DiffEqBase, DEProblem, get_concrete_u0, get_concrete_p, isconcreteu0,
    promote_u0
using LinearAlgebra: LinearAlgebra, Diagonal, NoPivot, Symmetric, cholesky,
    cholesky!, dot, ldiv!, mul!, transpose!
using SciMLBase: SciMLBase, @add_kwonly, NullParameters, promote_tspan, AbstractRODESolution,
    ODEFunction, remake, ConstantInterpolation, build_solution
using StaticArrays: StaticArrays, SVector, SMatrix, ismutable
using SymbolicIndexingInterface: SymbolicIndexingInterface, SymbolCache, variable_index

include("utilities_bangbang.jl")
include("utilities.jl")
include("problems/state_space_problems.jl")
include("solutions/state_space_solutions.jl")
include("solve.jl")
include("caches.jl")
include("workspace.jl")
include("algorithms/linear.jl")
include("algorithms/generic.jl")
include("precompilation.jl")

# Exports
export AbstractStateSpaceProblem, LinearStateSpaceProblem, StateSpaceProblem
export StateSpaceSolution, DirectIteration, KalmanFilter
export StateSpaceWorkspace

export solve, init, solve!

end # module
