
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
