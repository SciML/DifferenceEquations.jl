# SciML-compatible init / solve! API for cache reuse across solves

"""
    StateSpaceWorkspace{P, A, C}

Workspace for state-space problem solvers, holding the problem, algorithm, and
preallocated cache. Created by `CommonSolve.init` and consumed by `CommonSolve.solve!`.
"""
mutable struct StateSpaceWorkspace{P, A, C}
    prob::P
    alg::A
    cache::C
end

"""
    CommonSolve.init(prob::AbstractStateSpaceProblem, alg=default_alg(prob); kwargs...)

Create a `StateSpaceWorkspace` with preallocated cache for the given problem and algorithm.
The workspace can be reused across multiple `solve!` calls.
"""
function CommonSolve.init(prob::AbstractStateSpaceProblem, alg = default_alg(prob); kwargs...)
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    cache = alloc_cache(prob, alg, T)
    return StateSpaceWorkspace(prob, alg, cache)
end

"""
    CommonSolve.solve!(ws::StateSpaceWorkspace; kwargs...)

Solve the state-space problem using the preallocated workspace. Can be called
repeatedly on the same workspace for cache reuse.
"""
function CommonSolve.solve!(ws::StateSpaceWorkspace; kwargs...)
    return _solve_with_cache!(ws.prob, ws.alg, ws.cache; kwargs...)
end
