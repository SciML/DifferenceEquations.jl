# SciML-compatible init / solve! API
# Workspace holds pre-allocated solution output + scratch cache (like ODE integrators).

"""
    StateSpaceWorkspace{P, A, S, C}

Workspace for state-space problem solvers. Holds the problem, algorithm,
pre-allocated solution arrays (output), and scratch cache (temporary buffers).
Created by `CommonSolve.init` and consumed by `CommonSolve.solve!`.
"""
mutable struct StateSpaceWorkspace{P, A, S, C}
    prob::P
    alg::A
    sol::S     # pre-allocated solution output (u, P, z)
    cache::C   # scratch workspace buffers
end

"""
    CommonSolve.init(prob::AbstractStateSpaceProblem, alg=default_alg(prob); kwargs...)

Create a `StateSpaceWorkspace` with pre-allocated solution and scratch cache.
"""
function CommonSolve.init(prob::AbstractStateSpaceProblem, alg = default_alg(prob); kwargs...)
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    sol = alloc_sol(prob, alg, T)
    cache = alloc_cache(prob, alg, T)
    return StateSpaceWorkspace(prob, alg, sol, cache)
end

"""
    CommonSolve.solve!(ws::StateSpaceWorkspace; kwargs...)

Solve the state-space problem. Writes results into `ws.sol` and returns
a `StateSpaceSolution` wrapping the output.
"""
function CommonSolve.solve!(ws::StateSpaceWorkspace; kwargs...)
    return _solve!(ws.prob, ws.alg, ws.sol, ws.cache; kwargs...)
end
