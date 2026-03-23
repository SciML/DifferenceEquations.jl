# SciML-compatible init / solve! API
# Workspace holds pre-allocated output arrays + scratch cache.

"""
    StateSpaceWorkspace

Workspace for state-space problem solvers. Holds the problem, algorithm,
pre-allocated output arrays, and scratch cache.
Created by `CommonSolve.init` and consumed by `CommonSolve.solve!`.
"""
@concrete mutable struct StateSpaceWorkspace
    prob
    alg
    output   # pre-allocated output arrays (u, P, z) — NamedTuple
    cache    # scratch workspace buffers
end

"""
    CommonSolve.init(prob::AbstractStateSpaceProblem, alg=default_alg(prob); kwargs...)

Create a `StateSpaceWorkspace` with pre-allocated output arrays and scratch cache.
"""
function CommonSolve.init(prob::AbstractStateSpaceProblem, alg = default_alg(prob); kwargs...)
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    output = alloc_sol(prob, alg, T)
    cache = alloc_cache(prob, alg, T)
    return StateSpaceWorkspace(prob, alg, output, cache)
end

"""
    CommonSolve.solve!(ws::StateSpaceWorkspace; kwargs...)

Solve the state-space problem. Mutates `ws.output` arrays in place, then
wraps them in a `StateSpaceSolution` and returns it.
"""
function CommonSolve.solve!(ws::StateSpaceWorkspace; kwargs...)
    return _solve!(ws.prob, ws.alg, ws.output, ws.cache; kwargs...)
end
