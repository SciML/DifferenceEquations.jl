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
    save_everystep::Bool
end

# Public 4-arg constructor — assumes save_everystep=true (full trajectory storage).
# This is the form used by Enzyme wrappers and direct workspace construction.
# The 5-arg form with the Bool is only called internally by init().
function StateSpaceWorkspace(prob, alg, output, cache)
    return StateSpaceWorkspace(prob, alg, output, cache, true)
end

"""
    CommonSolve.init(prob::AbstractStateSpaceProblem, alg=default_alg(prob); save_everystep=true, kwargs...)

Create a `StateSpaceWorkspace` with pre-allocated output arrays and scratch cache.
When `save_everystep=false`, allocates minimal 2-element buffers (endpoints only).
"""
function CommonSolve.init(
        prob::AbstractStateSpaceProblem, alg = default_alg(prob);
        save_everystep = true, kwargs...
    )
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    if save_everystep
        output = alloc_sol(prob, alg, T)
        cache = alloc_cache(prob, alg, T)
    else
        se = Val(false)
        output = alloc_sol(prob, alg, T, se)
        cache = alloc_cache(prob, alg, T, se)
    end
    return StateSpaceWorkspace(prob, alg, output, cache, save_everystep)
end

"""
    CommonSolve.solve!(ws::StateSpaceWorkspace; kwargs...)

Solve the state-space problem. Mutates `ws.output` arrays in place, then
wraps them in a `StateSpaceSolution` and returns it.
"""
function CommonSolve.solve!(ws::StateSpaceWorkspace; kwargs...)
    if ws.save_everystep
        return _solve!(ws.prob, ws.alg, ws.output, ws.cache; kwargs...)
    else
        return _solve!(
            ws.prob, ws.alg, ws.output, ws.cache;
            save_everystep = Val(false), kwargs...
        )
    end
end
