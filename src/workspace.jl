# SciML-compatible init / solve! API
# Workspace holds pre-allocated output arrays + scratch cache.

"""
    StateSpaceWorkspace(prob, alg, output, cache)

Workspace for state-space problem solvers. Holds the problem, algorithm,
pre-allocated output arrays, and scratch cache. Created by [`init`](@ref) and
consumed by [`solve!`](@ref).

# Arguments

- `prob::AbstractStateSpaceProblem`: Problem to solve.
- `alg`: Difference-equation algorithm.
- `output`: Pre-allocated output buffers owned by the workspace.
- `cache`: Scratch buffers used by the solver.

# Fields

- `prob`: Problem solved by the workspace.
- `alg`: Algorithm used by [`solve!`](@ref).
- `output`: Named tuple of pre-allocated state, observation, and covariance buffers.
- `cache`: Scratch arrays and factorizations reused across solves.
- `save_everystep::Bool`: Whether the full trajectory is retained.

# Returns

- `StateSpaceWorkspace`: A reusable workspace for repeated solves.

# Interface

Construct workspaces with [`init`](@ref) unless an external AD or allocation
pipeline already owns compatible `output` and `cache` buffers. Call [`solve!`](@ref)
to overwrite the buffers and return a new [`StateSpaceSolution`](@ref).
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
    CommonSolve.init(prob::AbstractStateSpaceProblem, alg = default_alg(prob); save_everystep = true, kwargs...)

Create a `StateSpaceWorkspace` with pre-allocated output arrays and scratch cache.
When `save_everystep=false`, allocate minimal two-element buffers containing only
the initial and final states.

# Arguments

- `prob::AbstractStateSpaceProblem`: Problem to solve repeatedly.
- `alg`: Algorithm; defaults to the problem-dependent algorithm selected by
  `default_alg`.

# Keyword Arguments

- `save_everystep::Bool = true`: Store the full trajectory when `true`; retain only
  endpoint buffers when `false`.
- `kwargs...`: Additional allocation options forwarded to the workspace setup.

# Returns

- `StateSpaceWorkspace`: A reusable workspace consumed by [`solve!`](@ref).
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

Solve the state-space problem. Mutate `ws.output` arrays in place, then wrap them
in a [`StateSpaceSolution`](@ref) and return it.

# Arguments

- `ws::StateSpaceWorkspace`: Workspace created by [`init`](@ref).

# Keyword Arguments

- `kwargs...`: Solver options forwarded to the selected state-space algorithm.

# Returns

- `StateSpaceSolution`: The solution backed by the workspace's latest output.
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
