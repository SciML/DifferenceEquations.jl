"""
    StateSpaceSolution

Solution type returned by `solve` for all state-space problems.

# Fields
- `u`: State trajectory as `Vector{Vector{T}}` or a compatible collection.
- `u_analytic`: Analytic state trajectory, currently `nothing`.
- `errors`: Per-time errors, currently `nothing`.
- `t`: Time values, with one entry per saved state.
- `W`: Noise sequence, or `nothing` when no concrete noise was used.
- `prob`: The original [`AbstractStateSpaceProblem`](@ref).
- `alg`: The algorithm used to construct the solution.
- `interp`: Constant interpolation over the saved states.
- `dense`: Whether dense output is available.
- `tslocation`: Current time-series location used by SciMLBase indexing.
- `stats`: Solver statistics, when provided.
- `retcode`: Solver return code. Runtime errors are thrown as exceptions.
- `P`: Posterior covariances for [`KalmanFilter`](@ref), or `nothing` otherwise.
- `logpdf`: Log-likelihood value, zero when no observations are provided.
- `z`: Observation trajectory, or `nothing` when no observation equation exists.

# Returns

- `StateSpaceSolution`: A SciML-compatible solution containing states, observations,
  noise, and optional filtering statistics.

# Examples

# Symbolic Indexing
Access time series by symbol name:
```julia
sol[:x]      # state variable time series (requires `syms`)
sol[:output] # observation time series (requires `obs_syms`)
```

# Standard Indexing
```julia
sol[i]       # state at time step i (same as sol.u[i])
sol[end]     # final state
```
"""
struct StateSpaceSolution{
        T, N, uType, uType2, DType, tType, randType, P, A, IType, DE,
        PosteriorType,
        logpdfType, zType,
    } <: AbstractRODESolution{T, N, uType}
    u::uType
    u_analytic::uType2
    errors::DType
    t::tType
    W::randType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    stats::DE
    retcode::SciMLBase.ReturnCode.T
    P::PosteriorType
    logpdf::logpdfType
    z::zType
end

function SciMLBase.build_solution(
        prob::AbstractStateSpaceProblem, alg, t, u; P = nothing,
        logpdf = nothing,
        W = nothing, timeseries_errors = length(u) > 2,
        dense = false,
        dense_errors = dense, calculate_error = true,
        interp = ConstantInterpolation(t, u),
        retcode = ReturnCode.Default,
        stats = nothing, z = nothing, kwargs...
    )
    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    # TODO: add support for has_analytic in the future
    sol = StateSpaceSolution{
        T, N, typeof(u), Nothing, Nothing, typeof(t), typeof(W),
        typeof(prob),
        typeof(alg), typeof(interp), typeof(stats), typeof(P),
        typeof(logpdf),
        typeof(z),
    }(
        u, nothing, nothing, t, W, prob, alg, interp, dense,
        0,
        stats, retcode, P, logpdf, z
    )
    return sol
end

# TODO: Worth specializing interpolation beyond ConstantInterpolation?

"""Return observation symbols from the problem, or nothing."""
obs_syms(sol::StateSpaceSolution) = sol.prob.obs_syms

Base.@propagate_inbounds function Base.getindex(sol::StateSpaceSolution, sym::Symbol)
    # Check observation symbols first
    _obs_syms = sol.prob.obs_syms
    if _obs_syms !== nothing
        idx = findfirst(==(sym), _obs_syms)
        if idx !== nothing
            sol.z === nothing &&
                error("Observation symbol $sym found but no observations in solution")
            return [sol.z[t][idx] for t in eachindex(sol.z)]
        end
    end
    # Check state symbols via the ODEFunction's SymbolCache
    state_idx = variable_index(sol.prob.f.sys, sym)
    if state_idx !== nothing
        return [sol.u[t][state_idx] for t in eachindex(sol.u)]
    end
    throw(ArgumentError("Symbol $sym not found in state or observation symbols"))
end

# For recipes
SciMLBase.getindepsym(sol::StateSpaceSolution) = :t
SciMLBase.getindepsym_defaultt(sol::StateSpaceSolution) = :t
