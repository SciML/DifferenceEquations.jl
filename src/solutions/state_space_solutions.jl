"""
    StateSpaceSolution

Solution type returned by `solve` for all state-space problems.

# Fields
- `u`: State trajectory as `Vector{Vector{T}}`.
- `t`: Time values.
- `z`: Observation trajectory as `Vector{Vector{T}}`, or `nothing`.
- `W`: Noise sequence as `Vector{Vector{T}}`, or `nothing` (e.g., for `KalmanFilter`).
- `P`: Posterior covariances as `Vector{Matrix{T}}` (`KalmanFilter` only), or `nothing`.
- `logpdf`: Log-likelihood value. Zero when no `observables` are provided.
- `retcode`: `:Success` or `:Default`.
- `prob`: The original problem.
- `alg`: The algorithm used.

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
    retcode::Symbol
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
        retcode = :Default,
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

# Just using ConstantInterpolation for now.  Worth specializing?
# (sol::StateSpaceSolution)(t, ::Type{deriv} = Val{0}; idxs = nothing, continuity = :left) where {deriv} = _interpolate(sol, t, idxs)
# _interpolate(sol::StateSpaceSolution, t::Integer, idxs::Nothing) = sol.u[t]
# _interpolate(sol::StateSpaceSolution, t::Number, idxs::Nothing) = sol.u[Integer(round(t))]
# _interpolate(sol::StateSpaceSolution, t::Integer, idxs) = sol.u[t][idxs]

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
