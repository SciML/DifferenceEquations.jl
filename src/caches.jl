# Cache allocation: pre-allocated solution output + scratch workspace buffers
# Named-tuple storage, vector-of-vectors format

# =============================================================================
# Solution output allocation (u, P, z — owned by workspace, returned to user)
# =============================================================================

"""
    alloc_sol(prob::LinearStateSpaceProblem, ::DirectIteration, T)

Allocate solution output arrays for DirectIteration.
"""
function alloc_sol(prob::LinearStateSpaceProblem, ::DirectIteration, T)
    (; u0, C) = prob
    M = isnothing(C) ? 0 : size(C, 1)
    return (;
        u = [alloc_like(u0) for _ in 1:T],
        z = isnothing(C) ? nothing : [alloc_like(u0, M) for _ in 1:T],
    )
end

"""
    alloc_sol(prob::LinearStateSpaceProblem, ::KalmanFilter, T)

Allocate solution output arrays for KalmanFilter (filtered means, covariances, observations).
"""
function alloc_sol(prob::LinearStateSpaceProblem, ::KalmanFilter, T)
    (; u0_prior_mean, u0_prior_var, C) = prob
    L = size(C, 1)
    return (;
        u = [alloc_like(u0_prior_mean) for _ in 1:T],
        P = [alloc_like(u0_prior_var) for _ in 1:T],
        z = [alloc_like(u0_prior_mean, L) for _ in 1:T],
    )
end

"""
    alloc_sol(prob::StateSpaceProblem, ::DirectIteration, T)

Allocate solution output arrays for generic StateSpaceProblem.
"""
function alloc_sol(prob::StateSpaceProblem, ::DirectIteration, T)
    (; u0, n_obs) = prob
    return (;
        u = [alloc_like(u0) for _ in 1:T],
        z = n_obs > 0 ? [alloc_like(u0, n_obs) for _ in 1:T] : nothing,
    )
end

# =============================================================================
# Scratch cache allocation (temporary workspace buffers only)
# =============================================================================

"""
    alloc_cache(prob::LinearStateSpaceProblem, ::DirectIteration, T)

Allocate scratch workspace for DirectIteration (noise buffers, loglik workspace).
"""
function alloc_cache(prob::LinearStateSpaceProblem, ::DirectIteration, T)
    (; A, B, C, u0) = prob
    M = isnothing(C) ? 0 : size(C, 1)
    T_obs = T - 1
    has_obs_noise = !isnothing(prob.observables_noise)
    return (;
        noise = _alloc_noise(B, T),
        R = has_obs_noise ? alloc_like(u0, M, M) : nothing,
        R_chol = has_obs_noise ? alloc_like(u0, M, M) : nothing,
        innovation = has_obs_noise ? [alloc_like(u0, M) for _ in 1:T_obs] : nothing,
        innovation_solved = has_obs_noise ? [alloc_like(u0, M) for _ in 1:T_obs] : nothing,
    )
end

_alloc_noise(B, T) = [Vector{eltype(B)}(undef, size(B, 2)) for _ in 1:(T - 1)]
_alloc_noise(::Nothing, T) = nothing

"""
    alloc_cache(prob::LinearStateSpaceProblem, ::KalmanFilter, T)

Allocate scratch workspace for KalmanFilter (prediction, innovation, gain buffers).
"""
function alloc_cache(prob::LinearStateSpaceProblem, ::KalmanFilter, T)
    (; A, B, C, u0_prior_mean, u0_prior_var) = prob
    N = length(u0_prior_mean)
    L = size(C, 1)
    T_obs = T - 1
    K_noise = size(B, 2)

    return (;
        mu_pred = [alloc_like(u0_prior_mean) for _ in 1:T_obs],
        sigma_pred = [alloc_like(u0_prior_var) for _ in 1:T_obs],
        A_sigma = [alloc_like(u0_prior_var) for _ in 1:T_obs],
        sigma_Gt = [alloc_like(u0_prior_var, N, L) for _ in 1:T_obs],
        innovation = [alloc_like(u0_prior_mean, L) for _ in 1:T_obs],
        innovation_cov = [alloc_like(u0_prior_var, L, L) for _ in 1:T_obs],
        S_chol = [alloc_like(u0_prior_var, L, L) for _ in 1:T_obs],
        innovation_solved = [alloc_like(u0_prior_mean, L) for _ in 1:T_obs],
        gain_rhs = [alloc_like(C) for _ in 1:T_obs],
        gain = [alloc_like(u0_prior_var, N, L) for _ in 1:T_obs],
        gainG = [alloc_like(u0_prior_var) for _ in 1:T_obs],
        KgSigma = [alloc_like(u0_prior_var) for _ in 1:T_obs],
        mu_update = [alloc_like(u0_prior_mean) for _ in 1:T_obs],
        B_prod = alloc_like(u0_prior_var),
        B_t = alloc_like(B, K_noise, N),
    )
end

"""
    alloc_cache(prob::StateSpaceProblem, ::DirectIteration, T)

Allocate scratch workspace for generic StateSpaceProblem.
"""
function alloc_cache(prob::StateSpaceProblem, ::DirectIteration, T)
    (; u0, n_obs) = prob
    B = _noise_matrix(prob)
    M = n_obs
    T_obs = T - 1
    has_obs_noise = !isnothing(prob.observables_noise) && M > 0
    return (;
        noise = _alloc_noise(B, T),
        R = has_obs_noise ? alloc_like(u0, M, M) : nothing,
        R_chol = has_obs_noise ? alloc_like(u0, M, M) : nothing,
        innovation = has_obs_noise ? [alloc_like(u0, M) for _ in 1:T_obs] : nothing,
        innovation_solved = has_obs_noise ? [alloc_like(u0, M) for _ in 1:T_obs] : nothing,
    )
end

# =============================================================================
# Cache zeroing for Enzyme AD
# =============================================================================

"""
    zero_cache!!(cache)

Zero all scratch buffers for Enzyme AD compatibility.
"""
function zero_cache!!(cache, ::DirectIteration)
    if !isnothing(cache.noise)
        @inbounds for t in eachindex(cache.noise)
            cache.noise[t] = fill_zero!!(cache.noise[t])
        end
    end
    if !isnothing(cache.R)
        fill_zero!!(cache.R)
        fill_zero!!(cache.R_chol)
        @inbounds for t in eachindex(cache.innovation)
            cache.innovation[t] = fill_zero!!(cache.innovation[t])
            cache.innovation_solved[t] = fill_zero!!(cache.innovation_solved[t])
        end
    end
    return cache
end

function zero_cache!!(cache, ::KalmanFilter)
    T_obs = length(cache.mu_pred)
    @inbounds for t in 1:T_obs
        cache.mu_pred[t] = fill_zero!!(cache.mu_pred[t])
        cache.sigma_pred[t] = fill_zero!!(cache.sigma_pred[t])
        cache.A_sigma[t] = fill_zero!!(cache.A_sigma[t])
        cache.sigma_Gt[t] = fill_zero!!(cache.sigma_Gt[t])
        cache.innovation[t] = fill_zero!!(cache.innovation[t])
        cache.innovation_cov[t] = fill_zero!!(cache.innovation_cov[t])
        cache.S_chol[t] = fill_zero!!(cache.S_chol[t])
        cache.innovation_solved[t] = fill_zero!!(cache.innovation_solved[t])
        cache.gain_rhs[t] = fill_zero!!(cache.gain_rhs[t])
        cache.gain[t] = fill_zero!!(cache.gain[t])
        cache.gainG[t] = fill_zero!!(cache.gainG[t])
        cache.KgSigma[t] = fill_zero!!(cache.KgSigma[t])
        cache.mu_update[t] = fill_zero!!(cache.mu_update[t])
    end
    fill_zero!!(cache.B_prod)
    fill_zero!!(cache.B_t)
    return cache
end

"""
    zero_sol!!(sol)

Zero all solution output arrays for Enzyme AD compatibility.
"""
function zero_sol!!(sol)
    @inbounds for t in eachindex(sol.u)
        sol.u[t] = fill_zero!!(sol.u[t])
    end
    if !isnothing(sol.z)
        @inbounds for t in eachindex(sol.z)
            sol.z[t] = fill_zero!!(sol.z[t])
        end
    end
    if hasproperty(sol, :P)
        @inbounds for t in eachindex(sol.P)
            sol.P[t] = fill_zero!!(sol.P[t])
        end
    end
    return sol
end
