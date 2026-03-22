# Cache allocation functions for preallocated workspace buffers
# Named-tuple caches, vector-of-vectors storage

# =============================================================================
# Linear DirectIteration cache
# =============================================================================

"""
    alloc_direct_cache(prob::LinearStateSpaceProblem, T)

Allocate cache for linear DirectIteration solver. Returns a named tuple with
preallocated vector-of-vectors for states, observations, and noise.
"""
function alloc_direct_cache(prob::LinearStateSpaceProblem, T)
    (; A, B, C, u0) = prob
    return alloc_direct_cache(u0, A, B, C, T)
end

function alloc_direct_cache(u0, A, B, C, T)
    return (;
        u = [alloc_like(u0) for _ in 1:T],
        z = isnothing(C) ? nothing : [alloc_like(u0, size(C, 1)) for _ in 1:T],
        noise = _alloc_noise(B, T)
    )
end

_alloc_noise(B, T) = [Vector{eltype(B)}(undef, size(B, 2)) for _ in 1:(T - 1)]
_alloc_noise(::Nothing, T) = nothing

"""
    alloc_direct_loglik_cache(u0, A, B, C, H, T)

Allocate cache for DirectIteration with log-likelihood computation (Enzyme AD path).
Extends the basic DI cache with R = H*H' and Cholesky workspace.

# Arguments
- `H`: Observation noise input matrix (M×L), used as prototype for R buffers
"""
function alloc_direct_loglik_cache(u0, A, B, C, H, T)
    M = size(C, 1)
    L_noise = size(H, 2)
    T_obs = T - 1
    return (;
        u = [alloc_like(u0) for _ in 1:T],
        z = [alloc_like(u0, M) for _ in 1:T_obs],
        # Loglik workspace: R = H*H' computed at runtime, then Cholesky factored
        R = alloc_like(H, M, M),
        R_chol = alloc_like(H, M, M),
        H_t = alloc_like(H, L_noise, M),  # transpose buffer for mul_aat!! workaround
        innovation = [alloc_like(u0, M) for _ in 1:T_obs],
        innovation_solved = [alloc_like(u0, M) for _ in 1:T_obs]
    )
end

"""
    zero_direct_loglik_cache!!(cache)

Zero all buffers in a DirectIteration loglik cache for Enzyme AD compatibility.
"""
function zero_direct_loglik_cache!!(cache)
    @inbounds for t in eachindex(cache.u)
        cache.u[t] = fill_zero!!(cache.u[t])
    end
    @inbounds for t in eachindex(cache.z)
        cache.z[t] = fill_zero!!(cache.z[t])
    end
    fill_zero!!(cache.R)
    fill_zero!!(cache.R_chol)
    fill_zero!!(cache.H_t)
    @inbounds for t in eachindex(cache.innovation)
        cache.innovation[t] = fill_zero!!(cache.innovation[t])
        cache.innovation_solved[t] = fill_zero!!(cache.innovation_solved[t])
    end
    return cache
end

"""
    zero_direct_cache!!(cache)

Zero all buffers in a direct iteration cache for Enzyme AD compatibility.
"""
function zero_direct_cache!!(cache)
    @inbounds for t in eachindex(cache.u)
        cache.u[t] = fill_zero!!(cache.u[t])
    end
    if !isnothing(cache.z)
        @inbounds for t in eachindex(cache.z)
            cache.z[t] = fill_zero!!(cache.z[t])
        end
    end
    if !isnothing(cache.noise)
        @inbounds for t in eachindex(cache.noise)
            cache.noise[t] = fill_zero!!(cache.noise[t])
        end
    end
    return cache
end

# =============================================================================
# Kalman filter cache
# =============================================================================

"""
    alloc_kalman_cache(prob::LinearStateSpaceProblem, T)

Allocate cache for Kalman filter with all workspace arrays as vectors of vectors/matrices.
"""
function alloc_kalman_cache(prob::LinearStateSpaceProblem, T)
    (; A, B, C, u0_prior_mean, u0_prior_var) = prob
    N = length(u0_prior_mean)
    L = size(C, 1)
    T_obs = T - 1  # number of observation timesteps in Kalman loop

    # B_prod = B * B' computed once and stored
    K_noise = size(B, 2)
    B_prod = alloc_like(u0_prior_var)
    B_t = alloc_like(B, K_noise, N)  # transpose buffer for mul_aat!! workaround

    return (;
        # Per-timestep workspace buffers (T_obs entries for the Kalman loop)
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
        # Output arrays
        u = [alloc_like(u0_prior_mean) for _ in 1:T],
        P = [alloc_like(u0_prior_var) for _ in 1:T],
        z = [alloc_like(u0_prior_mean, L) for _ in 1:T],
        # Precomputed matrices
        B_prod,
        B_t
    )
end

"""
    zero_kalman_cache!!(cache)

Zero all buffers in a Kalman filter cache for Enzyme AD compatibility.
"""
function zero_kalman_cache!!(cache)
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
    T = length(cache.u)
    @inbounds for t in 1:T
        cache.u[t] = fill_zero!!(cache.u[t])
        cache.P[t] = fill_zero!!(cache.P[t])
        cache.z[t] = fill_zero!!(cache.z[t])
    end
    fill_zero!!(cache.B_prod)
    fill_zero!!(cache.B_t)
    return cache
end

# =============================================================================
# Cache dispatch
# =============================================================================

"""
    alloc_cache(prob, alg, T)

Dispatch to the appropriate cache allocation function based on problem and algorithm types.
"""
alloc_cache(prob::LinearStateSpaceProblem, ::DirectIteration, T) =
    alloc_direct_cache(prob, T)
alloc_cache(prob::LinearStateSpaceProblem, ::KalmanFilter, T) =
    alloc_kalman_cache(prob, T)

function alloc_cache(prob::StateSpaceProblem, ::DirectIteration, T)
    (; u0, n_obs) = prob
    B = _noise_matrix(prob)
    return (;
        u = [alloc_like(u0) for _ in 1:T],
        z = n_obs > 0 ? [alloc_like(u0, n_obs) for _ in 1:T] : nothing,
        noise = _alloc_noise(B, T)
    )
end
