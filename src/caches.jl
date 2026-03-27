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

# --- Quadratic solution output (same structure as linear) ---

function alloc_sol(prob::AnyQuadraticProblem, ::DirectIteration, T)
    (; u0, C_0) = prob
    M = isnothing(C_0) ? 0 : length(C_0)
    return (;
        u = [alloc_like(u0) for _ in 1:T],
        z = isnothing(C_0) ? nothing : [alloc_like(u0, M) for _ in 1:T],
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
    (; B, C, u0) = prob
    M = isnothing(C) ? 0 : size(C, 1)
    has_obs_noise = !isnothing(prob.observables_noise)
    return _alloc_di_base_cache(B, u0, M, T, has_obs_noise)
end

_alloc_noise(B, T) = [Vector{eltype(B)}(undef, size(B, 2)) for _ in 1:(T - 1)]
_alloc_noise(::Nothing, T) = nothing

# --- Shared base cache for DirectIteration (noise + loglik workspace) ---

function _alloc_di_base_cache(B, u0, M, T, has_obs_noise)
    T_obs = T - 1
    return (;
        noise = _alloc_noise(B, T),
        R = has_obs_noise ? alloc_like(u0, M, M) : nothing,
        R_chol = has_obs_noise ? alloc_like(u0, M, M) : nothing,
        innovation = has_obs_noise ? [alloc_like(u0, M) for _ in 1:T_obs] : nothing,
        innovation_solved = has_obs_noise ? [alloc_like(u0, M) for _ in 1:T_obs] : nothing,
    )
end

# --- Unpruned quadratic cache (same as linear) ---

function alloc_cache(prob::QuadraticStateSpaceProblem, ::DirectIteration, T)
    (; B, C_0, u0) = prob
    M = isnothing(C_0) ? 0 : length(C_0)
    has_obs_noise = !isnothing(prob.observables_noise)
    return _alloc_di_base_cache(B, u0, M, T, has_obs_noise)
end

# --- Pruned quadratic cache (base + u_f buffer) ---

function alloc_cache(prob::PrunedQuadraticStateSpaceProblem, ::DirectIteration, T)
    (; B, C_0, u0) = prob
    M = isnothing(C_0) ? 0 : length(C_0)
    has_obs_noise = !isnothing(prob.observables_noise)
    base = _alloc_di_base_cache(B, u0, M, T, has_obs_noise)
    u_f = [alloc_like(u0) for _ in 1:T]
    return (; base..., u_f)
end

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

