# =============================================================================
# Model interface for DirectIteration dispatch
# Each problem type defines these methods to plug into the generic solver loop.
# =============================================================================

# --- Noise matrix extraction ---
_noise_matrix(prob::LinearStateSpaceProblem) = prob.B

# --- Cache noise access ---
_cache_noise(cache) = cache.noise

# --- Model-specific initialization (e.g., quadratic u_f) ---
_init_model_state!!(::LinearStateSpaceProblem, cache) = nothing

# --- Observation flag (does this problem have an observation equation?) ---
_has_observations(cache) = !isnothing(cache.z)

# =============================================================================
# Linear state-space callbacks
# =============================================================================

"""
    _transition!!(x_next, x, w, prob::LinearStateSpaceProblem, cache, t)

Linear state transition: `x_next = A * x + B * w`
"""
@inline function _transition!!(x_next, x, w, prob::LinearStateSpaceProblem, cache, t)
    x_next = mul!!(x_next, prob.A, x)
    x_next = muladd!!(x_next, prob.B, w)
    return x_next
end

"""
    _observation!!(y, x, prob::LinearStateSpaceProblem, cache, t)

Linear observation: `y = C * x`
"""
@inline function _observation!!(y, x, prob::LinearStateSpaceProblem, cache, t)
    y = mul!!(y, prob.C, x)
    return y
end

# =============================================================================
# Generic DirectIteration solver — single loop for all problem types
# =============================================================================

# Function barrier: _noise_matrix may return a union type for StateSpaceProblem
# (n_shocks is a runtime Int). Splitting here lets Julia specialize the hot loop
# on the concrete B type.
function _solve_with_cache!(
        prob::AbstractStateSpaceProblem, alg::DirectIteration, cache; kwargs...
    )
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    B = _noise_matrix(prob)
    return _solve_direct_iteration!(prob, alg, cache, B, T; kwargs...)
end

function _solve_direct_iteration!(prob, alg, cache, B, T; kwargs...)
    # Get concrete noise and copy into cache
    noise_concrete = get_concrete_noise(prob, prob.noise, B, T - 1)
    observables_noise = make_observables_noise(prob.observables_noise)

    # Validate dimensions
    if !isnothing(noise_concrete)
        @assert length(noise_concrete) == T - 1
        @assert length(noise_concrete[1]) == size(B, 2)
    end
    @assert maybe_check_size(prob.observables, 2, T - 1)

    (; u, z) = cache
    noise = _cache_noise(cache)

    # Copy noise into cache buffers
    if !isnothing(noise) && !isnothing(noise_concrete)
        copy_noise_to_cache!(noise, noise_concrete)
    end

    # Initialize state
    u[1] = assign!!(u[1], prob.u0)
    _init_model_state!!(prob, cache)

    # Initial observation
    if _has_observations(cache)
        z[1] = _observation!!(z[1], u[1], prob, cache, 1)
    end

    loglik = zero(eltype(prob.u0))
    @inbounds for t in 2:T
        w_t = isnothing(noise) ? nothing : noise[t - 1]
        u[t] = _transition!!(u[t], u[t - 1], w_t, prob, cache, t)

        if _has_observations(cache)
            z[t] = _observation!!(z[t], u[t], prob, cache, t)
        end
        loglik += maybe_logpdf(observables_noise, prob.observables, t - 1, z, t)
    end

    maybe_add_observation_noise!(z, observables_noise, prob.observables)
    t_values = prob.tspan[1]:prob.tspan[2]

    ObsType = typeof(prob.observables)
    return build_solution(
        prob, alg, t_values, u; W = noise_concrete,
        logpdf = ObsType <: Nothing ? nothing : loglik, z,
        retcode = :Success
    )
end

# Single __solve route for all problem types with DirectIteration
function DiffEqBase.__solve(
        prob::AbstractStateSpaceProblem, alg::DirectIteration, args...;
        kwargs...
    )
    ws = CommonSolve.init(prob, alg; kwargs...)
    return CommonSolve.solve!(ws; kwargs...)
end

# =============================================================================
# Enzyme AD-compatible DirectIteration loglik (standalone, no prob struct)
# =============================================================================

"""
    _direct_iteration_loglik!(A, B, C, u0, noise, observables, H, cache;
                              perturb_diagonal = 0.0)

Linear DirectIteration simulation with log-likelihood (Enzyme AD-compatible hot path).
Computes R = H*H' via `muladd!!`, factors it once, then uses Cholesky-based loglik
per timestep. Returns scalar log-likelihood.

# Arguments
- `A`: State transition matrix
- `B`: Noise input matrix (state noise)
- `C`: Observation matrix
- `u0`: Initial state
- `noise`: Concrete noise vectors (vector-of-vectors)
- `observables`: Observations (vector-of-vectors or matrix)
- `H`: Observation noise input matrix (M×L), R = H*H' computed internally
- `cache`: DirectIteration loglik cache from `alloc_direct_loglik_cache`
- `perturb_diagonal`: Diagonal perturbation for numerical stability
"""
function _direct_iteration_loglik!(A, B, C, u0, noise, observables, H, cache;
        perturb_diagonal = 0.0)
    (; u, z) = cache

    # Extract cache arrays once (avoids repeated named tuple field access in loop)
    R = cache.R
    R_chol_buf = cache.R_chol
    innovation_buf = cache.innovation
    innovation_solved_buf = cache.innovation_solved

    # Compute R = H*H' and Cholesky factorize (once, outside loop)
    R = mul!!(R, H, transpose(H))
    R_chol_buf = symmetrize_upper!!(R_chol_buf, R, perturb_diagonal)
    F = cholesky!!(R_chol_buf, :U)

    # Precompute constant term
    M_obs = size(C, 1)
    logdetR = logdet_chol(F)
    log_const = M_obs * log(2π) + logdetR

    # Initialize state
    u[1] = copyto!!(u[1], u0)

    T = length(noise)
    loglik = zero(eltype(u0))
    is_mutable = ismutable(u[1])

    @inbounds for t in 1:T
        # Linear transition: x_{t+1} = A * x_t + B * w_t
        u[t + 1] = mul!!(u[t + 1], A, u[t])
        u[t + 1] = muladd!!(u[t + 1], B, noise[t])

        # Predicted observation: z = C * x
        z[t] = mul!!(z[t], C, u[t])

        # Innovation: ν = obs_t - z_t
        ν = innovation_buf[t]
        ν = copyto!!(ν, get_observable(observables, t))
        if is_mutable
            for i in eachindex(ν)
                ν[i] -= z[t][i]
            end
        else
            ν = ν - z[t]
        end
        innovation_buf[t] = ν

        # Quadratic form: ν' * R⁻¹ * ν
        ν_solved = innovation_solved_buf[t]
        ν_solved = ldiv!!(ν_solved, F, ν)
        innovation_solved_buf[t] = ν_solved
        quad = dot(ν, ν_solved)

        loglik -= 0.5 * (log_const + quad)
    end

    return loglik
end

# =============================================================================
# KalmanFilter solver — specific to LinearStateSpaceProblem
# =============================================================================

"""
    _kalman_loglik!(A, B, C, u0_prior_mean, u0_prior_var, R, observables, cache;
                    perturb_diagonal = 0.0)

Kalman filter log-likelihood on individual arrays (Enzyme AD-compatible hot path).
Returns only the scalar log-likelihood. No exceptions, no solution construction.

# Arguments
- `A`: State transition matrix
- `B`: State noise input matrix
- `C`: Observation matrix
- `u0_prior_mean`: Prior mean
- `u0_prior_var`: Prior covariance
- `R`: Observation noise covariance (precomputed)
- `observables`: Observations (vector-of-vectors or matrix)
- `cache`: Kalman cache from `alloc_kalman_cache`
- `perturb_diagonal`: Diagonal perturbation for numerical stability
"""
function _kalman_loglik!(A, B, C, u0_prior_mean, u0_prior_var, R, observables, cache;
        perturb_diagonal = 0.0)
    (; u, P, z, B_prod) = cache

    # Compute B*B' once
    B_prod = mul!!(B_prod, B, transpose(B))

    # Initialize
    u[1] = copyto!!(u[1], u0_prior_mean)
    P[1] = copyto!!(P[1], u0_prior_var)
    z[1] = mul!!(z[1], C, u[1])

    loglik = zero(eltype(u0_prior_var))
    is_mutable = ismutable(u[1])
    T_obs = length(cache.mu_pred)
    M_obs = size(C, 1)
    log_const_kf = M_obs * log(2π)

    @inbounds for t in 1:T_obs
        # Get cache buffers for this timestep
        μp = cache.mu_pred[t]
        Σp = cache.sigma_pred[t]
        AΣ = cache.A_sigma[t]
        ΣGt = cache.sigma_Gt[t]
        ν = cache.innovation[t]
        S = cache.innovation_cov[t]
        S_chol_buf = cache.S_chol[t]
        ν_solved = cache.innovation_solved[t]
        rhs = cache.gain_rhs[t]
        K_t = cache.gain[t]
        KG = cache.gainG[t]
        KGS = cache.KgSigma[t]
        μu = cache.mu_update[t]

        # Current state
        μt = u[t]
        Σt = P[t]

        # Predict mean: μp = A * μt
        μp = mul!!(μp, A, μt)

        # Predict covariance: Σp = A * Σt * A' + B * B'
        AΣ = mul!!(AΣ, A, Σt)
        Σp = mul!!(Σp, AΣ, transpose(A))
        if is_mutable
            @inbounds for i in eachindex(Σp)
                Σp[i] += B_prod[i]
            end
        else
            Σp = Σp + B_prod
        end

        # Predicted observation: z[t+1] = C * μp
        z[t + 1] = mul!!(z[t + 1], C, μp)

        # Innovation: ν = observables[t] - z[t+1]
        obs_t = get_observable(observables, t)
        ν = copyto!!(ν, obs_t)
        ν = mul!!(ν, C, μp, -1.0, 1.0)

        # Innovation covariance: S = C * Σp * C' + R
        ΣGt = mul!!(ΣGt, Σp, transpose(C))
        S = mul!!(S, C, ΣGt)
        if is_mutable
            @inbounds for i in eachindex(S)
                S[i] += R[i]
            end
        else
            S = S + R
        end

        # Symmetrize and Cholesky
        S_chol_buf = symmetrize_upper!!(S_chol_buf, S, perturb_diagonal)
        F = cholesky!!(S_chol_buf, :U)

        # Kalman gain: K = Σp * C' * S^{-1}
        rhs = transpose!!(rhs, ΣGt)
        rhs = ldiv!!(F, rhs)
        K_t = transpose!!(K_t, rhs)

        # Update mean: u[t+1] = μp + K * ν
        μu = mul!!(μu, K_t, ν)
        if is_mutable
            @inbounds for i in eachindex(μp)
                u[t + 1][i] = μp[i] + μu[i]
            end
        else
            cache.mu_pred[t] = μp
            cache.mu_update[t] = μu
            u[t + 1] = μp + μu
        end

        # Update covariance: P[t+1] = Σp - K * C * Σp
        KG = mul!!(KG, K_t, C)
        KGS = mul!!(KGS, KG, Σp)
        if is_mutable
            @inbounds for i in eachindex(Σp)
                P[t + 1][i] = Σp[i] - KGS[i]
            end
        else
            cache.sigma_pred[t] = Σp
            cache.KgSigma[t] = KGS
            P[t + 1] = Σp - KGS
        end

        # Log-likelihood contribution (allocation-free)
        ν_solved = ldiv!!(ν_solved, F, ν)
        cache.innovation[t] = ν
        cache.innovation_solved[t] = ν_solved
        logdetS = logdet_chol(F)
        quad = dot(ν_solved, ν)
        loglik -= 0.5 * (log_const_kf + logdetS + quad)
    end

    return loglik
end

function _solve_with_cache!(
        prob::LinearStateSpaceProblem, alg::KalmanFilter, cache;
        perturb_diagonal = 0.0, kwargs...
    )
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    @assert size(prob.observables, 2) == T - 1

    (; A, B, C, u0_prior_mean, u0_prior_var) = prob
    R = make_observables_covariance_matrix(prob.observables_noise)

    # Error handling kept in the SciML wrapper (not in the Enzyme hot path)
    retcode = :Failure
    loglik = convert(eltype(u0_prior_var), -Inf)
    try
        loglik = _kalman_loglik!(A, B, C, u0_prior_mean, u0_prior_var, R,
            prob.observables, cache; perturb_diagonal)
        retcode = :Success
    catch
        loglik = convert(eltype(u0_prior_var), -Inf)
    end

    t_values = prob.tspan[1]:prob.tspan[2]
    return build_solution(
        prob, alg, t_values, cache.u; P = cache.P, W = nothing, logpdf = loglik,
        z = cache.z, retcode
    )
end

function DiffEqBase.__solve(
        prob::LinearStateSpaceProblem, alg::KalmanFilter, args...;
        kwargs...
    )
    ws = CommonSolve.init(prob, alg; kwargs...)
    return CommonSolve.solve!(ws; kwargs...)
end
