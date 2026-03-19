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

# Function barrier: _noise_matrix may return a union type for GenericStateSpaceProblem
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
# KalmanFilter solver — specific to LinearStateSpaceProblem
# =============================================================================

function _solve_with_cache!(
        prob::LinearStateSpaceProblem, alg::KalmanFilter, cache;
        perturb_diagonal = 0.0, kwargs...
    )
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)

    @assert size(prob.observables, 2) == T - 1

    (; A, B, C, u0_prior_mean, u0_prior_var) = prob

    R = make_observables_covariance_matrix(prob.observables_noise)

    # Zero cache for Enzyme AD compatibility
    zero_kalman_cache!!(cache)

    (; u, P, z, B_prod) = cache

    # Compute B*B' once
    B_prod = mul!!(B_prod, B, transpose(B))

    # Initialize
    u[1] = copyto!!(u[1], u0_prior_mean)
    P[1] = copyto!!(P[1], u0_prior_var)
    z[1] = mul!!(z[1], C, u[1])

    loglik = zero(eltype(u0_prior_var))
    is_mutable = ismutable(u[1])
    T_obs = T - 1

    retcode = :Failure
    try
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
            obs_t = get_observable(prob.observables, t)
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
            M_obs = length(ν)
            quad = dot(ν_solved, ν)
            loglik -= 0.5 * (M_obs * log(2π) + logdetS + quad)
        end
        retcode = :Success
    catch e
        loglik = convert(typeof(loglik), -Inf)
    end

    t_values = prob.tspan[1]:prob.tspan[2]
    return build_solution(
        prob, alg, t_values, u; P, W = nothing, logpdf = loglik, z,
        retcode
    )
end

function DiffEqBase.__solve(
        prob::LinearStateSpaceProblem, alg::KalmanFilter, args...;
        kwargs...
    )
    ws = CommonSolve.init(prob, alg; kwargs...)
    return CommonSolve.solve!(ws; kwargs...)
end
