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

# --- Observation flag ---
_has_observations(sol) = !isnothing(sol.z)

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

# --- Default 7-arg fallbacks for save_everystep=false endpoints loops ---
# PrunedQuadratic overrides these; all other problem types fall through.
@inline _transition!!(x_next, x, w, prob, cache, t, ::Val) =
    _transition!!(x_next, x, w, prob, cache, t)
@inline _observation!!(y, x, prob, cache, t, ::Val) =
    _observation!!(y, x, prob, cache, t)
@inline _init_model_state!!(prob, cache, ::Val) =
    _init_model_state!!(prob, cache)

# Function barrier: _noise_matrix may return a union type for StateSpaceProblem
# (n_shocks is a runtime Int). Splitting here lets Julia specialize the hot loop
# on the concrete B type.
function _solve!(
        prob::AbstractStateSpaceProblem, alg::DirectIteration, sol, cache;
        save_everystep::Val{SE} = Val(true), kwargs...
    ) where {SE}
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    B = _noise_matrix(prob)
    if SE
        return _solve_direct_iteration!(prob, alg, sol, cache, B, T; kwargs...)
    else
        return _solve_direct_iteration_endpoints!(prob, alg, sol, cache, B, T; kwargs...)
    end
end

function _solve_direct_iteration!(
        prob, alg, sol, cache, B, T;
        perturb_diagonal = 0.0, kwargs...
    )
    # Get concrete noise and copy into cache
    noise_concrete = get_concrete_noise(prob, prob.noise, B, T - 1)

    # Validate dimensions
    if !isnothing(noise_concrete)
        length(noise_concrete) == T - 1 ||
            throw(ArgumentError("noise length $(length(noise_concrete)) must equal T-1 = $(T - 1)"))
        length(noise_concrete[1]) == size(B, 2) ||
            throw(ArgumentError("noise dimension $(length(noise_concrete[1])) must equal number of shocks $(size(B, 2))"))
    end
    maybe_check_size(prob.observables, 2, T - 1) ||
        throw(ArgumentError("observables length must equal T-1 = $(T - 1)"))

    (; u, z) = sol
    noise = _cache_noise(cache)

    # Copy noise into cache buffers
    if !isnothing(noise) && !isnothing(noise_concrete)
        copy_noise_to_cache!(noise, noise_concrete)
    end

    # Initialize state
    u[1] = assign!!(u[1], prob.u0)
    _init_model_state!!(prob, cache)

    # Initial observation
    if _has_observations(sol)
        z[1] = _observation!!(z[1], u[1], prob, cache, 1)
    end

    # Pre-compute observation noise Cholesky (used for loglik and/or simulation noise)
    has_obs_noise = !isnothing(prob.observables_noise) && !isnothing(cache.R)
    has_obs = has_obs_noise && !isnothing(prob.observables)
    if has_obs_noise
        R_cov = make_observables_covariance_matrix(prob.observables_noise)
        R_buf = cache.R
        R_chol_buf = cache.R_chol
        R_buf = copyto!!(R_buf, R_cov)
        R_chol_buf = symmetrize_upper!!(R_chol_buf, R_buf, perturb_diagonal)
        F_obs = cholesky!!(R_chol_buf, :U)
    end
    if has_obs
        logdetR = logdet_chol(F_obs)
        M_obs = size(R_buf, 1)
        log_const = M_obs * log(2π) + logdetR
    end

    loglik = zero(eltype(prob.u0))
    is_mutable = ismutable(u[1])

    @inbounds for t in 2:T
        w_t = isnothing(noise) ? nothing : noise[t - 1]
        u[t] = _transition!!(u[t], u[t - 1], w_t, prob, cache, t)

        if _has_observations(sol)
            z[t] = _observation!!(z[t], u[t], prob, cache, t)
        end

        # Log-likelihood contribution (Cholesky-based, allocation-free)
        if has_obs
            obs_t = get_observable(prob.observables, t - 1)
            ν = cache.innovation[t - 1]
            ν = copyto!!(ν, obs_t)
            if is_mutable
                for i in eachindex(ν)
                    ν[i] -= z[t][i]
                end
            else
                ν = ν - z[t]
            end
            cache.innovation[t - 1] = ν

            ν_solved = cache.innovation_solved[t - 1]
            ν_solved = ldiv!!(ν_solved, F_obs, ν)
            cache.innovation_solved[t - 1] = ν_solved
            quad = dot(ν, ν_solved)
            loglik -= 0.5 * (log_const + quad)
        end
    end

    # Add observation noise for simulation (when no observables provided)
    if has_obs_noise && isnothing(prob.observables)
        _add_observation_noise!!(z, F_obs)
    end

    t_values = prob.tspan[1]:1:prob.tspan[2]

    return build_solution(
        prob, alg, t_values, u; W = noise_concrete,
        logpdf = loglik, z,
        retcode = ReturnCode.Success
    )
end

# =============================================================================
# DirectIteration endpoints solver (save_everystep=false)
# Ping-pong between 2-element u/z, single-slot innovation cache.
# =============================================================================

function _solve_direct_iteration_endpoints!(
        prob, alg, sol, cache, B, T;
        perturb_diagonal = 0.0, kwargs...
    )
    noise_concrete = get_concrete_noise(prob, prob.noise, B, T - 1)

    if !isnothing(noise_concrete)
        length(noise_concrete) == T - 1 ||
            throw(ArgumentError("noise length $(length(noise_concrete)) must equal T-1 = $(T - 1)"))
        length(noise_concrete[1]) == size(B, 2) ||
            throw(ArgumentError("noise dimension $(length(noise_concrete[1])) must equal number of shocks $(size(B, 2))"))
    end
    maybe_check_size(prob.observables, 2, T - 1) ||
        throw(ArgumentError("observables length must equal T-1 = $(T - 1)"))

    (; u, z) = sol
    noise = _cache_noise(cache)
    _se = Val(false)

    if !isnothing(noise) && !isnothing(noise_concrete)
        copy_noise_to_cache!(noise, noise_concrete)
    end

    # Initialize state at ping-pong slot 1
    u[1] = assign!!(u[1], prob.u0)
    _init_model_state!!(prob, cache, _se)

    if _has_observations(sol)
        z[1] = _observation!!(z[1], u[1], prob, cache, 1, _se)
    end

    has_obs_noise = !isnothing(prob.observables_noise) && !isnothing(cache.R)
    has_obs = has_obs_noise && !isnothing(prob.observables)
    if has_obs_noise
        R_cov = make_observables_covariance_matrix(prob.observables_noise)
        R_buf = cache.R
        R_chol_buf = cache.R_chol
        R_buf = copyto!!(R_buf, R_cov)
        R_chol_buf = symmetrize_upper!!(R_chol_buf, R_buf, perturb_diagonal)
        F_obs = cholesky!!(R_chol_buf, :U)
    end
    if has_obs
        logdetR = logdet_chol(F_obs)
        M_obs = size(R_buf, 1)
        log_const = M_obs * log(2π) + logdetR
    end

    loglik = zero(eltype(prob.u0))
    is_mutable = ismutable(u[1])

    @inbounds for t in 2:T
        w_t = isnothing(noise) ? nothing : noise[t - 1]
        ci = _u_idx_pingpong(t)
        pi = _u_idx_pingpong(t - 1)
        u[ci] = _transition!!(u[ci], u[pi], w_t, prob, cache, t, _se)

        if _has_observations(sol)
            z[ci] = _observation!!(z[ci], u[ci], prob, cache, t, _se)
        end

        if has_obs
            obs_t = get_observable(prob.observables, t - 1)
            ν = cache.innovation[1]
            ν = copyto!!(ν, obs_t)
            if is_mutable
                for i in eachindex(ν)
                    ν[i] -= z[ci][i]
                end
            else
                ν = ν - z[ci]
            end
            cache.innovation[1] = ν

            ν_solved = cache.innovation_solved[1]
            ν_solved = ldiv!!(ν_solved, F_obs, ν)
            cache.innovation_solved[1] = ν_solved
            quad = dot(ν, ν_solved)
            loglik -= 0.5 * (log_const + quad)
        end
    end

    # Add observation noise for simulation (when no observables provided)
    if has_obs_noise && isnothing(prob.observables)
        _add_observation_noise!!(z, F_obs)
    end

    # Fixup: ensure u[1]=u0, u[2]=final state
    final_idx = _u_idx_pingpong(T)
    if final_idx == 1
        u[2] = assign!!(u[2], u[1])
    end
    u[1] = assign!!(u[1], prob.u0)
    if _has_observations(sol)
        if final_idx == 1
            z[2] = assign!!(z[2], z[1])
        end
        z[1] = _observation!!(z[1], u[1], prob, cache, 1, _se)
    end

    _step = max(1, prob.tspan[2] - prob.tspan[1])
    t_values = prob.tspan[1]:_step:prob.tspan[2]
    return build_solution(
        prob, alg, t_values, u; W = noise_concrete,
        logpdf = loglik, z,
        retcode = ReturnCode.Success
    )
end

# Single __solve route for all problem types with DirectIteration
function DiffEqBase.__solve(
        prob::AbstractStateSpaceProblem, alg::DirectIteration, args...;
        save_everystep = true, kwargs...
    )
    ws = CommonSolve.init(prob, alg; save_everystep, kwargs...)
    return CommonSolve.solve!(ws; kwargs...)
end

# =============================================================================
# ConditionalLikelihood solver — generic for all problem types
# Prediction error decomposition for fully-observed state-space models.
# =============================================================================

# Function barrier: same pattern as DirectIteration
function _solve!(
        prob::AbstractStateSpaceProblem, alg::ConditionalLikelihood, sol, cache;
        save_everystep::Val{SE} = Val(true), kwargs...
    ) where {SE}
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    B = _noise_matrix(prob)
    if SE
        return _solve_conditional_likelihood!(prob, alg, sol, cache, B, T; kwargs...)
    else
        return _solve_conditional_likelihood_endpoints!(prob, alg, sol, cache, B, T; kwargs...)
    end
end

function _solve_conditional_likelihood!(
        prob, alg, sol, cache, B, T;
        perturb_diagonal = 0.0, kwargs...
    )
    # Validate requirements
    isnothing(prob.observables) &&
        throw(ArgumentError("ConditionalLikelihood requires observables"))
    isnothing(prob.observables_noise) &&
        throw(ArgumentError("ConditionalLikelihood requires observables_noise"))
    maybe_check_size(prob.observables, 2, T - 1) ||
        throw(ArgumentError("observables length must equal T-1 = $(T - 1)"))

    # Get concrete noise and copy into cache
    noise_concrete = get_concrete_noise(prob, prob.noise, B, T - 1)

    (; u, z) = sol
    noise = _cache_noise(cache)
    has_obs_func = _has_observations(sol)

    if !isnothing(noise) && !isnothing(noise_concrete)
        copy_noise_to_cache!(noise, noise_concrete)
    end

    # Initialize state
    u[1] = assign!!(u[1], prob.u0)
    _init_model_state!!(prob, cache)

    # Initial observation (for diagnostics, not used in loglik)
    if has_obs_func
        z[1] = _observation!!(z[1], u[1], prob, cache, 1)
    end

    # Pre-compute observation noise Cholesky
    R_cov = make_observables_covariance_matrix(prob.observables_noise)
    R_buf = copyto!!(cache.R, R_cov)
    R_chol_buf = symmetrize_upper!!(cache.R_chol, R_buf, perturb_diagonal)
    F_obs = cholesky!!(R_chol_buf, :U)
    logdetR = logdet_chol(F_obs)
    M_obs = size(R_buf, 1)
    log_const = M_obs * log(2π) + logdetR

    loglik = zero(eltype(prob.u0))
    is_mutable = ismutable(u[1])

    @inbounds for t in 2:T
        w_t = isnothing(noise) ? nothing : noise[t - 1]

        # Predict into u[t] (temporary)
        u[t] = _transition!!(u[t], u[t - 1], w_t, prob, cache, t)

        # Predicted observation
        if has_obs_func
            z[t] = _observation!!(z[t], u[t], prob, cache, t)
            z_pred = z[t]
        else
            z_pred = u[t]
        end

        # Innovation: ν = obs_t - z_pred
        obs_t = get_observable(prob.observables, t - 1)
        ν = cache.innovation[t - 1]
        ν = copyto!!(ν, obs_t)
        if is_mutable
            for i in eachindex(ν)
                ν[i] -= z_pred[i]
            end
        else
            ν = ν - z_pred
        end
        cache.innovation[t - 1] = ν

        # Log-likelihood contribution
        ν_solved = cache.innovation_solved[t - 1]
        ν_solved = ldiv!!(ν_solved, F_obs, ν)
        cache.innovation_solved[t - 1] = ν_solved
        quad = dot(ν, ν_solved)
        loglik -= 0.5 * (log_const + quad)

        # CLAMP: set state to observation for next step
        u[t] = assign!!(u[t], obs_t)
    end

    t_values = prob.tspan[1]:1:prob.tspan[2]
    return build_solution(
        prob, alg, t_values, u; W = noise_concrete, logpdf = loglik, z,
        retcode = ReturnCode.Success
    )
end

# =============================================================================
# ConditionalLikelihood endpoints solver (save_everystep=false)
# =============================================================================

function _solve_conditional_likelihood_endpoints!(
        prob, alg, sol, cache, B, T;
        perturb_diagonal = 0.0, kwargs...
    )
    isnothing(prob.observables) &&
        throw(ArgumentError("ConditionalLikelihood requires observables"))
    isnothing(prob.observables_noise) &&
        throw(ArgumentError("ConditionalLikelihood requires observables_noise"))
    maybe_check_size(prob.observables, 2, T - 1) ||
        throw(ArgumentError("observables length must equal T-1 = $(T - 1)"))

    noise_concrete = get_concrete_noise(prob, prob.noise, B, T - 1)

    (; u, z) = sol
    noise = _cache_noise(cache)
    has_obs_func = _has_observations(sol)
    _se = Val(false)

    if !isnothing(noise) && !isnothing(noise_concrete)
        copy_noise_to_cache!(noise, noise_concrete)
    end

    u[1] = assign!!(u[1], prob.u0)
    _init_model_state!!(prob, cache, _se)

    if has_obs_func
        z[1] = _observation!!(z[1], u[1], prob, cache, 1, _se)
    end

    R_cov = make_observables_covariance_matrix(prob.observables_noise)
    R_buf = copyto!!(cache.R, R_cov)
    R_chol_buf = symmetrize_upper!!(cache.R_chol, R_buf, perturb_diagonal)
    F_obs = cholesky!!(R_chol_buf, :U)
    logdetR = logdet_chol(F_obs)
    M_obs = size(R_buf, 1)
    log_const = M_obs * log(2π) + logdetR

    loglik = zero(eltype(prob.u0))
    is_mutable = ismutable(u[1])

    @inbounds for t in 2:T
        w_t = isnothing(noise) ? nothing : noise[t - 1]
        ci = _u_idx_pingpong(t)
        pi = _u_idx_pingpong(t - 1)

        # Predict into u[ci]
        u[ci] = _transition!!(u[ci], u[pi], w_t, prob, cache, t, _se)

        # Predicted observation
        if has_obs_func
            z[ci] = _observation!!(z[ci], u[ci], prob, cache, t, _se)
            z_pred = z[ci]
        else
            z_pred = u[ci]
        end

        # Innovation
        obs_t = get_observable(prob.observables, t - 1)
        ν = cache.innovation[1]
        ν = copyto!!(ν, obs_t)
        if is_mutable
            for i in eachindex(ν)
                ν[i] -= z_pred[i]
            end
        else
            ν = ν - z_pred
        end
        cache.innovation[1] = ν

        ν_solved = cache.innovation_solved[1]
        ν_solved = ldiv!!(ν_solved, F_obs, ν)
        cache.innovation_solved[1] = ν_solved
        quad = dot(ν, ν_solved)
        loglik -= 0.5 * (log_const + quad)

        # CLAMP
        u[ci] = assign!!(u[ci], obs_t)
    end

    # Fixup: u[1]=u0, u[2]=final clamped state
    final_idx = _u_idx_pingpong(T)
    if final_idx == 1
        u[2] = assign!!(u[2], u[1])
    end
    u[1] = assign!!(u[1], prob.u0)
    if has_obs_func
        if final_idx == 1
            z[2] = assign!!(z[2], z[1])
        end
        z[1] = _observation!!(z[1], u[1], prob, cache, 1, _se)
    end

    _step = max(1, prob.tspan[2] - prob.tspan[1])
    t_values = prob.tspan[1]:_step:prob.tspan[2]
    return build_solution(
        prob, alg, t_values, u; W = noise_concrete, logpdf = loglik, z,
        retcode = ReturnCode.Success
    )
end

function DiffEqBase.__solve(
        prob::AbstractStateSpaceProblem, alg::ConditionalLikelihood, args...;
        save_everystep = true, kwargs...
    )
    ws = CommonSolve.init(prob, alg; save_everystep, kwargs...)
    return CommonSolve.solve!(ws; kwargs...)
end

# =============================================================================
# KalmanFilter solver — specific to LinearStateSpaceProblem
# =============================================================================

function _solve!(
        prob::LinearStateSpaceProblem, alg::KalmanFilter, sol, cache;
        save_everystep::Val{SE} = Val(true), perturb_diagonal = 0.0, kwargs...
    ) where {SE}
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    if !SE
        return _solve_kalman_endpoints!(
            prob, alg, sol, cache, T;
            perturb_diagonal, kwargs...
        )
    end
    length(prob.observables) == T - 1 ||
        throw(ArgumentError("observables length $(length(prob.observables)) must equal T-1 = $(T - 1)"))

    (; A, B, C, u0_prior_mean, u0_prior_var) = prob
    R = make_observables_covariance_matrix(prob.observables_noise)

    (; u, P, z) = sol
    (; B_prod, B_t) = cache

    # Compute B*B' once (mul_aat!! avoids BLAS syrk path for Enzyme AD correctness)
    B_prod = mul_aat!!(B_prod, B, B_t)

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
        # Get scratch buffers for this timestep
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

        # Current state (from solution output)
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
        quad = dot(ν_solved, ν)
        loglik -= 0.5 * (log_const_kf + logdetS + quad)
    end

    t_values = prob.tspan[1]:1:prob.tspan[2]
    return build_solution(
        prob, alg, t_values, sol.u; P = sol.P, W = nothing, logpdf = loglik,
        z = sol.z, retcode = ReturnCode.Success
    )
end

# =============================================================================
# KalmanFilter endpoints solver (save_everystep=false)
# Pure recursive filter: 2-element sol, 1-slot cache arrays.
# =============================================================================

function _solve_kalman_endpoints!(
        prob, alg, sol, cache, T;
        perturb_diagonal = 0.0, kwargs...
    )
    T_obs = T - 1
    length(prob.observables) == T_obs ||
        throw(ArgumentError("observables length $(length(prob.observables)) must equal T-1 = $(T_obs)"))

    (; A, B, C, u0_prior_mean, u0_prior_var) = prob
    R = make_observables_covariance_matrix(prob.observables_noise)

    (; u, P, z) = sol
    (; B_prod, B_t) = cache

    B_prod = mul_aat!!(B_prod, B, B_t)

    # Initialize at ping-pong slot 1
    u[1] = copyto!!(u[1], u0_prior_mean)
    P[1] = copyto!!(P[1], u0_prior_var)
    z[1] = mul!!(z[1], C, u[1])

    loglik = zero(eltype(u0_prior_var))
    is_mutable = ismutable(u[1])
    M_obs = size(C, 1)
    log_const_kf = M_obs * log(2π)

    @inbounds for t in 1:T_obs
        ci = _u_idx_pingpong(t)       # current filtered state
        ni = _u_idx_pingpong(t + 1)   # next filtered state

        # Single-slot cache buffers
        μp = cache.mu_pred[1]
        Σp = cache.sigma_pred[1]
        AΣ = cache.A_sigma[1]
        ΣGt = cache.sigma_Gt[1]
        ν = cache.innovation[1]
        S = cache.innovation_cov[1]
        S_chol_buf = cache.S_chol[1]
        ν_solved = cache.innovation_solved[1]
        rhs = cache.gain_rhs[1]
        K_t = cache.gain[1]
        KG = cache.gainG[1]
        KGS = cache.KgSigma[1]
        μu = cache.mu_update[1]

        μt = u[ci]
        Σt = P[ci]

        # Predict mean
        μp = mul!!(μp, A, μt)

        # Predict covariance
        AΣ = mul!!(AΣ, A, Σt)
        Σp = mul!!(Σp, AΣ, transpose(A))
        if is_mutable
            @inbounds for i in eachindex(Σp)
                Σp[i] += B_prod[i]
            end
        else
            Σp = Σp + B_prod
        end

        # Predicted observation
        z[ni] = mul!!(z[ni], C, μp)

        # Innovation
        obs_t = get_observable(prob.observables, t)
        ν = copyto!!(ν, obs_t)
        ν = mul!!(ν, C, μp, -1.0, 1.0)

        # Innovation covariance
        ΣGt = mul!!(ΣGt, Σp, transpose(C))
        S = mul!!(S, C, ΣGt)
        if is_mutable
            @inbounds for i in eachindex(S)
                S[i] += R[i]
            end
        else
            S = S + R
        end

        S_chol_buf = symmetrize_upper!!(S_chol_buf, S, perturb_diagonal)
        F = cholesky!!(S_chol_buf, :U)

        # Kalman gain
        rhs = transpose!!(rhs, ΣGt)
        rhs = ldiv!!(F, rhs)
        K_t = transpose!!(K_t, rhs)

        # Update mean
        μu = mul!!(μu, K_t, ν)
        if is_mutable
            @inbounds for i in eachindex(μp)
                u[ni][i] = μp[i] + μu[i]
            end
        else
            cache.mu_pred[1] = μp
            cache.mu_update[1] = μu
            u[ni] = μp + μu
        end

        # Update covariance
        KG = mul!!(KG, K_t, C)
        KGS = mul!!(KGS, KG, Σp)
        if is_mutable
            @inbounds for i in eachindex(Σp)
                P[ni][i] = Σp[i] - KGS[i]
            end
        else
            cache.sigma_pred[1] = Σp
            cache.KgSigma[1] = KGS
            P[ni] = Σp - KGS
        end

        # Log-likelihood
        ν_solved = ldiv!!(ν_solved, F, ν)
        cache.innovation[1] = ν
        cache.innovation_solved[1] = ν_solved
        logdetS = logdet_chol(F)
        quad = dot(ν_solved, ν)
        loglik -= 0.5 * (log_const_kf + logdetS + quad)
    end

    # Fixup: u[1]=initial, u[2]=final, P[1]=initial, P[2]=final, z[1]=initial, z[2]=final
    final_idx = _u_idx_pingpong(T)  # where final state ended up
    if final_idx == 1
        u[2] = assign!!(u[2], u[1])
        P[2] = copyto!!(P[2], P[1])
        z[2] = assign!!(z[2], z[1])
    end
    u[1] = copyto!!(u[1], u0_prior_mean)
    P[1] = copyto!!(P[1], u0_prior_var)
    z[1] = mul!!(z[1], C, u[1])

    _step = max(1, prob.tspan[2] - prob.tspan[1])
    t_values = prob.tspan[1]:_step:prob.tspan[2]
    return build_solution(
        prob, alg, t_values, sol.u; P = sol.P, W = nothing, logpdf = loglik,
        z = sol.z, retcode = ReturnCode.Success
    )
end

function DiffEqBase.__solve(
        prob::LinearStateSpaceProblem, alg::KalmanFilter, args...;
        save_everystep = true, kwargs...
    )
    ws = CommonSolve.init(prob, alg; save_everystep, kwargs...)
    return CommonSolve.solve!(ws; kwargs...)
end
