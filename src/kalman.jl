function _solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype},
    solver::KalmanFilter,
    args...;
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype<:Distribution, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    A, B, C = prob.A, prob.B, prob.C
    # The following line could be cov(prob.obs_noise) if the measurement error distribution is not MvNormal
    R = Diagonal(abs2.(prob.noise.σ)) # Extract covariance from noise distribution
    B_prod = B * B'

    # Gaussian Prior
    # u0 has to be a multivariate Normal distribution
    u0_mean = prob.u0.m
    u0_variance = prob.u0.C.U' * prob.u0.C.U

    u = Zygote.Buffer(Vector{typeof(u0_mean)}(undef, T)) # Mean of Kalman filter inferred latent states
    P = Zygote.Buffer(Vector{Matrix{eltype(u0_mean)}}(undef, T)) # Posterior variance of Kalman filter inferred latent states
    z = Zygote.Buffer(Vector{typeof(u0_mean)}(undef, T)) # Mean of observables, generated from mean of latent states

    u[1] = u0_mean
    P[1] = u0_variance
    z[1] = C * u[1]

    loglik = 0.0

    for t in 2:T
        t_n = t - 1 + prob.tspan[1]
        # Kalman iteration
        u[t] = A * u[t - 1]
        P[t] = A * P[t - 1] * A' + B_prod
        z[t] = C * u[t]

        CP_t = C * P[t]
        V = Symmetric(CP_t * C' + R)
        loglik += logpdf(MvNormal(z[t], V), prob.observables[t_n])
        K = CP_t' / V  # Kalman gain
        u[t] += K * (prob.observables[t_n] - z[t])
        P[t] -= K * CP_t
    end

    return StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
end
