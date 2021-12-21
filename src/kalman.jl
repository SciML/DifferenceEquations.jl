function _solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype},
    solver::KalmanFilter,
    args...;
    vectype=identity,
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype<:Distribution, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2]
    A,B,C = prob.A, prob.B, prob.C
    R = cov(prob.obs_noise) # Extract covariance from noise distribution
    B_prod = B * B'
    K = size(B, 1) # Rows of latent states
    L = size(C, 1) # Rows of observations

    # Gaussian Prior
    u0_mean = mean(prob.u0) # TODO: Need more considerate way of handling priors
    u0_variance = cov(prob.u0)

    u = vectype(Vector{typeof(u0_mean)}(undef, T)) # Latent states
    P = vectype(Vector{Matrix{eltype(u0_mean)}}(undef, T)) # Latent noise
    z = vectype(Vector{typeof(u0_mean)}(undef, T)) # Observables generated

    u[1] = u0_mean
    P[1] = u0_variance
    z[1] = C * u[1]

    loglik = 0.0

    for t in 2:T
        # Kalman iteration
        u[t] = A * u[t-1]
        P[t] = A * P[t-1] * A' + B_prod
        z[t] = C * u[t]

        CP_t = C * P[t]
        V = Symmetric(CP_t * C' + R)
        loglik += logpdf(MvNormal(z[t], V), prob.observables[t-1])
        # loglik += logpdf(MvNormal(z[i], V), observables[i-1])
        K = CP_t' / V  # gain
        u[t] += K * (prob.observables[t-1] - z[t])
        P[t] -= K * CP_t
    end

    return StateSpaceSolution(copy(z), copy(u), nothing, copy(P), loglik)
end
