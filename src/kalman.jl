function _solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype},
    solver::KalmanFilter,
    args...;
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype<:AbstractMatrix, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2]
    A,B,C,R = prob.A, prob.B, prob.C, prob.R
    B_prod = B * B'
    K = size(B, 1) # Rows of latent states
    L = size(C, 1) # Rows of observations

    # Gaussian Prior
    u0_mean = mean(prob.u0) # TODO: Need more considerate way of handling priors
    u0_variance = cov(prob.u0)

    u = Vector{typeof(u0_mean)}(undef, T+1) # Latent states
    P = Vector{Matrix{eltype(u0_mean)}}(undef, T+1) # Latent noise
    z = Vector{typeof(u0_mean)}(undef, T+1) # Observables generated

    u[1] = u0_mean
    P[1] = u0_variance
    z[1] = C * u[1]

    for i = 2:T+1
        # Kalman iteration
        u[i] = A * u[i-1]
        P[i] = A * P[i-1] * A' + B_prod
        z[i] = C * u[i]

        CP_i = C * P[i]
        V = Symmetric(CP_i * C' + R)
        # loglik += logpdf(MvNormal(z[i], V), observables[i-1])
        K = CP_i' / V  # gain
        u[i] += K * (prob.observables[i-1] - z[i])
        P[i] -= K * CP_i
    end

    return StateSpaceSolution(copy(z), copy(u), nothing, copy(P), nothing)
    # return StateSpaceSolution(copy(z), copy(u), nothing, copy(P), loglik)
end
