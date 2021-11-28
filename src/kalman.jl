# LTI with observables and without conditioning on noise = Kalman Filter
function _solve(
    alg::LTI,
    noise::Nothing,
    observables,
    A::AbstractMatrix,
    B::AbstractMatrix,
    C::AbstractMatrix,
    D::TuringDiagMvNormal,
    u0,
    tspan,
)
    # hardcoded right now for tspan = (0, T) for T+1 points
    T = tspan[2]
    @assert tspan[1] == 0
    @assert length(observables) == T # i.e. we do not calculate the likelihood of the initial condition

    # Gaussian Prior
    u0_mean = u0.m
    u0_variance = u0.C.U' * u0.C.U
    R = Diagonal(abs2.(D.σ))  # the DistributionsAD doesn't have cov defined the covariance of the MvNormal
    B_prod = B * B'

    # TODO: when saveall = false, etc. don't allocate everything, or at least don't save it
    u = Zygote.Buffer(Vector{Vector{Float64}}(undef, T + 1)) # prior mean
    P = Zygote.Buffer(Vector{Matrix{Float64}}(undef, T + 1)) # prior variance
    z = Zygote.Buffer(Vector{Vector{Float64}}(undef, T + 1)) # mean observation

    u[1] = u0_mean
    P[1] = u0_variance
    z[1] = C * u[1]
    loglik = 0.0

    for i = 2:T+1
        # Kalman iteration
        u[i] = A * u[i-1]
        P[i] = A * P[i-1] * A' + B_prod
        z[i] = C * u[i]

        CP_i = C * P[i]
        V = Symmetric(CP_i * C' + R)
        loglik += logpdf(MvNormal(z[i], V), observables[i-1])
        K = CP_i' / V  # gain
        u[i] += K * (observables[i-1] - z[i])
        P[i] -= K * CP_i
    end
    return StateSpaceSolution(copy(z), copy(u), nothing, copy(P), loglik)
end
