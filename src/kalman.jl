
function _solve!(prob::LinearStateSpaceProblem{isinplace,Atype,Btype,Ctype,wtype,Rtype,utype,ttype,
                                               otype}, solver::KalmanFilter, args...;
                 kwargs...) where {isinplace,Atype,Btype,Ctype,wtype,Rtype<:Distribution,utype,
                                   ttype,otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    @unpack A, B, C, u0 = prob
    N = length(u0)
    M = size(C, 1)

    # The following line could be cov(prob.obs_noise) if the measurement error distribution is not MvNormal
    R = prob.obs_noise.Σ # Extract covariance from noise distribution
    B_prod = B * B'

    # TODO: move to internal algorithm cache
    # This method of preallocation won't work with staticarrays.  Note that we can't use eltype(mean(u0)) since it may be special case of FillArrays.zeros
    u = [Vector{eltype(u0)}(undef, N) for _ in 1:T] # Mean of Kalman filter inferred latent states
    P = [Matrix{eltype(u0)}(undef, N, N) for _ in 1:T] # Posterior variance of Kalman filter inferred latent states
    z = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T] # Mean of observables, generated from mean of latent states

    # TODO: these intermediates should be of size T-1 instead as the first was skipped.  Left in for checks on timing
    innovation = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T]
    K = [Matrix{eltype(u0)}(undef, N, M) for _ in 1:T] # Gain
    CP = [Matrix{eltype(u0)}(undef, M, N) for _ in 1:T] # C * P[t]
    V = [PDMat{eltype(u0),Matrix{eltype(u0)}}(M, Matrix{eltype(u0)}(undef, M, M),
                                              Cholesky{eltype(u0),Matrix{eltype(u0)}}(Matrix{eltype(u0)}(undef,
                                                                                                         M,
                                                                                                         M),
                                                                                      'U', 0))
         for _ in 1:T] # preallocated buffers for cholesky and matrix itself

    # Gaussian Prior
    u[1] .= mean(u0)
    P[1] .= cov(u0)
    z[1] .= C * u[1]

    loglik = 0.0

    # temp buffers.  Could be moved into algorithm settings
    temp_N_N = Matrix{eltype(u0)}(undef, N, N)
    temp_M_M = Matrix{eltype(u0)}(undef, M, M)
    temp_M_N = Matrix{eltype(u0)}(undef, M, N)

    @inbounds for t in 2:T
        # Kalman iteration
        mul!(u[t], A, u[t - 1]) # u[t] = A u[t-1]
        mul!(z[t], C, u[t]) # z[t] = C u[t]

        # P[t] = A * P[t - 1] * A' + B * B'
        mul!(temp_N_N, P[t - 1], A')
        mul!(P[t], A, temp_N_N)
        P[t] .+= B_prod

        mul!(CP[t], C, P[t]) # CP[t] = C * P[t]
        V_t = V[t].mat # matrix in the posdef structure

        # V[t] = CP[t] * C' + R
        mul!(V_t, CP[t], C')
        V_t .+= R

        # V_t .= (V_t + V_t') / 2 # classic hack to deal with stability of not being quite symmetric
        transpose!(temp_M_M, V_t)
        V_t .+= temp_M_M
        lmul!(0.5, V_t)

        copy!(V[t].chol.factors, V_t) # copy over to the factors for the cholesky and do in place
        cholesky!(V[t].chol.factors, Val(false); check = false) # inplace uses V_t with cholesky.  Now V[t]'s chol is upper-triangular        
        innovation[t] .= prob.observables[:, t - 1] - z[t]
        loglik += logpdf(MvNormal(V[t]), innovation[t])  # no allocations since V[t] is a PDMat

        # K[t] .= CP[t]' / V[t]  # Kalman gain
        # Can rewrite as K[t]' = V[t] \ CP[t] since V[t] is symmetric
        ldiv!(temp_M_N, V[t].chol, CP[t])
        transpose!(K[t], temp_M_N)

        #u[t] += K[t] * innovation[t]
        mul!(u[t], K[t], innovation[t], 1, 1)

        #P[t] -= K[t] * CP[t]
        mul!(P[t], K[t], CP[t], -1, 1)
    end

    return StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
end

function ChainRulesCore.rrule(::typeof(_solve!),
                              prob::LinearStateSpaceProblem{isinplace,Atype,Btype,Ctype,wtype,Rtype,
                                                            utype,ttype,otype}, ::KalmanFilter,
                              args...;
                              kwargs...) where {isinplace,Atype,Btype,Ctype,wtype,Rtype,utype,ttype,
                                                otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    A, B, C = prob.A, prob.B, prob.C
    # The following line could be cov(prob.obs_noise) if the measurement error distribution is not MvNormal
    R = prob.obs_noise.Σ  # Extract covariance from noise distribution
    B_prod = B * B'

    # Gaussian Prior
    # u0 has to be a multivariate Normal distribution
    u0_mean = prob.u0.μ
    u0_variance = prob.u0.Σ

    u = Vector{Vector{eltype(u0_mean)}}(undef, T) # Mean of Kalman filter inferred latent states
    u_mid = Vector{Vector{eltype(u0_mean)}}(undef, T)
    P = Vector{Matrix{eltype(u0_mean)}}(undef, T) # Posterior variance of Kalman filter inferred latent states
    P_mid = Vector{Matrix{eltype(u0_mean)}}(undef, T)
    z = Vector{Vector{eltype(u0_mean)}}(undef, T) # Mean of observables, generated from mean of latent states
    innovation = Vector{Vector{eltype(u0_mean)}}(undef, T) # temporary for innovations

    u[1] = u0_mean
    P[1] = u0_variance
    z[1] = C * u[1]

    loglik = 0.0

    for t in 2:T
        t_n = t - 1 + prob.tspan[1]
        # Kalman iteration
        u_mid[t] = A * u[t - 1]
        P_mid[t] = A * P[t - 1] * A' + B_prod
        z[t] = C * u_mid[t]

        CP_t = C * P_mid[t]
        V = Symmetric(CP_t * C' + R)
        loglik += logpdf(MvNormal(z[t], V), prob.observables[:, t_n])
        K = CP_t' / V  # Kalman gain
        u[t] = u_mid[t] + K * (prob.observables[:, t_n] - z[t])
        P[t] = P_mid[t] - K * CP_t
    end

    sol = StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
    function solve_pb(Δsol)
        Δlogpdf = Δsol.loglikelihood
        if iszero(Δlogpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end
        # A bunch of initializations here
        ΔP = zero(u0_variance)
        Δu = zero(u0_mean)
        ΔA = zero(A)
        ΔB = zero(B)
        ΔC = zero(C)
        for t in T:-1:2
            t_n = t - 1 + prob.tspan[1]
            # Generate some intermediate variables
            CP_t = C * P_mid[t]
            V = Symmetric(CP_t * C' + R)
            inv_V = inv(V)
            K = CP_t' / V
            # Sensitivity accumulation
            ΔP_mid = copy(ΔP)
            ΔK = -ΔP * CP_t'
            ΔCP_t = -K' * ΔP
            Δu_mid = copy(Δu)
            ΔK += Δu * (prob.observables[:, t_n] - z[t])'
            Δz = -K' * Δu
            ΔCP_t += inv_V * ΔK'
            ΔV = -inv_V' * CP_t * ΔK * inv_V'
            ΔC += ΔCP_t * P_mid[t]'
            ΔP_mid += C' * ΔCP_t
            Δz += Δlogpdf * inv_V * (prob.observables[:, t_n] - z[t]) # Σ^-1 * (z_obs - z)
            ΔV -= Δlogpdf *
                  0.5 *
                  (inv_V -
                   inv_V *
                   (prob.observables[:, t_n] - z[t]) *
                   (prob.observables[:, t_n] - z[t])' *
                   inv_V) # -0.5 * (Σ^-1 - Σ^-1(z_obs - z)(z_obx - z)'Σ^-1)
            ΔC += ΔV * C * P_mid[t]' + ΔV' * C * P_mid[t]
            ΔP_mid += C' * ΔV * C
            ΔC += Δz * u_mid[t]'
            Δu_mid += C' * Δz
            ΔA += (ΔP_mid + ΔP_mid') * A * P[t - 1]
            ΔP = A' * ΔP_mid * A # pass into next period
            ΔB += (ΔP_mid + ΔP_mid') * B
            ΔA += Δu_mid * u[t - 1]'
            Δu = A' * Δu_mid
        end
        ΔΣ = Tangent{typeof(prob.u0.Σ)}(; mat = ΔP)
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC,
                                      u0 = Tangent{typeof(prob.u0)}(; μ = Δu, Σ = ΔΣ)), NoTangent(),
                map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
