
function _solve!(prob::LinearStateSpaceProblem{isinplace,Atype,Btype,Ctype,wtype,Rtype,utype,ttype,
                                               otype}, solver::KalmanFilter, args...;
                 kwargs...) where {isinplace,Atype,Btype,Ctype,wtype,Rtype<:Distribution,utype,
                                   ttype,otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    @unpack A, B, C, u0 = prob
    N = length(u0)
    M = size(C, 1)

    # TODO: move to internal algorithm cache
    # This method of preallocation won't work with staticarrays.  Note that we can't use eltype(mean(u0)) since it may be special case of FillArrays.zeros
    u = [Vector{eltype(u0)}(undef, N) for _ in 1:T] # Mean of Kalman filter inferred latent states
    P = [Matrix{eltype(u0)}(undef, N, N) for _ in 1:T] # Posterior variance of Kalman filter inferred latent states
    z = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T] # Mean of observables, generated from mean of latent states

    # TODO: these intermediates should be of size T-1 instead as the first was skipped.  Left in for checks on timing
    # Maintaining allocations for these intermediates is necessary for the rrule, but not for forward only.  Code could be refactored along those lines with solid unit tests.
    B_prod = Matrix{eltype(u0)}(undef, N, N)
    u_mid = [Vector{eltype(u0)}(undef, N) for _ in 1:T] # intermediate in u calculation
    P_mid = [Matrix{eltype(u0)}(undef, N, N) for _ in 1:T] # intermediate in P calculation
    innovation = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T]
    K = [Matrix{eltype(u0)}(undef, N, M) for _ in 1:T] # Gain
    CP = [Matrix{eltype(u0)}(undef, M, N) for _ in 1:T] # C * P[t]
    V = [PDMat{eltype(u0),Matrix{eltype(u0)}}(M, Matrix{eltype(u0)}(undef, M, M),
                                              Cholesky{eltype(u0),Matrix{eltype(u0)}}(Matrix{eltype(u0)}(undef,
                                                                                                         M,
                                                                                                         M),
                                                                                      'U', 0))
         for _ in 1:T] # preallocated buffers for cholesky and matrix itself

    # The following line could be cov(prob.obs_noise) if the measurement error distribution is not MvNormal
    R = prob.obs_noise.Σ # Extract covariance from noise distribution
    mul!(B_prod, B, B')

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
        mul!(u_mid[t], A, u[t - 1]) # u[t] = A u[t-1]
        mul!(z[t], C, u_mid[t]) # z[t] = C u[t]

        # P[t] = A * P[t - 1] * A' + B * B'
        mul!(temp_N_N, P[t - 1], A')
        mul!(P_mid[t], A, temp_N_N)
        P_mid[t] .+= B_prod

        mul!(CP[t], C, P_mid[t]) # CP[t] = C * P[t]

        # V[t] = CP[t] * C' + R
        mul!(V[t].mat, CP[t], C')
        V[t].mat .+= R

        # V_t .= (V_t + V_t') / 2 # classic hack to deal with stability of not being quite symmetric
        transpose!(temp_M_M, V[t].mat)
        V[t].mat .+= temp_M_M
        lmul!(0.5, V[t].mat)

        copy!(V[t].chol.factors, V[t].mat) # copy over to the factors for the cholesky and do in place
        cholesky!(V[t].chol.factors, Val(false); check = false) # inplace uses V_t with cholesky.  Now V[t]'s chol is upper-triangular        
        innovation[t] .= prob.observables[:, t - 1] - z[t]
        loglik += logpdf(MvNormal(V[t]), innovation[t])  # no allocations since V[t] is a PDMat

        # K[t] .= CP[t]' / V[t]  # Kalman gain
        # Can rewrite as K[t]' = V[t] \ CP[t] since V[t] is symmetric
        ldiv!(temp_M_N, V[t].chol, CP[t])
        transpose!(K[t], temp_M_N)

        #u[t] += K[t] * innovation[t]
        copy!(u[t], u_mid[t])
        mul!(u[t], K[t], innovation[t], 1, 1)

        #P[t] -= K[t] * CP[t]
        copy!(P[t], P_mid[t])
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
    @unpack A, B, C, u0 = prob
    N = length(u0)
    M = size(C, 1)

    # TODO: move to internal algorithm cache
    # This method of preallocation won't work with staticarrays.  Note that we can't use eltype(mean(u0)) since it may be special case of FillArrays.zeros
    B_prod = Matrix{eltype(u0)}(undef, N, N)
    u = [Vector{eltype(u0)}(undef, N) for _ in 1:T] # Mean of Kalman filter inferred latent states
    P = [Matrix{eltype(u0)}(undef, N, N) for _ in 1:T] # Posterior variance of Kalman filter inferred latent states
    z = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T] # Mean of observables, generated from mean of latent states

    # TODO: these intermediates should be of size T-1 instead as the first was skipped.  Left in for checks on timing
    # Maintaining allocations for these intermediates is necessary for the rrule, but not for forward only.  Code could be refactored along those lines with solid unit tests.
    u_mid = [Vector{eltype(u0)}(undef, N) for _ in 1:T] # intermediate in u calculation
    P_mid = [Matrix{eltype(u0)}(undef, N, N) for _ in 1:T] # intermediate in P calculation
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
    # The following line could be cov(prob.obs_noise) if the measurement error distribution is not MvNormal
    R = prob.obs_noise.Σ # Extract covariance from noise distribution
    mul!(B_prod, B, B')

    u[1] .= mean(u0)
    P[1] .= cov(u0)
    z[1] .= C * u[1]

    loglik = 0.0

    # temp buffers.  Could be moved into algorithm settings
    temp_N_N = Matrix{eltype(u0)}(undef, N, N)
    temp_M_M = Matrix{eltype(u0)}(undef, M, M)
    temp_M_N = Matrix{eltype(u0)}(undef, M, N)
    temp_N_M = Matrix{eltype(u0)}(undef, N, M)
    temp_M = Vector{eltype(u0)}(undef, M)
    temp_N = Vector{eltype(u0)}(undef, N)

    @inbounds for t in 2:T
        # Kalman iteration
        mul!(u_mid[t], A, u[t - 1]) # u[t] = A u[t-1]
        mul!(z[t], C, u_mid[t]) # z[t] = C u[t]

        # P[t] = A * P[t - 1] * A' + B * B'
        mul!(temp_N_N, P[t - 1], A')
        mul!(P_mid[t], A, temp_N_N)
        P_mid[t] .+= B_prod

        mul!(CP[t], C, P_mid[t]) # CP[t] = C * P[t]

        # V[t] = CP[t] * C' + R
        mul!(V[t].mat, CP[t], C')
        V[t].mat .+= R

        # V_t .= (V_t + V_t') / 2 # classic hack to deal with stability of not being quite symmetric
        transpose!(temp_M_M, V[t].mat)
        V[t].mat .+= temp_M_M
        lmul!(0.5, V[t].mat)

        copy!(V[t].chol.factors, V[t].mat) # copy over to the factors for the cholesky and do in place
        cholesky!(V[t].chol.factors, Val(false); check = false) # inplace uses V_t with cholesky.  Now V[t]'s chol is upper-triangular        
        innovation[t] .= prob.observables[:, t - 1] - z[t]
        loglik += logpdf(MvNormal(V[t]), innovation[t])  # no allocations since V[t] is a PDMat

        # K[t] .= CP[t]' / V[t]  # Kalman gain
        # Can rewrite as K[t]' = V[t] \ CP[t] since V[t] is symmetric
        ldiv!(temp_M_N, V[t].chol, CP[t])
        transpose!(K[t], temp_M_N)

        #u[t] += K[t] * innovation[t]
        copy!(u[t], u_mid[t])
        mul!(u[t], K[t], innovation[t], 1, 1)

        #P[t] -= K[t] * CP[t]
        copy!(P[t], P_mid[t])
        mul!(P[t], K[t], CP[t], -1, 1)
    end

    sol = StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
    function solve_pb(Δsol)
        Δlogpdf = Δsol.loglikelihood
        if iszero(Δlogpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end
        # Buffers
        ΔP = zero(P[1])
        Δu = zero(u[1])
        ΔA = zero(A)
        ΔB = zero(B)
        ΔC = zero(C)
        ΔK = zero(K[1])
        ΔP_mid = zero(ΔP)
        ΔP_mid_sum = zero(ΔP)
        ΔCP = zero(CP[1])
        Δu_mid = zero(u_mid[1])
        Δz = zero(z[1])
        ΔV = zero(V[1].mat)

        for t in T:-1:2
            # The inverse is used throughout, including in quadratic forms.  For large systems this might not be stable            
            inv_V = Symmetric(inv(V[t].chol)) # use cholesky factorization to invert.  Symmetric

            # Sensitivity accumulation
            copy!(ΔP_mid, ΔP)
            mul!(ΔK, ΔP, CP[t]', -1, 0) # i.e. ΔK = -ΔP * CP[t]'
            mul!(ΔCP, K[t]', ΔP, -1, 0) # i.e. ΔCP = - K[t]' * ΔP
            copy!(Δu_mid, Δu)
            mul!(ΔK, Δu, innovation[t]', 1, 1) # ΔK += Δu * innovation[t]'
            mul!(Δz, K[t]', Δu, -1, 0)  # i.e, Δz = -K[t]'* Δu
            mul!(ΔCP, inv_V, ΔK', 1, 1) # ΔCP += inv_V * ΔK'

            # ΔV .= -inv_V * CP[t] * ΔK * inv_V
            mul!(temp_M_N, inv_V, CP[t])
            mul!(temp_N_M, ΔK, inv_V)
            mul!(ΔV, temp_M_N, temp_N_M, -1, 0)

            mul!(ΔC, ΔCP, P_mid[t]', 1, 1) # ΔC += ΔCP * P_mid[t]'
            mul!(ΔP_mid, C', ΔCP, 1, 1) # ΔP_mid += C' * ΔCP
            mul!(Δz, inv_V, innovation[t], Δlogpdf, 1) # Δz += Δlogpdf * inv_V * innovation[t] # Σ^-1 * (z_obs - z)

            #ΔV -= Δlogpdf * 0.5 * (inv_V - inv_V * innovation[t] * innovation[t]' * inv_V) # -0.5 * (Σ^-1 - Σ^-1(z_obs - z)(z_obx - z)'Σ^-1)
            mul!(temp_M, inv_V, innovation[t])
            mul!(temp_M_M, temp_M, temp_M')
            temp_M_M .-= inv_V
            rmul!(temp_M_M, Δlogpdf * 0.5)
            ΔV += temp_M_M

            #ΔC += ΔV * C * P_mid[t]' + ΔV' * C * P_mid[t]
            mul!(temp_M_N, C, P_mid[t])
            transpose!(temp_M_M, ΔV)
            temp_M_M .+= ΔV
            mul!(ΔC, temp_M_M, temp_M_N, 1, 1)

            # ΔP_mid += C' * ΔV * C
            mul!(temp_M_N, ΔV, C)
            mul!(ΔP_mid, C', temp_M_N, 1, 1)

            mul!(ΔC, Δz, u_mid[t]', 1, 1) # ΔC += Δz * u_mid[t]'
            mul!(Δu_mid, C', Δz, 1, 1) # Δu_mid += C' * Δz

            # Calculates (ΔP_mid + ΔP_mid')
            transpose!(ΔP_mid_sum, ΔP_mid)
            ΔP_mid_sum .+= ΔP_mid

            # ΔA += (ΔP_mid + ΔP_mid') * A * P[t - 1]
            mul!(temp_N_N, A, P[t - 1])
            mul!(ΔA, ΔP_mid_sum, temp_N_N, 1, 1)

            # ΔP .= A' * ΔP_mid * A # pass into next period
            mul!(temp_N_N, ΔP_mid, A)
            mul!(ΔP, A', temp_N_N)

            mul!(ΔB, ΔP_mid_sum, B, 1, 1) # ΔB += ΔP_mid_sum * B
            mul!(ΔA, Δu_mid, u[t - 1]', 1, 1) # ΔA += Δu_mid * u[t - 1]'
            mul!(Δu, A', Δu_mid)
        end
        ΔΣ = Tangent{typeof(prob.u0.Σ)}(; mat = ΔP) # TODO: This is not exactly correct since it doesn't do the "chol".  Add to prevent misuse.
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC,
                                      u0 = Tangent{typeof(prob.u0)}(; μ = Δu, Σ = ΔΣ)), NoTangent(),
                map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
