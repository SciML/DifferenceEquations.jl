
function _solve!(prob::LinearStateSpaceProblem{isinplace,Atype,Btype,Ctype,wtype,Rtype,utype,ttype,
                                               otype}, solver::KalmanFilter, args...;
                 kwargs...) where {isinplace,Atype,Btype,Ctype,wtype,Rtype<:Distribution,utype,
                                   ttype,otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    @unpack A, B, C, u0 = prob
    @unpack u, z, P, B_prod, u_temp, P_temp, K, CP, V, temp_N_N, temp_L_L, temp_L_N, temp_N_L = prob.cache
    @assert length(u) >= T && length(z) >= T

    N = length(u0) # number of states
    M = size(B, 2) # number of shocks on evolution matrix
    L = size(C, 1) # number of observables

    # The following line could be cov(prob.obs_noise) if the measurement error distribution is not MvNormal
    R = prob.obs_noise.Σ # Extract covariance from noise distribution
    mul!(B_prod, B, B')

    # Gaussian Prior
    u[1] .= mean(u0)
    P[1] .= cov(u0)
    z[1] .= C * u[1]

    loglik = 0.0

    @views @inbounds for t in 2:T
        # Kalman iteration
        mul!(u_temp[t], A, u[t - 1]) # u[t] = A u[t-1]
        mul!(z[t], C, u_temp[t]) # z[t] = C u[t]

        # P[t] = A * P[t - 1] * A' + B * B'
        mul!(temp_N_N, P[t - 1], A')
        mul!(P_temp[t], A, temp_N_N)
        P_temp[t] .+= B_prod

        mul!(CP[t], C, P_temp[t]) # CP[t] = C * P[t]

        # V[t] = CP[t] * C' + R
        mul!(V[t].mat, CP[t], C')
        V[t].mat .+= R

        # V_t .= (V_t + V_t') / 2 # classic hack to deal with stability of not being quite symmetric
        transpose!(temp_L_L, V[t].mat)
        V[t].mat .+= temp_L_L
        lmul!(0.5, V[t].mat)

        copy!(V[t].chol.factors, V[t].mat) # copy over to the factors for the cholesky and do in place
        cholesky!(V[t].chol.factors, Val(false); check = false) # inplace uses V_t with cholesky.  Now V[t]'s chol is upper-triangular        
        innovation = view(prob.observables, :, t - 1) - z[t]
        loglik += logpdf(MvNormal(V[t]), innovation)  # no allocations since V[t] is a PDMat

        # K[t] .= CP[t]' / V[t]  # Kalman gain
        # Can rewrite as K[t]' = V[t] \ CP[t] since V[t] is symmetric
        ldiv!(temp_L_N, V[t].chol, CP[t])
        transpose!(K[t], temp_L_N)

        #u[t] += K[t] * innovation[t]
        copy!(u[t], u_temp[t])
        mul!(u[t], K[t], innovation, 1, 1)

        #P[t] -= K[t] * CP[t]
        copy!(P[t], P_temp[t])
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
    @unpack u, z, P, B_prod, u_temp, P_temp, K, CP, V, temp_N_N, temp_L_L, temp_L_N, temp_N_L = prob.cache
    @assert length(u) >= T && length(z) >= T

    N = length(u0) # number of states
    M = size(B, 2) # number of shocks on evolution matrix
    L = size(C, 1) # number of observables

    # The following line could be cov(prob.obs_noise) if the measurement error distribution is not MvNormal
    R = prob.obs_noise.Σ # Extract covariance from noise distribution
    mul!(B_prod, B, B')

    # Gaussian Prior
    u[1] .= mean(u0)
    P[1] .= cov(u0)
    z[1] .= C * u[1]

    loglik = 0.0

    @views @inbounds for t in 2:T
        # Kalman iteration
        mul!(u_temp[t], A, u[t - 1]) # u[t] = A u[t-1]
        mul!(z[t], C, u_temp[t]) # z[t] = C u[t]

        # P[t] = A * P[t - 1] * A' + B * B'
        mul!(temp_N_N, P[t - 1], A')
        mul!(P_temp[t], A, temp_N_N)
        P_temp[t] .+= B_prod

        mul!(CP[t], C, P_temp[t]) # CP[t] = C * P[t]

        # V[t] = CP[t] * C' + R
        mul!(V[t].mat, CP[t], C')
        V[t].mat .+= R

        # V_t .= (V_t + V_t') / 2 # classic hack to deal with stability of not being quite symmetric
        transpose!(temp_L_L, V[t].mat)
        V[t].mat .+= temp_L_L
        lmul!(0.5, V[t].mat)

        copy!(V[t].chol.factors, V[t].mat) # copy over to the factors for the cholesky and do in place
        cholesky!(V[t].chol.factors, Val(false); check = false) # inplace uses V_t with cholesky.  Now V[t]'s chol is upper-triangular        
        innovation = view(prob.observables, :, t - 1) - z[t]
        loglik += logpdf(MvNormal(V[t]), innovation)  # no allocations since V[t] is a PDMat

        # K[t] .= CP[t]' / V[t]  # Kalman gain
        # Can rewrite as K[t]' = V[t] \ CP[t] since V[t] is symmetric
        ldiv!(temp_L_N, V[t].chol, CP[t])
        transpose!(K[t], temp_L_N)

        #u[t] += K[t] * innovation[t]
        copy!(u[t], u_temp[t])
        mul!(u[t], K[t], innovation, 1, 1)

        #P[t] -= K[t] * CP[t]
        copy!(P[t], P_temp[t])
        mul!(P[t], K[t], CP[t], -1, 1)
    end
    sol = StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
    function solve_pb(Δsol)
        @unpack temp_L, temp_N = prob.cache

        Δlogpdf = Δsol.loglikelihood
        if iszero(Δlogpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end
        # Buffers.  Could move to cache for big problems
        ΔP = zero(P[1])
        Δu = zero(u[1])
        ΔA = zero(A)
        ΔB = zero(B)
        ΔC = zero(C)
        ΔK = zero(K[1])
        ΔP_temp = zero(ΔP)
        ΔP_temp_sum = zero(ΔP)
        ΔCP = zero(CP[1])
        Δu_temp = zero(u_temp[1])
        Δz = zero(z[1])
        ΔV = zero(V[1].mat)

        for t in T:-1:2
            # The inverse is used throughout, including in quadratic forms.  For large systems this might not be stable            
            inv_V = Symmetric(inv(V[t].chol)) # use cholesky factorization to invert.  Symmetric

            # Sensitivity accumulation
            copy!(ΔP_temp, ΔP)
            mul!(ΔK, ΔP, CP[t]', -1, 0) # i.e. ΔK = -ΔP * CP[t]'
            mul!(ΔCP, K[t]', ΔP, -1, 0) # i.e. ΔCP = - K[t]' * ΔP
            copy!(Δu_temp, Δu)
            innovation = view(prob.observables, :, t - 1) - z[t]
            mul!(ΔK, Δu, innovation', 1, 1) # ΔK += Δu * innovation[t]'
            mul!(Δz, K[t]', Δu, -1, 0)  # i.e, Δz = -K[t]'* Δu
            mul!(ΔCP, inv_V, ΔK', 1, 1) # ΔCP += inv_V * ΔK'

            # ΔV .= -inv_V * CP[t] * ΔK * inv_V
            mul!(temp_L_N, inv_V, CP[t])
            mul!(temp_N_L, ΔK, inv_V)
            mul!(ΔV, temp_L_N, temp_N_L, -1, 0)

            mul!(ΔC, ΔCP, P_temp[t]', 1, 1) # ΔC += ΔCP * P_temp[t]'
            mul!(ΔP_temp, C', ΔCP, 1, 1) # ΔP_temp += C' * ΔCP
            mul!(Δz, inv_V, innovation, Δlogpdf, 1) # Δz += Δlogpdf * inv_V * innovation[t] # Σ^-1 * (z_obs - z)

            #ΔV -= Δlogpdf * 0.5 * (inv_V - inv_V * innovation[t] * innovation[t]' * inv_V) # -0.5 * (Σ^-1 - Σ^-1(z_obs - z)(z_obx - z)'Σ^-1)
            mul!(temp_L, inv_V, innovation)
            mul!(temp_L_L, temp_L, temp_L')
            temp_L_L .-= inv_V
            rmul!(temp_L_L, Δlogpdf * 0.5)
            ΔV += temp_L_L

            #ΔC += ΔV * C * P_temp[t]' + ΔV' * C * P_temp[t]
            mul!(temp_L_N, C, P_temp[t])
            transpose!(temp_L_L, ΔV)
            temp_L_L .+= ΔV
            mul!(ΔC, temp_L_L, temp_L_N, 1, 1)

            # ΔP_temp += C' * ΔV * C
            mul!(temp_L_N, ΔV, C)
            mul!(ΔP_temp, C', temp_L_N, 1, 1)

            mul!(ΔC, Δz, u_temp[t]', 1, 1) # ΔC += Δz * u_temp[t]'
            mul!(Δu_temp, C', Δz, 1, 1) # Δu_temp += C' * Δz

            # Calculates (ΔP_temp + ΔP_temp')
            transpose!(ΔP_temp_sum, ΔP_temp)
            ΔP_temp_sum .+= ΔP_temp

            # ΔA += (ΔP_temp + ΔP_temp') * A * P[t - 1]
            mul!(temp_N_N, A, P[t - 1])
            mul!(ΔA, ΔP_temp_sum, temp_N_N, 1, 1)

            # ΔP .= A' * ΔP_temp * A # pass into next period
            mul!(temp_N_N, ΔP_temp, A)
            mul!(ΔP, A', temp_N_N)

            mul!(ΔB, ΔP_temp_sum, B, 1, 1) # ΔB += ΔP_temp_sum * B
            mul!(ΔA, Δu_temp, u[t - 1]', 1, 1) # ΔA += Δu_temp * u[t - 1]'
            mul!(Δu, A', Δu_temp)
        end
        ΔΣ = Tangent{typeof(prob.u0.Σ)}(; mat = ΔP) # TODO: This is not exactly correct since it doesn't do the "chol".  Add to prevent misuse.
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC,
                                      u0 = Tangent{typeof(prob.u0)}(; μ = Δu, Σ = ΔΣ),
                                      cache = NoTangent(), observables = NoTangent(),
                                      obs_noise = NoTangent()), NoTangent(),
                map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
