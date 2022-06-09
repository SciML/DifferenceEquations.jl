
maybe_logpdf(observables_noise, observables::Nothing, t, z) = 0.0
maybe_logpdf(observables_noise, observables, t, z) = logpdf(observables_noise,
                                                            view(observables, :, t) - z)
make_observables_noise(observables_noise::Nothing) = nothing
make_observables_noise(observables_noise::AbstractMatrix) = MvNormal(observables_noise)
make_observables_noise(observables_noise::AbstractVector) = MvNormal(observables_noise)

function DiffEqBase.__solve(prob::LinearStateSpaceProblem{uType,uPriorMeanType,uPriorVarType,tType,
                                                          P,NP,F,AType,BType,
                                                          CType,RType,ObsType,K},
                            alg::DirectIteration, args...;
                            kwargs...) where {uType,uPriorMeanType,uPriorVarType,tType,P,NP,F,AType,
                                              BType,CType,RType,
                                              ObsType,K}
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    @unpack A, B, C = prob

    # checks on bounds
    noise = get_concrete_noise(prob, prob.noise, prob.B, T - 1)  # concrete noise for simulations as required.
    observables_noise = make_observables_noise(prob.observables_noise)

    @assert size(noise, 1) == size(prob.B, 2)
    @assert size(noise, 2) == T - 1
    @assert maybe_check_size(prob.observables, 2, T - 1)

    u = [zero(prob.u0) for _ in 1:T]
    z1 = C * prob.u0
    z = [zero(z1) for _ in 1:T]

    # Initialize
    u[1] .= prob.u0
    z[1] .= z1

    loglik = 0.0
    @inbounds for t in 2:T
        mul!(u[t], A, u[t - 1])
        mul!(u[t], B, view(noise, :, t - 1), 1, 1)

        mul!(z[t], C, u[t])
        loglik += maybe_logpdf(observables_noise, prob.observables, t - 1, z[t])
    end
    t_values = prob.tspan[1]:prob.tspan[2]
    return build_solution(prob, alg, t_values, u; W = noise,
                          logpdf = ObsType <: Nothing ? nothing : loglik, z, retcode = :Success)
end

# Ideally hook into existing sensitity dispatching
# Trouble with Zygote.  The problem isn't the _concrete_solve_adjoint but rather something in the
# adjoint of the basic solve and `solve_up`.  Probably promotion on the prob

# function DiffEqBase._concrete_solve_adjoint(prob::LinearStateSpaceProblem, alg::DirectIteration,
#                                             sensealg, u0, p, args...; kwargs...)
function ChainRulesCore.rrule(::typeof(DiffEqBase.solve), prob::LinearStateSpaceProblem,
                              alg::DirectIteration, args...; kwargs...)
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    @assert !isnothing(prob.noise)  # need to have concrete noise for this simple method
    noise = get_concrete_noise(prob, prob.noise, prob.B, T - 1)  # concrete noise for simulations as required.
    observables_noise = make_observables_noise(prob.observables_noise)
    @assert typeof(observables_noise) <: ZeroMeanDiagNormal  # can extend to more general in rrule

    # checks on bounds
    @assert size(noise, 1) == size(prob.B, 2)
    @assert maybe_check_size(prob.observables, 2, T - 1)
    @assert size(noise, 2) == T - 1

    @unpack A, B, C = prob
    u = [zero(prob.u0) for _ in 1:T] # TODO: move to internal algorithm cache
    z1 = C * prob.u0
    z = [zero(z1) for _ in 1:T] # TODO: move to internal algorithm cache

    # Initialize
    u[1] .= prob.u0
    z[1] .= z1

    loglik = 0.0
    @inbounds for t in 2:T
        mul!(u[t], A, u[t - 1])
        mul!(u[t], B, view(noise, :, t - 1), 1, 1)

        mul!(z[t], C, u[t])
        loglik += logpdf(observables_noise, view(prob.observables, :, t - 1) - z[t])
    end
    t_values = prob.tspan[1]:prob.tspan[2]
    sol = build_solution(prob, alg, t_values, u; W = noise, logpdf = loglik, z, retcode = :Success)

    function solve_pb(Δsol)
        # Currently only changes in the logpdf are supported in the rrule
        @assert Δsol.u == ZeroTangent()
        @assert Δsol.W == ZeroTangent()
        @assert Δsol.z == ZeroTangent()

        Δlogpdf = Δsol.logpdf
        if iszero(Δlogpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end
        ΔA = zero(A)
        ΔB = zero(B)
        ΔC = zero(C)
        Δnoise = similar(noise)
        Δu = zero(u[1])
        Δu_temp = zero(u[1])
        Δz = zero(z[1])

        # Assert checked above about being diagonal
        observables_noise_cov = prob.observables_noise

        @views @inbounds for t in T:-1:2
            Δz .= Δlogpdf * (view(prob.observables, :, t - 1) - z[t]) ./
                  observables_noise_cov # This is using the vector directly.   More generally, it should be Σ^-1 * (z_obs - z)
            # TODO: check if this can be repalced with the following and if it has a performance regression for diagonal noise covariance
            # ldiv!(Δz, observables_noise.Σ.chol, innovation[t])
            # rmul!(Δlogpdf, Δz)

            copy!(Δu_temp, Δu)
            mul!(Δu_temp, C', Δz, 1, 1)
            mul!(Δu, A', Δu_temp)
            mul!(view(Δnoise, :, t - 1), B', Δu_temp)
            # Now, deal with the coefficients
            mul!(ΔA, Δu_temp, u[t - 1]', 1, 1)
            mul!(ΔB, Δu_temp, view(noise, :, t - 1)', 1, 1)
            mul!(ΔC, Δz, u[t]', 1, 1)
        end
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC, u0 = Δu, noise = Δnoise,
                                      observables = NoTangent(), # not implemented
                                      observables_noise = NoTangent()), NoTangent(),
                map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end

function DiffEqBase.__solve(prob::LinearStateSpaceProblem, alg::KalmanFilter, args...; kwargs...)
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)

    # checks on bounds
    @assert size(prob.observables, 2) == T - 1

    @unpack A, B, C, u0_prior_mean, u0_prior_var = prob
    N = length(u0_prior_mean)
    L = size(C, 1)

    # TODO: move to internal algorithm cache
    # This method of preallocation won't work with staticarrays.  Note that we can't use eltype(mean(u0)) since it may be special case of FillArrays.zeros
    u = [Vector{eltype(u0_prior_var)}(undef, N) for _ in 1:T] # Mean of Kalman filter inferred latent states
    P = [Matrix{eltype(u0_prior_var)}(undef, N, N) for _ in 1:T] # Posterior variance of Kalman filter inferred latent states
    z = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T] # Mean of observables, generated from mean of latent states

    # TODO: these intermediates should be of size T-1 instead as the first was skipped.  Left in for checks on timing
    # Maintaining allocations for these intermediates is necessary for the rrule, but not for forward only.  Code could be refactored along those lines with solid unit tests.
    B_prod = Matrix{eltype(u0_prior_var)}(undef, N, N)
    u_mid = [Vector{eltype(u0_prior_var)}(undef, N) for _ in 1:T] # intermediate in u calculation
    P_mid = [Matrix{eltype(u0_prior_var)}(undef, N, N) for _ in 1:T] # intermediate in P calculation
    innovation = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T]
    K = [Matrix{eltype(u0_prior_var)}(undef, N, L) for _ in 1:T] # Gain
    CP = [Matrix{eltype(u0_prior_var)}(undef, L, N) for _ in 1:T] # C * P[t]
    V = [PDMat{eltype(u0_prior_var),Matrix{eltype(u0_prior_var)}}(L,
                                                                  Matrix{eltype(u0_prior_var)}(undef,
                                                                                               L,
                                                                                               L),
                                                                  Cholesky{eltype(u0_prior_var),
                                                                           Matrix{eltype(u0_prior_var)}}(Matrix{eltype(u0_prior_var)}(undef,
                                                                                                                                      L,
                                                                                                                                      L),
                                                                                                         'U',
                                                                                                         0))
         for _ in 1:T] # preallocated buffers for cholesky and matrix itself

    R = prob.observables_noise # assume a matrix for now.  Later can make more general
    mul!(B_prod, B, B')

    # Gaussian Prior
    u[1] .= u0_prior_mean
    P[1] .= u0_prior_var
    z[1] .= C * u[1]

    loglik = 0.0

    # temp buffers.  Could be moved into algorithm settings
    temp_N_N = Matrix{eltype(u0_prior_var)}(undef, N, N)
    temp_L_L = Matrix{eltype(u0_prior_var)}(undef, L, L)
    temp_L_N = Matrix{eltype(u0_prior_var)}(undef, L, N)

    retcode = :Failure
    try
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
            transpose!(temp_L_L, V[t].mat)
            V[t].mat .+= temp_L_L
            lmul!(0.5, V[t].mat)

            copy!(V[t].chol.factors, V[t].mat) # copy over to the factors for the cholesky and do in place
            cholesky!(V[t].chol.factors, Val(false); check = false) # inplace uses V_t with cholesky.  Now V[t]'s chol is upper-triangular        
            innovation[t] .= prob.observables[:, t - 1] - z[t]
            loglik += logpdf(MvNormal(V[t]), innovation[t])  # no allocations since V[t] is a PDMat

            # K[t] .= CP[t]' / V[t]  # Kalman gain
            # Can rewrite as K[t]' = V[t] \ CP[t] since V[t] is symmetric
            ldiv!(temp_L_N, V[t].chol, CP[t])
            transpose!(K[t], temp_L_N)

            #u[t] += K[t] * innovation[t]
            copy!(u[t], u_mid[t])
            mul!(u[t], K[t], innovation[t], 1, 1)

            #P[t] -= K[t] * CP[t]
            copy!(P[t], P_mid[t])
            mul!(P[t], K[t], CP[t], -1, 1)
        end
        retcode = :Success
    catch e
        loglik = -Inf
    end

    t_values = prob.tspan[1]:prob.tspan[2]
    return build_solution(prob, alg, t_values, u; P, W = nothing, logpdf = loglik, z,
                          retcode)
end

# NOTE: when moving to ._concrete_solve_adjoint will need to be careful to ensure the u0 sensitivity
# takes into account any promotion in the `remake_model` side.  We want u0 to be the prior and have the
# sensitivity of it as a distribution, not a draw from it which might happen in the remake(...)

# function DiffEqBase._concrete_solve_adjoint(prob::LinearStateSpaceProblem, alg::KalmanFilter,
#                                             sensealg, u0, p, args...; kwargs...)
function ChainRulesCore.rrule(::typeof(DiffEqBase.solve), prob::LinearStateSpaceProblem,
                              alg::KalmanFilter, args...; kwargs...)
    # Preallocate values
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    # checks on bounds
    @assert size(prob.observables, 2) == T - 1

    @unpack A, B, C, u0_prior_mean, u0_prior_var = prob
    N = length(u0_prior_mean)
    N = length(u0)
    L = size(C, 1)

    # TODO: move to internal algorithm cache
    # This method of preallocation won't work with staticarrays.  Note that we can't use eltype(mean(u0)) since it may be special case of FillArrays.zeros
    B_prod = Matrix{eltype(u0_prior_var)}(undef, N, N)
    u = [Vector{eltype(u0_prior_var)}(undef, N) for _ in 1:T] # Mean of Kalman filter inferred latent states
    P = [Matrix{eltype(u0_prior_var)}(undef, N, N) for _ in 1:T] # Posterior variance of Kalman filter inferred latent states
    z = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T] # Mean of observables, generated from mean of latent states

    # TODO: these intermediates should be of size T-1 instead as the first was skipped.  Left in for checks on timing
    # Maintaining allocations for these intermediates is necessary for the rrule, but not for forward only.  Code could be refactored along those lines with solid unit tests.
    u_mid = [Vector{eltype(u0_prior_var)}(undef, N) for _ in 1:T] # intermediate in u calculation
    P_mid = [Matrix{eltype(u0_prior_var)}(undef, N, N) for _ in 1:T] # intermediate in P calculation
    innovation = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1)) for _ in 1:T]
    K = [Matrix{eltype(u0_prior_var)}(undef, N, L) for _ in 1:T] # Gain
    CP = [Matrix{eltype(u0_prior_var)}(undef, L, N) for _ in 1:T] # C * P[t]
    V = [PDMat{eltype(u0_prior_var),Matrix{eltype(u0_prior_var)}}(L,
                                                                  Matrix{eltype(u0_prior_var)}(undef,
                                                                                               L,
                                                                                               L),
                                                                  Cholesky{eltype(u0_prior_var),
                                                                           Matrix{eltype(u0_prior_var)}}(Matrix{eltype(u0_prior_var)}(undef,
                                                                                                                                      L,
                                                                                                                                      L),
                                                                                                         'U',
                                                                                                         0))
         for _ in 1:T] # preallocated buffers for cholesky and matrix itself

    R = prob.observables_noise # assume a matrix for now.  Later can make more general
    mul!(B_prod, B, B')

    u[1] .= u0_prior_mean
    P[1] .= u0_prior_var
    z[1] .= C * u[1]

    loglik = 0.0

    # temp buffers.  Could be moved into algorithm settings
    temp_N_N = Matrix{eltype(u0_prior_var)}(undef, N, N)
    temp_L_L = Matrix{eltype(u0_prior_var)}(undef, L, L)
    temp_L_N = Matrix{eltype(u0_prior_var)}(undef, L, N)
    temp_N_L = Matrix{eltype(u0_prior_var)}(undef, N, L)
    temp_M = Vector{eltype(u0_prior_var)}(undef, L)
    temp_N = Vector{eltype(u0_prior_var)}(undef, N)
    retcode = :Failure
    try
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
            transpose!(temp_L_L, V[t].mat)
            V[t].mat .+= temp_L_L
            lmul!(0.5, V[t].mat)

            copy!(V[t].chol.factors, V[t].mat) # copy over to the factors for the cholesky and do in place
            cholesky!(V[t].chol.factors, Val(false); check = false) # inplace uses V_t with cholesky.  Now V[t]'s chol is upper-triangular        
            innovation[t] .= prob.observables[:, t - 1] - z[t]
            loglik += logpdf(MvNormal(V[t]), innovation[t])  # no allocations since V[t] is a PDMat

            # K[t] .= CP[t]' / V[t]  # Kalman gain
            # Can rewrite as K[t]' = V[t] \ CP[t] since V[t] is symmetric
            ldiv!(temp_L_N, V[t].chol, CP[t])
            transpose!(K[t], temp_L_N)

            #u[t] += K[t] * innovation[t]
            copy!(u[t], u_mid[t])
            mul!(u[t], K[t], innovation[t], 1, 1)

            #P[t] -= K[t] * CP[t]
            copy!(P[t], P_mid[t])
            mul!(P[t], K[t], CP[t], -1, 1)
        end
        retcode = :Success
    catch e
        loglik = -Inf
    end
    t_values = prob.tspan[1]:prob.tspan[2]
    sol = build_solution(prob, alg, t_values, u; P, W = nothing, logpdf = loglik, z,
                         retcode)
    function solve_pb(Δsol)
        # Currently only changes in the logpdf are supported in the rrule
        @assert Δsol.u == ZeroTangent()
        @assert Δsol.W == ZeroTangent()
        @assert Δsol.P == ZeroTangent()
        @assert Δsol.z == ZeroTangent()

        Δlogpdf = Δsol.logpdf

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

        # If it was a failure, just return and hope the gradients are ignored!
        if retcode == :Success
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
                mul!(temp_L_N, inv_V, CP[t])
                mul!(temp_N_L, ΔK, inv_V)
                mul!(ΔV, temp_L_N, temp_N_L, -1, 0)

                mul!(ΔC, ΔCP, P_mid[t]', 1, 1) # ΔC += ΔCP * P_mid[t]'
                mul!(ΔP_mid, C', ΔCP, 1, 1) # ΔP_mid += C' * ΔCP
                mul!(Δz, inv_V, innovation[t], Δlogpdf, 1) # Δz += Δlogpdf * inv_V * innovation[t] # Σ^-1 * (z_obs - z)

                #ΔV -= Δlogpdf * 0.5 * (inv_V - inv_V * innovation[t] * innovation[t]' * inv_V) # -0.5 * (Σ^-1 - Σ^-1(z_obs - z)(z_obx - z)'Σ^-1)
                mul!(temp_M, inv_V, innovation[t])
                mul!(temp_L_L, temp_M, temp_M')
                temp_L_L .-= inv_V
                rmul!(temp_L_L, Δlogpdf * 0.5)
                ΔV += temp_L_L

                #ΔC += ΔV * C * P_mid[t]' + ΔV' * C * P_mid[t]
                mul!(temp_L_N, C, P_mid[t])
                transpose!(temp_L_L, ΔV)
                temp_L_L .+= ΔV
                mul!(ΔC, temp_L_L, temp_L_N, 1, 1)

                # ΔP_mid += C' * ΔV * C
                mul!(temp_L_N, ΔV, C)
                mul!(ΔP_mid, C', temp_L_N, 1, 1)

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
        end
        ΔΣ = Tangent{typeof(prob.u0_prior.Σ)}(; mat = ΔP, chol = NoTangent(), dim = NoTangent()) # TODO: This is not exactly correct since it doesn't do the "chol".  Add to prevent misuse.
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC, u0 = ZeroTangent(), # u0 not used in kalman filter
                                      u0_prior_mean = Δu, u0_prior_var = ΔP),
                NoTangent(), map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
