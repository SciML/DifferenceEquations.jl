#Many of these utilities shoudl be moved to utilities.jl after testing is complete with the 2nd order.

# Temporary.  Eventually, move to use sciml NoiseProcess with better rng support/etc.
get_concrete_noise(prob, noise, B, T) = noise # maybe do a promotion to an AbstractVectorOfVector type
get_concrete_noise(prob, noise, B::Nothing, T) = nothing # if no noise matrix given, do not create noise
get_concrete_noise(prob, noise::Nothing, B::Nothing, T) = nothing # if no noise matrix given, do not create noise
get_concrete_noise(prob, noise::Nothing, B, T) = randn(eltype(B), size(B, 2), T) # default is unit Gaussian
get_concrete_noise(prob, noise::UnivariateDistribution, B, T) = rand(noise, size(B, 2), T) # iid

# Utility functions to conditionally check size if not-nothing
maybe_check_size(m::AbstractMatrix, index, val) = (size(m, index) == val)
maybe_check_size(m::Nothing, index, val) = true

function maybe_check_size(m1::AbstractArray, index1, m2::AbstractArray, index2)
    (size(m1, index1) ==
     size(m2, index2))
end
maybe_check_size(m1::Nothing, index1, m2, index2) = true
maybe_check_size(m1, index1, m2::Nothing, index2) = true
maybe_check_size(m1::Nothing, index1, m2::Nothing, index2) = true

Base.@propagate_inbounds @inline function maybe_logpdf(observables_noise::Distribution,
                                                       observables::AbstractMatrix, t,
                                                       z::AbstractVector, s)
    logpdf(observables_noise,
           view(observables,
                :,
                t) -
           z[s])
end
# Don't accumulate likelihoods if no observations or observatino noise
maybe_logpdf(observables_noise, observable, t, z, s) = 0.0

# If no noise process is given, don't add in noise in simulation
Base.@propagate_inbounds @inline function maybe_muladd!(x, B, noise, t)
    mul!(x, B, view(noise, :, t), 1, 1)
end
maybe_muladd!(x, B::Nothing, noise, t) = nothing

Base.@propagate_inbounds @inline maybe_muladd!(x, A, B) = mul!(x, A, B, 1, 1)
maybe_muladd!(x, A::Nothing, B) = nothing

# need transpose versions for gradients
Base.@propagate_inbounds @inline maybe_muladd_transpose!(x, C, Δz) = mul!(x, C', Δz, 1, 1)
maybe_muladd_transpose!(x, C::Nothing, Δz) = nothing
Base.@propagate_inbounds @inline function maybe_muladd_transpose!(ΔB::AbstractMatrix,
                                                                  Δu_temp,
                                                                  noise::AbstractMatrix, t)
    mul!(ΔB, Δu_temp, view(noise, :, t)', 1, 1)
    return nothing
end
maybe_muladd_transpose!(ΔB, Δu_temp, noise, t) = nothing
Base.@propagate_inbounds @inline maybe_mul!(x, t, A, y, s) = mul!(x[t], A, y[s])
maybe_mul!(x::Nothing, t, A, y, s) = nothing
# Need transpose versions for rrule
Base.@propagate_inbounds @inline maybe_mul_transpose!(x, t, A, y, s) = mul!(x[t], A', y[s])
maybe_mul_transpose!(x::Nothing, t, A, y, s) = nothing
Base.@propagate_inbounds @inline function maybe_mul_transpose!(Δnoise, t, B, y)
    mul!(view(Δnoise, :, t),
         B', y)
end
maybe_mul_transpose!(Δnoise::Nothing, t, B, y) = nothing

# Utilities to get distribution for logpdf from observation error argument
make_observables_noise(observables_noise::Nothing) = nothing
make_observables_noise(observables_noise::AbstractMatrix) = MvNormal(observables_noise)
function make_observables_noise(observables_noise::AbstractVector)
    MvNormal(Diagonal(observables_noise))
end

# Utilities to get covariance matrix from observation error argument for kalman filter.  e.g. vector is diagonal, etc.
make_observables_covariance_matrix(observables_noise::AbstractMatrix) = observables_noise
function make_observables_covariance_matrix(observables_noise::AbstractVector)
    Diagonal(observables_noise)
end

#Add in observation noise to the output if simulated (i.e, observables not given) and there is observation_noise provided
function maybe_add_observation_noise!(z, observables_noise::Distribution,
                                      observables::Nothing)
    # add noise to the vector of vectors componentwise
    for z_val in z
        z_val .+= rand(observables_noise)
    end
    return nothing
end
maybe_add_observation_noise!(z, observables_noise, observables) = nothing  #otherwise do nothing

#Maybe add observation noise, if observables and their adjoints given
Base.@propagate_inbounds @inline function maybe_add_Δ!(Δz, Δsol_z::AbstractVector, t)
    Δz .+= Δsol_z[t]
    return nothing
end
maybe_add_Δ!(Δz, Δsol_z, t) = nothing

Base.@propagate_inbounds @inline function maybe_add_Δ_slice!(Δnoise::AbstractMatrix,
                                                             ΔW::AbstractMatrix, t)
    Δnoise[:, t] .+= view(ΔW, :, t)
    return nothing
end
maybe_add_Δ_slice!(Δz, Δsol_A, t) = nothing

# Don't add logpdf to observables unless provided
# TODO: check if this can be repalced with the following and if it has a performance regression for diagonal noise covariance
# ldiv!(Δz, observables_noise.Σ.chol, innovation[t])
# rmul!(Δlogpdf, Δz)
Base.@propagate_inbounds @inline function maybe_add_Δ_logpdf!(Δz::AbstractArray{<:Real, 1},
                                                              Δlogpdf::Number,
                                                              observables::AbstractArray{
                                                                                         <:Real,
                                                                                         2},
                                                              z::AbstractArray{T, 1},
                                                              t,
                                                              observables_noise_cov::AbstractArray{
                                                                                                   <:Real,
                                                                                                   1
                                                                                                   }) where {
                                                                                                             T
                                                                                                             }
    Δz .= Δlogpdf * (view(observables, :, t - 1) - z[t]) ./
          observables_noise_cov
    return nothing
end
# Otherwise do nothing
function maybe_add_Δ_logpdf!(Δz, Δlogpdf, observables, z, t, observables_noise_cov)
    nothing
end

# Only allocate if observation equation
allocate_z(prob, C, u0, T) = [zeros(size(C, 1)) for _ in 1:T]
allocate_z(prob, C::Nothing, u0, T) = nothing

# Maybe zero
maybe_zero(A::AbstractArray) = zero(A)
maybe_zero(A::Nothing) = nothing
maybe_zero(A::AbstractArray, i::Int64) = zero(A[i])
maybe_zero(A::Nothing, i) = nothing

function DiffEqBase.__solve(prob::LinearStateSpaceProblem{uType, uPriorMeanType,
                                                          uPriorVarType, tType,
                                                          P, NP, F, AType, BType,
                                                          CType, RType, ObsType, K},
                            alg::DirectIteration, args...;
                            kwargs...) where {uType, uPriorMeanType, uPriorVarType, tType,
                                              P, NP, F, AType,
                                              BType, CType, RType,
                                              ObsType, K}
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    @unpack A, B, C = prob

    # checks on bounds
    noise = get_concrete_noise(prob, prob.noise, prob.B, T - 1)  # concrete noise for simulations as required.
    observables_noise = make_observables_noise(prob.observables_noise)

    @assert maybe_check_size(noise, 1, prob.B, 2)
    @assert maybe_check_size(noise, 2, T - 1)
    @assert maybe_check_size(prob.observables, 2, T - 1)

    # Initialize
    u = [zero(prob.u0) for _ in 1:T]
    u[1] .= prob.u0

    z = allocate_z(prob, C, prob.u0, T)
    maybe_mul!(z, 1, C, u, 1)  # update the first of z if it isn't nothing

    loglik = 0.0
    @inbounds for t in 2:T
        mul!(u[t], A, u[t - 1])
        maybe_muladd!(u[t], B, noise, t - 1) # was:  mul!(u[t], B, view(noise, :, t - 1), 1, 1)

        maybe_mul!(z, t, C, u, t)  # does mul!(z[t], C, u[t]) if C is not nothing
        loglik += maybe_logpdf(observables_noise, prob.observables, t - 1, z, t)
    end
    maybe_add_observation_noise!(z, observables_noise, prob.observables)
    t_values = prob.tspan[1]:prob.tspan[2]

    return build_solution(prob, alg, t_values, u; W = noise,
                          logpdf = ObsType <: Nothing ? nothing : loglik, z,
                          retcode = :Success)
end

# Ideally hook into existing sensitity dispatching
# Trouble with Zygote.  The problem isn't the _concrete_solve_adjoint but rather something in the
# adjoint of the basic solve and `solve_up`.  Probably promotion on the prob

# function DiffEqBase._concrete_solve_adjoint(prob::LinearStateSpaceProblem, alg::DirectIteration,
#                                             sensealg, u0, p, args...; kwargs...)
function ChainRulesCore.rrule(::typeof(DiffEqBase.solve),
                              prob::LinearStateSpaceProblem{uType, uPriorMeanType,
                                                            uPriorVarType,
                                                            tType,
                                                            P, NP, F, AType, BType,
                                                            CType, RType, ObsType, K},
                              alg::DirectIteration, args...;
                              kwargs...) where {uType, uPriorMeanType, uPriorVarType, tType,
                                                P, NP, F,
                                                AType,
                                                BType, CType, RType,
                                                ObsType, K}
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    @unpack A, B, C = prob
    # @assert !isnothing(prob.noise) || isnothing(prob.B)  # need to have concrete noise or no noise for this simple method

    # checks on bounds
    noise = get_concrete_noise(prob, prob.noise, prob.B, T - 1)  # concrete noise for simulations as required.
    observables_noise = make_observables_noise(prob.observables_noise)
    @assert typeof(observables_noise) <: Union{ZeroMeanDiagNormal, Nothing}  # can extend to more general in rrule later

    @assert maybe_check_size(noise, 1, prob.B, 2)
    @assert maybe_check_size(noise, 2, T - 1)
    @assert maybe_check_size(prob.observables, 2, T - 1)

    # Initialize
    u = [zero(prob.u0) for _ in 1:T]
    u[1] .= prob.u0

    z = allocate_z(prob, C, prob.u0, T)
    maybe_mul!(z, 1, C, u, 1)  # update the first of z if it isn't nothing

    loglik = 0.0
    @inbounds for t in 2:T
        mul!(u[t], A, u[t - 1])
        maybe_muladd!(u[t], B, noise, t - 1) # was:  mul!(u[t], B, view(noise, :, t - 1), 1, 1)

        maybe_mul!(z, t, C, u, t)  # does mul!(z[t], C, u[t]) if C is not nothing
        loglik += maybe_logpdf(observables_noise, prob.observables, t - 1, z, t)
    end
    maybe_add_observation_noise!(z, observables_noise, prob.observables)
    t_values = prob.tspan[1]:prob.tspan[2]

    sol = build_solution(prob, alg, t_values, u; W = noise,
                         logpdf = ObsType <: Nothing ? nothing : loglik, z,
                         retcode = :Success)
    function solve_pb(Δsol)
        ΔA = zero(A)
        ΔB = maybe_zero(B)
        ΔC = maybe_zero(C)
        Δnoise = maybe_zero(noise)
        Δu = zero(u[1])
        Δu_temp = zero(u[1])

        # Assert checked above about being diagonal and Normal
        observables_noise_cov = prob.observables_noise

        @views @inbounds for t in T:-1:2
            Δz = maybe_zero(z, 1) # zero out in case no logpdf but z observations available 
            maybe_add_Δ_logpdf!(Δz, Δsol.logpdf, prob.observables, z, t,
                                observables_noise_cov)
            maybe_add_Δ!(Δz, Δsol.z, t)  # only accumulte if z provided

            copy!(Δu_temp, Δu)
            maybe_muladd_transpose!(Δu_temp, C, Δz) # mul!(Δu_temp, C', Δz, 1, 1)
            maybe_add_Δ!(Δu_temp, Δsol.u, t)  # only accumulate if not NoTangent and if observables provided
            mul!(Δu, A', Δu_temp)
            maybe_mul_transpose!(Δnoise, t - 1, B, Δu_temp)
            maybe_add_Δ_slice!(Δnoise, Δsol.W, t - 1)
            mul!(ΔA, Δu_temp, u[t - 1]', 1, 1)
            maybe_muladd_transpose!(ΔB, Δu_temp, noise, t - 1)
            maybe_muladd!(ΔC, Δz, u[t]')
        end
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC, u0 = Δu, noise = Δnoise,
                                      observables = NoTangent(), # not implemented
                                      observables_noise = NoTangent()), NoTangent(),
                map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end

function DiffEqBase.__solve(prob::LinearStateSpaceProblem, alg::KalmanFilter, args...;
                            kwargs...)
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
    innovation = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1))
                  for _ in 1:T]
    K = [Matrix{eltype(u0_prior_var)}(undef, N, L) for _ in 1:T] # Gain
    CP = [Matrix{eltype(u0_prior_var)}(undef, L, N) for _ in 1:T] # C * P[t]
    V = [PDMat{eltype(u0_prior_var), Matrix{eltype(u0_prior_var)}}(L,
                                                                   Matrix{
                                                                          eltype(u0_prior_var)
                                                                          }(undef,
                                                                            L,
                                                                            L),
                                                                   Cholesky{
                                                                            eltype(u0_prior_var),
                                                                            Matrix{
                                                                                   eltype(u0_prior_var)
                                                                                   }}(Matrix{
                                                                                             eltype(u0_prior_var)
                                                                                             }(undef,
                                                                                               L,
                                                                                               L),
                                                                                      'U',
                                                                                      0))
         for _ in 1:T] # preallocated buffers for cholesky and matrix itself

    R = make_observables_covariance_matrix(prob.observables_noise)  # Support diagonal or matrix covariance matrices.
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
    innovation = [Vector{eltype(prob.observables)}(undef, size(prob.observables, 1))
                  for _ in 1:T]
    K = [Matrix{eltype(u0_prior_var)}(undef, N, L) for _ in 1:T] # Gain
    CP = [Matrix{eltype(u0_prior_var)}(undef, L, N) for _ in 1:T] # C * P[t]
    V = [PDMat{eltype(u0_prior_var), Matrix{eltype(u0_prior_var)}}(L,
                                                                   Matrix{
                                                                          eltype(u0_prior_var)
                                                                          }(undef,
                                                                            L,
                                                                            L),
                                                                   Cholesky{
                                                                            eltype(u0_prior_var),
                                                                            Matrix{
                                                                                   eltype(u0_prior_var)
                                                                                   }}(Matrix{
                                                                                             eltype(u0_prior_var)
                                                                                             }(undef,
                                                                                               L,
                                                                                               L),
                                                                                      'U',
                                                                                      0))
         for _ in 1:T] # preallocated buffers for cholesky and matrix itself

    R = make_observables_covariance_matrix(prob.observables_noise)  # Support diagonal or matrix covariance matrices.
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
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC, u0 = ZeroTangent(), # u0 not used in kalman filter
                                      u0_prior_mean = Δu, u0_prior_var = ΔP),
                NoTangent(), map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
