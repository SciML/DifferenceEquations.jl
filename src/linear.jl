"""
u(t+1) = A u(t) + B w(t+1)
z(t) = C u(t)
z_tilde(t) = z(t) + v(t+1)
"""

# Includes both Kalman and first-order joint.  Can make kalman filter optional later
Base.@kwdef struct LinearStateSpaceProblemCache{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,
                                                T16}
    u::T1
    z::T2
    innovation::T7

    # Only required for the Kalman filter
    B_prod::T3
    P::T4
    u_temp::T5
    P_temp::T6
    K::T8
    CP::T9
    V::T10
    temp_N_N::T11
    temp_L_L::T12
    temp_L_N::T13
    temp_N_L::T14
    temp_L::T15
    temp_N::T16
end
# Not setup to support staticarrays/etc. at this point
# Maintaining allocations for some intermediates is necessary for the rrule, but not for forward only.  Could be refactored later
# N states, M shocks, L observables
function LinearStateSpaceProblemCache{DT}(N, M, L, T,
                                          ::Val{AllocateKalman} = Val(true)) where {DT,
                                                                                    AllocateKalman}
    return LinearStateSpaceProblemCache(; u = [Vector{DT}(undef, N) for _ in 1:T],
                                        z = [Vector{DT}(undef, L) for _ in 1:T],
                                        innovation = [Vector{DT}(undef, L) for _ in 1:T],
                                        # kalman filter cache
                                        B_prod = AllocateKalman ? Matrix{DT}(undef, N, N) : nothing,
                                        P = AllocateKalman ?
                                            [Matrix{DT}(undef, N, N) for _ in 1:T] : nothing,
                                        u_temp = AllocateKalman ?
                                                 [Vector{DT}(undef, N) for _ in 1:T] : nothing,
                                        P_temp = AllocateKalman ?
                                                 [Matrix{DT}(undef, N, N) for _ in 1:T] : nothing,
                                        K = AllocateKalman ?
                                            [Matrix{DT}(undef, N, L) for _ in 1:T] : nothing,
                                        CP = AllocateKalman ?
                                             [Matrix{DT}(undef, L, N) for _ in 1:T] : nothing,
                                        V = AllocateKalman ?
                                            [PDMat{DT,Matrix{DT}}(L, Matrix{DT}(undef, L, L),
                                                                  Cholesky{DT,Matrix{DT}}(Matrix{DT}(undef,
                                                                                                     L,
                                                                                                     L),
                                                                                          'U', 0))
                                             for _ in 1:T] : nothing,# caches for cholesky and matrix itself
                                        temp_N_N = AllocateKalman ? Matrix{DT}(undef, N, N) :
                                                   nothing,
                                        temp_L_L = AllocateKalman ? Matrix{DT}(undef, L, L) :
                                                   nothing,
                                        temp_L_N = AllocateKalman ? Matrix{DT}(undef, L, N) :
                                                   nothing,
                                        temp_N_L = AllocateKalman ? Matrix{DT}(undef, N, L) :
                                                   nothing,
                                        temp_L = AllocateKalman ? Vector{DT}(undef, L) : nothing,
                                        temp_N = AllocateKalman ? Vector{DT}(undef, N) : nothing)
end

struct LinearStateSpaceProblem{isinplace,Atype<:AbstractArray,Btype<:AbstractArray,
                               Ctype<:AbstractArray,wtype,Rtype, # Distributions only
                               utype,ttype,otype,ctype} <: AbstractStateSpaceProblem{isinplace}
    A::Atype # Evolution matrix
    B::Btype # Noise matrix
    C::Ctype # Observation matrix
    noise::wtype # Latent noises
    obs_noise::Rtype # Observation noise / measurement error distribution
    u0::utype # Initial condition
    tspan::ttype # Timespan to use
    observables::otype # Observed data to use, if any
    cache::ctype # cache
end

function LinearStateSpaceProblem(A::Atype, B::Btype, C::Ctype, u0::utype, tspan::ttype,
                                 ::Val{AllocateKalman} = Val(true);
                                 obs_noise = (h0 = C * u0;
                                              MvNormal(zeros(eltype(h0), length(h0)), I)), # Assume the default measurement error is MvNormal with identity covariance
                                 observables = nothing, noise = nothing,
                                 cache = LinearStateSpaceProblemCache{eltype(u0)}(length(u0),
                                                                                  size(B, 2),
                                                                                  size(observables,
                                                                                       1),
                                                                                  size(observables,
                                                                                       2) + 1,
                                                                                  Val(AllocateKalman))) where {Atype<:AbstractArray,
                                                                                                               Btype<:AbstractArray,
                                                                                                               Ctype<:AbstractArray,
                                                                                                               utype,
                                                                                                               ttype,
                                                                                                               AllocateKalman}
    return LinearStateSpaceProblem{Val(false),Atype,Btype,Ctype,typeof(noise),typeof(obs_noise),
                                   utype,ttype,typeof(observables),typeof(cache)}(A, # Evolution matrix
                                                                                  B, # Noise matrix
                                                                                  C, # Observation matrix
                                                                                  noise, # Latent noise distribution
                                                                                  obs_noise, # Observation noise matrix
                                                                                  u0, # Initial condition
                                                                                  tspan, # Timespan to use
                                                                                  observables,
                                                                                  cache)
end

# Default is NoiseConditionalFilter
function CommonSolve.init(prob::LinearStateSpaceProblem, args...; kwargs...)
    return StateSpaceCache(prob, NoiseConditionalFilter())
end

function CommonSolve.init(prob::LinearStateSpaceProblem, solver::SciMLBase.SciMLAlgorithm, args...;
                          kwargs...)
    return StateSpaceCache(prob, solver)
end

function _solve!(prob::LinearStateSpaceProblem{isinplace,Atype,Btype,Ctype,wtype,Rtype,utype,ttype,
                                               otype}, ::NoiseConditionalFilter, args...;
                 kwargs...) where {isinplace,Atype,Btype,Ctype,wtype,Rtype,utype,ttype,otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    @unpack A, B, C = prob
    @unpack u, z, innovation = prob.cache  # problem cache
    @assert length(u) >= T && length(z) >= T && length(innovation) >= T # ensure enough space allocated

    z1 = C * prob.u0

    # Initialize
    u[1] .= prob.u0
    z[1] .= z1

    loglik = 0.0
    @inbounds for t in 2:T
        mul!(u[t], A, u[t - 1])
        mul!(u[t], B, prob.noise[:, t - 1], 1, 1)
        mul!(z[t], C, u[t])
        innovation[t] .= prob.observables[:, t - 1] - z[t]
        loglik += logpdf(prob.obs_noise, innovation[t])
    end

    return StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
end

function ChainRulesCore.rrule(::typeof(_solve!),
                              prob::LinearStateSpaceProblem{isinplace,Atype,Btype,Ctype,wtype,Rtype,
                                                            utype,ttype,otype},
                              ::NoiseConditionalFilter, args...;
                              kwargs...) where {isinplace,Atype,Btype,Ctype,wtype,Rtype,utype,ttype,
                                                otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    @unpack A, B, C = prob
    @unpack u, z, innovation = prob.cache  # problem cache
    @assert length(u) >= T && length(z) >= T && length(innovation) >= T # ensure enough space allocated

    z1 = C * prob.u0

    # Initialize
    u[1] .= prob.u0
    z[1] .= z1

    loglik = 0.0
    @inbounds for t in 2:T
        mul!(u[t], A, u[t - 1])
        mul!(u[t], B, prob.noise[:, t - 1], 1, 1)
        mul!(z[t], C, u[t])
        innovation[t] .= prob.observables[:, t - 1] - z[t]
        loglik += logpdf(prob.obs_noise, innovation[t])
    end

    sol = StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)

    function solve_pb(Δsol)
        Δlogpdf = Δsol.loglikelihood
        if iszero(Δlogpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end
        ΔA = zero(A)
        ΔB = zero(B)
        ΔC = zero(C)
        Δnoise = similar(prob.noise)
        Δu = [zero(prob.u0) for _ in 1:T]  # TODO: doesn't need to allocate entire vector, see kalman filter implementation
        Δz = zero(z[1])
        @inbounds for t in T:-1:2
            Δz .= Δlogpdf * innovation[t] ./ diag(prob.obs_noise.Σ) # More generally, it should be Σ^-1 * (z_obs - z)
            # TODO: check if this can be repalced with the following and if it has a performance regression for diagonal noise covariance
            # ldiv!(Δz, obs_noise.Σ.chol, innovation[t])
            # rmul!(Δlogpdf, Δz)

            mul!(Δu[t], C', Δz, 1, 1)
            mul!(Δu[t - 1], A', Δu[t])
            mul!(view(Δnoise, :, t - 1), B', Δu[t])
            # Now, deal with the coefficients
            mul!(ΔA, Δu[t], u[t - 1]', 1, 1)
            mul!(ΔB, Δu[t], prob.noise[:, t - 1]', 1, 1)
            mul!(ΔC, Δz, u[t]', 1, 1)
        end
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC, u0 = Δu[1], noise = Δnoise),
                NoTangent(), map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
