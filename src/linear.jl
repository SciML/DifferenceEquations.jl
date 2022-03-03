"""
u(t+1) = A u(t) + B w(t+1)
z(t) = C u(t)
z_tilde(t) = z(t) + v(t+1)
"""
struct LinearStateSpaceProblem{isinplace,Atype<:AbstractArray,Btype<:AbstractArray,
                               Ctype<:AbstractArray,wtype,Rtype, # Distributions only
                               utype,ttype,otype} <: AbstractStateSpaceProblem{isinplace}
    A::Atype # Evolution matrix
    B::Btype # Noise matrix
    C::Ctype # Observation matrix
    noise::wtype # Latent noises
    obs_noise::Rtype # Observation noise / measurement error distribution
    u0::utype # Initial condition
    tspan::ttype # Timespan to use
    observables::otype # Observed data to use, if any
end

function LinearStateSpaceProblem(A::Atype, B::Btype, C::Ctype, u0::utype, tspan::ttype;
                                 obs_noise = nothing, # Assume the default measurement error is MvNormal with identity covariance
                                 observables = nothing,
                                 noise = nothing) where {Atype<:AbstractArray,Btype<:AbstractArray,
                                                         Ctype<:AbstractArray,utype,ttype}
    return LinearStateSpaceProblem{Val(false),Atype,Btype,Ctype,typeof(noise),typeof(obs_noise),
                                   utype,ttype,typeof(observables)}(A, # Evolution matrix
                                                                    B, # Noise matrix
                                                                    C, # Observation matrix
                                                                    noise, # Latent noise distribution
                                                                    obs_noise, # Observation noise matrix
                                                                    u0, # Initial condition
                                                                    tspan, # Timespan to use
                                                                    observables)
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

    # checks on bounds
    @assert size(prob.noise, 1) == size(prob.B, 2)
    @assert size(prob.noise, 2) == size(prob.observables, 2)
    @assert size(prob.noise, 2) == T - 1

    u = [zero(prob.u0) for _ in 1:T] # TODO: move to internal algorithm cache
    z1 = C * prob.u0
    z = [zero(z1) for _ in 1:T] # TODO: move to internal algorithm cache

    # Initialize
    u[1] .= prob.u0
    z[1] .= z1

    loglik = 0.0
    @inbounds for t in 2:T
        mul!(u[t], A, u[t - 1])
        mul!(u[t], B, view(prob.noise, :, t - 1), 1, 1)

        mul!(z[t], C, u[t])
        loglik += logpdf(prob.obs_noise, view(prob.observables, :, t - 1) - z[t])
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
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    # checks on bounds
    @assert size(prob.noise, 1) == size(prob.B, 2)
    @assert size(prob.noise, 2) == size(prob.observables, 2)
    @assert size(prob.noise, 2) == T - 1

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
        mul!(u[t], B, view(prob.noise, :, t - 1), 1, 1)

        mul!(z[t], C, u[t])
        loglik += logpdf(prob.obs_noise, view(prob.observables, :, t - 1) - z[t])
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
        Δu = zero(u[1])
        Δu_temp = zero(u[1])
        Δz = zero(z[1])

        @views @inbounds for t in T:-1:2
            Δz .= Δlogpdf * (view(prob.observables, :, t - 1) - z[t]) ./ diag(prob.obs_noise.Σ) # More generally, it should be Σ^-1 * (z_obs - z)
            # TODO: check if this can be repalced with the following and if it has a performance regression for diagonal noise covariance
            # ldiv!(Δz, obs_noise.Σ.chol, innovation[t])
            # rmul!(Δlogpdf, Δz)

            copy!(Δu_temp, Δu)
            mul!(Δu_temp, C', Δz, 1, 1)
            mul!(Δu, A', Δu_temp)
            mul!(view(Δnoise, :, t - 1), B', Δu_temp)
            # Now, deal with the coefficients
            mul!(ΔA, Δu_temp, u[t - 1]', 1, 1)
            mul!(ΔB, Δu_temp, view(prob.noise, :, t - 1)', 1, 1)
            mul!(ΔC, Δz, u[t]', 1, 1)
        end
        return (NoTangent(),
                Tangent{typeof(prob)}(; A = ΔA, B = ΔB, C = ΔC, u0 = Δu, noise = Δnoise,
                                      observables = NoTangent(), # not implemented
                                      obs_noise = NoTangent()), NoTangent(),
                map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
