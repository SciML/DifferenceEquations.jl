
abstract type DifferenceProblem <: SciMLProblem end
abstract type AbstractStateSpaceProblem{isinplace} <: DifferenceProblem end

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

"""
u_f(t+1) = A_1 u_f(t) .+ B * noise(t+1)
u(t+1) = A_0 + A_1 u(t) + quad(A_2, u_f(t)) .+ B noise(t+1)
z(t) = C_0 + C_1 u(t) + quad(C_2, u_f(t))
z_tilde(t) = z(t) + v(t+1)
"""
struct QuadraticStateSpaceProblem{isinplace,A_0type<:AbstractArray,A_1type<:AbstractArray,
                                  A_2type<:AbstractArray,Btype<:AbstractArray,
                                  C_0type<:AbstractArray,C_1type<:AbstractArray,
                                  C_2type<:AbstractArray,wtype,Rtype, # Distributions only
                                  utype,ttype,otype} <: AbstractStateSpaceProblem{isinplace}
    A_0::A_0type
    A_1::A_1type
    A_2::A_2type # Evolution matrix
    B::Btype # Noise matrix
    C_0::C_0type
    C_1::C_1type
    C_2::C_2type # Observation matrix
    noise::wtype # Latent noises
    obs_noise::Rtype # Observation noise / measurement error distribution
    u0::utype # Initial condition
    tspan::ttype # Timespan to use
    observables::otype # Observed data to use, if any
end

function QuadraticStateSpaceProblem(A_0::A_0type, A_1::A_1type, A_2::A_2type, B::Btype,
                                    C_0::C_0type, C_1::C_1type, C_2::C_2type, u0::utype,
                                    tspan::ttype; obs_noise = nothing, observables = nothing,
                                    noise = nothing) where {A_0type<:AbstractArray,
                                                            A_1type<:AbstractArray,
                                                            A_2type<:AbstractArray,
                                                            Btype<:AbstractArray,
                                                            C_0type<:AbstractArray,
                                                            C_1type<:AbstractArray,
                                                            C_2type<:AbstractArray,utype,ttype}
    return QuadraticStateSpaceProblem{Val(false),A_0type,A_1type,A_2type,Btype,C_0type,C_1type,
                                      C_2type,typeof(noise),typeof(obs_noise),utype,ttype,
                                      typeof(observables)}(A_0::A_0type, A_1::A_1type, A_2::A_2type, # Evolution matrix
                                                           B::Btype, # Noise matrix
                                                           C_0::C_0type, C_1::C_1type, C_2::C_2type, # Observation matrix
                                                           noise, # Latent noise distribution
                                                           obs_noise, # Observation noise matrix
                                                           u0, # Initial condition
                                                           tspan, # Timespan to use
                                                           observables)
end

# Default is NoiseConditionalFilter
function CommonSolve.init(prob::QuadraticStateSpaceProblem, args...; kwargs...)
    return StateSpaceCache(prob, NoiseConditionalFilter())
end

function CommonSolve.init(prob::QuadraticStateSpaceProblem, solver::SciMLBase.SciMLAlgorithm,
                          args...; kwargs...)
    return StateSpaceCache(prob, solver)
end
