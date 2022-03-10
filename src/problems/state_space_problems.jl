abstract type AbstractStateSpaceProblem <: DiffEqBase.DEProblem end
struct LinearStateSpaceProblem{uType,uPriorType,tType,P,NP,AType,BType,CType,RType,ObsType,K,
                               SymsType} <: AbstractStateSpaceProblem
    f::Nothing  # HACK: need something called "f" which supports "isinplace" for plotting
    A::AType
    B::BType
    C::CType
    R::RType
    observables::ObsType
    u0::uType
    u0_prior::uPriorType
    tspan::tType
    p::P
    noise::NP
    kwargs::K
    seed::UInt64
    syms::SymsType
    @add_kwonly function LinearStateSpaceProblem{iip}(A, B, u0, tspan, p = NullParameters();
                                                      f = nothing, u0_prior = u0, # default copies the u0 to u0_prior so it doesn't mess up get_concrete
                                                      C = nothing, R = nothing,
                                                      observables = nothing, noise = nothing,
                                                      seed = UInt64(0), syms = nothing,
                                                      kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        _observables = promote_vv(observables)
        _noise = promote_vv(noise)

        return new{typeof(u0),typeof(u0_prior),typeof(_tspan),typeof(p),typeof(_noise),typeof(A),
                   typeof(B),typeof(C),typeof(R),typeof(_observables),typeof(kwargs),typeof(syms)}(f,
                                                                                                   A,
                                                                                                   B,
                                                                                                   C,
                                                                                                   R,
                                                                                                   _observables,
                                                                                                   u0,
                                                                                                   u0_prior,
                                                                                                   _tspan,
                                                                                                   p,
                                                                                                   _noise,
                                                                                                   kwargs,
                                                                                                   seed,
                                                                                                   syms)
    end
end
# just forwards to a iip = false case
LinearStateSpaceProblem(args...; kwargs...) = LinearStateSpaceProblem{false}(args...; kwargs...)

# """
# u_f(t+1) = A_1 u_f(t) .+ B * noise(t+1)
# u(t+1) = A_0 + A_1 u(t) + quad(A_2, u_f(t)) .+ B noise(t+1)
# z(t) = C_0 + C_1 u(t) + quad(C_2, u_f(t))
# z_tilde(t) = z(t) + v(t+1)
# """
# struct QuadraticStateSpaceProblem{isinplace,A_0type<:AbstractArray,A_1type<:AbstractArray,
#                                   A_2type<:AbstractArray,Btype<:AbstractArray,
#                                   C_0type<:AbstractArray,C_1type<:AbstractArray,
#                                   C_2type<:AbstractArray,wtype,Rtype, # Distributions only
#                                   utype,ttype,otype} <: AbstractStateSpaceProblem{isinplace}
#     A_0::A_0type
#     A_1::A_1type
#     A_2::A_2type # Evolution matrix
#     B::Btype # Noise matrix
#     C_0::C_0type
#     C_1::C_1type
#     C_2::C_2type # Observation matrix
#     noise::wtype # Latent noises
#     obs_noise::Rtype # Observation noise / measurement error distribution
#     u0::utype # Initial condition
#     tspan::ttype # Timespan to use
#     observables::otype # Observed data to use, if any
# end

# function QuadraticStateSpaceProblem(A_0::A_0type, A_1::A_1type, A_2::A_2type, B::Btype,
#                                     C_0::C_0type, C_1::C_1type, C_2::C_2type, u0::utype,
#                                     tspan::ttype; obs_noise = nothing, observables = nothing,
#                                     noise = nothing) where {A_0type<:AbstractArray,
#                                                             A_1type<:AbstractArray,
#                                                             A_2type<:AbstractArray,
#                                                             Btype<:AbstractArray,
#                                                             C_0type<:AbstractArray,
#                                                             C_1type<:AbstractArray,
#                                                             C_2type<:AbstractArray,utype,ttype}
#     return QuadraticStateSpaceProblem{Val(false),A_0type,A_1type,A_2type,Btype,C_0type,C_1type,
#                                       C_2type,typeof(noise),typeof(obs_noise),utype,ttype,
#                                       typeof(observables)}(A_0::A_0type, A_1::A_1type, A_2::A_2type, # Evolution matrix
#                                                            B::Btype, # Noise matrix
#                                                            C_0::C_0type, C_1::C_1type, C_2::C_2type, # Observation matrix
#                                                            noise, # Latent noise distribution
#                                                            obs_noise, # Observation noise matrix
#                                                            u0, # Initial condition
#                                                            tspan, # Timespan to use
#                                                            observables)
# end

# # Default is NoiseConditionalFilter
# function CommonSolve.init(prob::QuadraticStateSpaceProblem, args...; kwargs...)
#     return StateSpaceCache(prob)
# end

# function CommonSolve.init(prob::QuadraticStateSpaceProblem, solver::SciMLBase.SciMLAlgorithm,
#                           args...; kwargs...)
#     return StateSpaceCache(prob, solver)
# end
