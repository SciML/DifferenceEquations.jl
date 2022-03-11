abstract type AbstractStateSpaceProblem <: DiffEqBase.DEProblem end
abstract type AbstractPerturbationProblem <: AbstractStateSpaceProblem end

using DiffEqBase: get_concrete_tspan, get_concrete_u0, get_concrete_p, promote_u0, promote_tspan,
                  isconcreteu0
# Perturbation problesm don't have f, g
# In discrete time, tspan should not have a sensitivity so the concretization is less obvious
function DiffEqBase.get_concrete_problem(prob::AbstractPerturbationProblem, isadapt; kwargs...)
    p = DiffEqBase.get_concrete_p(prob, kwargs)
    tspan = prob.tspan #get_concrete_tspan(prob, isadapt, kwargs, p)
    u0 = get_concrete_u0(prob, isadapt, tspan[1], kwargs)
    u0_promote = promote_u0(u0, p, tspan[1])

    if isconcreteu0(prob, tspan[1], kwargs) &&
       typeof(u0_promote) === typeof(prob.u0) &&
       p === prob.p &&
       return prob
    else
        return remake(prob; u0 = u0_promote, p = p)
    end
end

struct LinearStateSpaceProblem{uType,uPriorType,tType,P,NP,AType,BType,CType,RType,ObsType,K,
                               SymsType} <: AbstractPerturbationProblem
    f::Nothing  # HACK: need something called "f" which supports "isinplace" for plotting
    A::AType
    B::BType
    C::CType
    observables_noise::RType
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
                                                      C = nothing, observables_noise = nothing,
                                                      observables = nothing, noise = nothing,
                                                      seed = UInt64(0), syms = nothing,
                                                      kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        # _observables = promote_vv(observables)
        # _noise = promote_vv(noise)
        _observables = observables
        _noise = noise

        # Require integer distances between time periods for now.  Later could check with dt != 1
        @assert round(_tspan[2] - _tspan[1]) - (_tspan[2] - _tspan[1]) ≈ 0.0

        return new{typeof(u0),typeof(u0_prior),typeof(_tspan),typeof(p),typeof(_noise),typeof(A),
                   typeof(B),typeof(C),typeof(observables_noise),typeof(_observables),
                   typeof(kwargs),typeof(syms)}(f, A, B, C, observables_noise, _observables, u0,
                                                u0_prior, _tspan, p, _noise, kwargs, seed, syms)
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
struct QuadraticStateSpaceProblem{uType,uPriorType,tType,P,NP,A0Type,A1Type,A2Type,BType,C0Type,
                                  C1Type,C2Type,RType,ObsType,K,SymsType} <:
       AbstractPerturbationProblem
    f::Nothing  # HACK: need something called "f" which supports "isinplace" for plotting
    A_0::A0Type
    A_1::A1Type
    A_2::A2Type
    B::BType
    C_0::C0Type
    C_1::C1Type
    C_2::C2Type
    observables_noise::RType
    observables::ObsType
    u0::uType
    u0_prior::uPriorType
    tspan::tType
    p::P
    noise::NP
    kwargs::K
    seed::UInt64
    syms::SymsType
    @add_kwonly function QuadraticStateSpaceProblem{iip}(A_0, A_1, A_2, B, u0, tspan,
                                                         p = NullParameters(); f = nothing,
                                                         u0_prior = u0, # default copies the u0 to u0_prior so it doesn't mess up get_concrete
                                                         C_0 = nothing, C_1 = nothing,
                                                         C_2 = nothing, observables_noise = nothing,
                                                         observables = nothing, noise = nothing,
                                                         seed = UInt64(0), syms = nothing,
                                                         kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        # _observables = promote_vv(observables)
        # _noise = promote_vv(noise)
        _observables = observables
        _noise = noise

        # Require integer distances between time periods for now.  Later could check with dt != 1
        @assert round(_tspan[2] - _tspan[1]) - (_tspan[2] - _tspan[1]) ≈ 0.0

        return new{typeof(u0),typeof(u0_prior),typeof(_tspan),typeof(p),typeof(_noise),typeof(A_0),
                   typeof(A_1),typeof(A_2),typeof(B),typeof(C_0),typeof(C_1),typeof(C_2),
                   typeof(observables_noise),typeof(_observables),typeof(kwargs),typeof(syms)}(f,
                                                                                               A_0,
                                                                                               A_1,
                                                                                               A_2,
                                                                                               B,
                                                                                               C_0,
                                                                                               C_1,
                                                                                               C_2,
                                                                                               observables_noise,
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
QuadraticStateSpaceProblem(args...; kwargs...) = QuadraticStateSpaceProblem{false}(args...;
                                                                                   kwargs...)
