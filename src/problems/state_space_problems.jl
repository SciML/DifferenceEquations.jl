abstract type AbstractStateSpaceProblem <: DiffEqBase.DEProblem end
abstract type AbstractPerturbationProblem <: AbstractStateSpaceProblem end

using DiffEqBase: get_concrete_u0, promote_u0, isconcreteu0

# TODO: Can add in more checks on the algorithm choice
DiffEqBase.check_prob_alg_pairing(prob::AbstractStateSpaceProblem, alg) = nothing

# Perturbation problesm don't have f, g
# In discrete time, tspan should not have a sensitivity so the concretization is less obvious
function DiffEqBase.get_concrete_problem(prob::AbstractPerturbationProblem, isadapt;
        kwargs...)
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

SciMLBase.isinplace(prob::AbstractPerturbationProblem) = false  # necessary for the get_concrete_u0 overloads

# the {iip} isn't relevant here at this point, but if we remove it then there are failures in the "remake" call above
# when using the Ensemble unit tests
struct LinearStateSpaceProblem{
    uType, uPriorMeanType, uPriorVarType, tType, P, NP, F, AType,
    BType, CType,
    RType, ObsType, K} <:
       AbstractPerturbationProblem
    f::F # HACK: used only for standard interfaces/syms/etc., not used in calculations
    A::AType
    B::BType
    C::CType
    observables_noise::RType
    observables::ObsType
    u0::uType
    u0_prior_mean::uPriorMeanType
    u0_prior_var::uPriorVarType
    tspan::tType
    p::P
    noise::NP
    kwargs::K
    @add_kwonly function LinearStateSpaceProblem{iip}(
            A, B, u0, tspan, p = NullParameters();
            u0_prior_mean = nothing,
            u0_prior_var = nothing, C = nothing,
            observables_noise = nothing,
            observables = nothing,
            noise = nothing,
            syms = nothing,
            f = ODEFunction{false}(((u, p, t) -> error("not implemented"));
                syms = syms),
            kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        # _observables = promote_vv(observables)
        _observables = observables

        # Require integer distances between time periods for now.  Later could check with dt != 1
        @assert round(_tspan[2] - _tspan[1]) - (_tspan[2] - _tspan[1]) ≈ 0.0

        return new{typeof(u0), typeof(u0_prior_mean), typeof(u0_prior_var), typeof(_tspan),
            typeof(p),
            typeof(noise), typeof(f),
            typeof(A), typeof(B), typeof(C), typeof(observables_noise),
            typeof(_observables),
            typeof(kwargs)}(f, A, B, C, observables_noise, _observables, u0,
            u0_prior_mean,
            u0_prior_var,
            _tspan, p, noise, kwargs)
    end
end
# just forwards to a iip = false case
function LinearStateSpaceProblem(args...; kwargs...)
    LinearStateSpaceProblem{false}(args...; kwargs...)
end

# """
# u_f(t+1) = A_1 u_f(t) .+ B * noise(t+1)
# u(t+1) = A_0 + A_1 u(t) + quad(A_2, u_f(t)) .+ B noise(t+1)
# z(t) = C_0 + C_1 u(t) + quad(C_2, u_f(t))
# z_tilde(t) = z(t) + v(t+1)
# """
struct QuadraticStateSpaceProblem{uType, uPriorMeanType, uPriorVarType, tType, P, NP, F,
    A0Type, A1Type,
    A2Type, BType, C0Type,
    C1Type, C2Type, RType, ObsType, K} <:
       AbstractPerturbationProblem
    f::F # HACK: used only for standard interfaces/syms/etc., not used in calculations
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
    u0_prior_mean::uPriorMeanType
    u0_prior_var::uPriorVarType
    tspan::tType
    p::P
    noise::NP
    kwargs::K
    @add_kwonly function QuadraticStateSpaceProblem{iip}(A_0, A_1, A_2, B, u0, tspan,
            p = NullParameters();
            u0_prior_mean = nothing,
            u0_prior_var = nothing,
            C_0 = nothing, C_1 = nothing,
            C_2 = nothing,
            observables_noise = nothing,
            observables = nothing,
            noise = nothing,
            syms = nothing,
            f = ODEFunction{false}(((u, p, t) -> error("not implemented"));
                syms = syms),
            kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        # _observables = promote_vv(observables)
        _observables = observables

        # Require integer distances between time periods for now.  Later could check with dt != 1
        @assert round(_tspan[2] - _tspan[1]) - (_tspan[2] - _tspan[1]) ≈ 0.0

        return new{typeof(u0), typeof(u0_prior_mean), typeof(u0_prior_var), typeof(_tspan),
            typeof(p),
            typeof(noise), typeof(f),
            typeof(A_0), typeof(A_1), typeof(A_2), typeof(B), typeof(C_0),
            typeof(C_1),
            typeof(C_2), typeof(observables_noise), typeof(_observables),
            typeof(kwargs)}(f,
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
            u0_prior_mean,
            u0_prior_var,
            _tspan,
            p,
            noise,
            kwargs)
    end
end
# just forwards to a iip = false case
function QuadraticStateSpaceProblem(args...; kwargs...)
    QuadraticStateSpaceProblem{false}(args...;
        kwargs...)
end
