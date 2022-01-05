struct StateSpaceProblem{
    isinplace, 
    ftype, # TODO: Replace with LinearOperator 
    gtype, # TODO: Replace with LinearOperator 
    htype, # TODO: Replace with LinearOperator 
    wtype, 
    vtype, # We assume for vtype <: Distribution
    utype,
    ttype,
    otype,
    ptype
} <: AbstractStateSpaceProblem{isinplace}
    f::ftype # Evolution function
    g::gtype # Noise function
    h::htype # Observation function
    noise::wtype # Latent noises
    obs_noise::vtype # Observation noise / measurement error distribution
    u0::utype # Initial condition
    tspan::ttype # Timespan to use
    observables::otype # Observed data to use, if any
    params::ptype # Parameters, if any
end

function StateSpaceProblem(
    f::ftype, 
    g::gtype, 
    h::htype,
    u0::utype,
    tspan::ttype,
    params = nothing;
    obs_noise = (h0 = h(u0, params, tspan[1]); MvNormal(zeros(eltype(h0), length(h0)), I)), # Assume the default measurement error is MvNormal with identity covariance
    observables = nothing,
    noise = nothing,
) where {
    ftype, 
    gtype, 
    htype, 
    utype,
    ttype,
}

    return StateSpaceProblem{
        Val(false), 
        ftype, 
        gtype, 
        htype, 
        typeof(noise), 
        typeof(obs_noise),
        utype,
        ttype,
        typeof(observables),
        typeof(params)
    }(
        f, # Evolution function
        g, # Noise function
        h, # Observation function
        noise, # Latent noises
        obs_noise, # Observation noise distribution
        u0, # Initial condition
        tspan, # Timespan to use
        observables, # Observed data to use, if any
        params
    )
end

# Default is NoiseConditionalFilter
function CommonSolve.init(
    prob::StateSpaceProblem, 
    args...; 
    kwargs...
)
    return StateSpaceCache(prob, NoiseConditionalFilter())
end

function CommonSolve.init(
    prob::StateSpaceProblem,
    solver::SciMLBase.SciMLAlgorithm,
    args...;
    kwargs...
) 
    return StateSpaceCache(prob, solver)
end

function _solve!(
    prob::StateSpaceProblem{isinplace, ftype, gtype, htype, wtype, vtype, utype, ttype, otype},
    solver::NoiseConditionalFilter,
    args...;
    kwargs...
) where {isinplace, ftype, gtype, htype, wtype, vtype, utype, ttype, otype<:Nothing}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1

    u = Zygote.Buffer(Vector{utype}(undef, T)) # Latent states
    n1 = prob.noise[1] # This is only to grab the type of the noises. We won't use it in the simulation
    n = Zygote.Buffer(Vector{typeof(n1)}(undef, T)) # Latent noise
    z1 = prob.h(prob.u0, prob.params, prob.tspan[1]) # Grab the type of the observations of the initial latent states
    z = Zygote.Buffer(Vector{typeof(z1)}(undef, T)) # Observables generated

    # Initialize
    u[1] = prob.u0
    z[1] = z1

    for t in 2:T
        t_n = t - 1 + prob.tspan[1]
        n[t] = prob.noise[t_n]
        u[t] = prob.f(u[t - 1], prob.params, t_n - 1) .+ prob.g(u[t - 1], prob.params, t_n - 1) * n[t]
        z[t] = prob.h(u[t], prob.params, t_n)
    end

    return StateSpaceSolution(copy(z), copy(u), copy(n), nothing, nothing)
end

function _solve!(
    prob::StateSpaceProblem{isinplace, ftype, gtype, htype, wtype, vtype, utype, ttype, otype}, 
    solver::NoiseConditionalFilter,
    args...;
    kwargs...
) where {isinplace, ftype, gtype, htype, wtype, vtype, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1

    u = Zygote.Buffer(Vector{utype}(undef, T)) # Latent states
    n1 = prob.noise[1] # This is only to grab the type of the noises. We won't use it in the simulation
    n = Zygote.Buffer(Vector{typeof(n1)}(undef, T)) # Latent noise
    z1 = prob.h(prob.u0, prob.params, prob.tspan[1]) # Grab the type of the observations of the initial latent states
    z = Zygote.Buffer(Vector{typeof(z1)}(undef, T)) # Observables generated

    # Initialize
    u[1] = prob.u0
    z[1] = z1

    loglik = 0.0
    for t in 2:T
        t_n = t - 1 + prob.tspan[1]
        n[t] = prob.noise[t_n]
        u[t] = prob.f(u[t - 1], prob.params, t_n - 1) .+ prob.g(u[t - 1], prob.params, t_n - 1) * n[t]
        z[t] = prob.h(u[t], prob.params, t_n)
        # Likelihood accumulation when data observations are provided
        loglik += logpdf(prob.obs_noise, prob.observables[t_n] - z[t])
    end

    return StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
end
