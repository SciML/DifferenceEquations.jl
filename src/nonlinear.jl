"""

u(t+1) = A u(t) + B w(t+1)
z(t) = C u(t) + v(t+1)
"""
struct StateSpaceProblem{
    isinplace, 
    ftype<:Function, 
    gtype<:Function, 
    htype<:Function, 
    wtype, 
    vtype, # TODO: Add support methods for vtype <: Distribution
    utype,
    ttype,
    otype
} <: AbstractStateSpaceProblem{isinplace}
    f::ftype # Evolution function
    g::gtype # Noise function
    h::htype # Observation function
    noise::wtype # Latent noise distribution
    obs_noise::vtype # Observation noise matrix
    u0::utype # Initial condition
    tspan::ttype # Timespan to use
    observables::otype # Observed data to use, if any
end

function StateSpaceProblem(
    f::Function, 
    g::Function, 
    h::Function,
    u0::utype,
    tspan::ttype;
    obs_noise = StandardGaussian(length(u0)),
    observables = nothing,
    noise = StandardGaussian(length(u0)),
) where {
    ftype<:Function, 
    gtype<:Function, 
    htype<:Function, 
    utype,
    ttype,
}
    v = diagm(ones(size(C,1)))

    if obs_noise isa Vector
        @assert length(obs_noise) == 1
        obs_noise = hcat(obs_noise) # Convert to matrix
    end

    return StateSpaceProblem{
        Val(false), 
        ftype, 
        gtype, 
        htype, 
        typeof(noise), 
        typeof(obs_noise),
        utype,
        ttype,
        typeof(observables)
    }(
        f, # Evolution function
        g, # Noise function
        h, # Observation function
        noise, # Latent noise matrix/function/distribution
        obs_noise, # Observation noise matrix/function/distribution
        u0, # Initial condition
        tspan, # Timespan to use
        observables # Observed data to use, if any
    )
end


CommonSolve.init(::StateSpaceProblem, args...; kwargs...) = LinearGaussian()
solve!(::LinearGaussian) = StateSpaceSolution(missing,missing,missing,missing) # TODO: Need more sensible defaults

function SciMLBase.solve(
    prob::StateSpaceProblem{isinplace, ftype, gtype, htype, wtype, vtype, utype, ttype, otype}, 
    ::LinearGaussian, 
    args...; 
    kwargs...
) where {isinplace, ftype, gtype, htype, wtype, vtype, utype, ttype, otype<:Nothing}
    # Preallocate values
    T = prob.tspan[2] # Don't like this at all, need to ensure tspan promotion occurs in constructor

    u = Vector{utype}(undef, T+1) # Latent states
    n = Vector{utype}(undef, T+1) # Latent noise
    z = Vector{utype}(undef, T+1) # Observables generated

    u[1] = prob.u0
    z[1] = prob.h(u[1], p, 1) + noise(prob.obs_noise, 1)
    n[1] = noise(prob.noise, 1)  # XXX: This noise term is never used?

    # Simulate it, homie
    for t in 2:T+1
        n[t] = noise(prob.noise, t)
        u[t] = prob.f(u[t-1], p, t-1) + prob.g(u[t-1], p, t-1) * n[t]
        z[t] = prob.h(u[t], p, t) + noise(prob.obs_noise, 1)
    end

    return StateSpaceSolution(z, u, n, nothing)
end

function SciMLBase.solve(
    prob::StateSpaceProblem{isinplace, ftype, gtype, htype, wtype, vtype, utype, ttype, otype}, 
    ::LinearGaussian, 
    args...; 
    kwargs...
) where {isinplace, ftype, gtype, htype, wtype, vtype, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2] # Don't like this at all, need to ensure tspan promotion occurs in constructor

    u = Vector{utype}(undef, T+1) # Latent states
    n = Vector{utype}(undef, T+1) # Latent noise
    z = Vector{utype}(undef, T+1) # Observables generated

    u[1] = prob.u0
    z[1] = prob.h(u[1], p, 1) + noise(prob.obs_noise, 1)
    n[1] = noise(prob.noise, 1)  # XXX: This noise term is never used?

    # Simulate it, homie
    for t in 2:T+1
        n[t] = noise(prob.noise, t)
        u[t] = prob.f(u[t-1], p, t-1) + prob.g(u[t-1], p, t-1) * n[t]
        z[t] = prob.h(u[t], p, t) + noise(prob.obs_noise, 1)
        err = z[t] - prob.observables[t]
        loglik += logpdf(MvNormal(R), err)
    end

    return StateSpaceSolution(z, u, n, loglik)
end