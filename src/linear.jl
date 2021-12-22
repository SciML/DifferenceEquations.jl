"""

u(t+1) = A u(t) + B w(t+1)
z(t) = C u(t) + v(t+1)
"""
struct LinearStateSpaceProblem{
    isinplace, 
    Atype<:AbstractArray, 
    Btype<:AbstractArray, 
    Ctype<:AbstractArray, 
    wtype, 
    Rtype, # Should be expanded later on to include distributions
    utype,
    ttype,
    otype
} <: AbstractStateSpaceProblem{isinplace}
    A::Atype # Evolution matrix
    B::Btype # Noise matrix
    C::Ctype # Observation matrix
    noise::wtype # Latent noise distribution
    obs_noise::Rtype # Observation noise matrix
    u0::utype # Initial condition
    tspan::ttype # Timespan to use
    observables::otype # Observed data to use, if any
end

function LinearStateSpaceProblem(
    A::Atype, 
    B::Btype, 
    C::Ctype,
    u0::utype,
    tspan::ttype;
    obs_noise = diagm(ones(size(C,1))),
    observables = nothing,
    noise = StandardGaussian(size(B, 1)),
) where {
    Atype<:AbstractArray, 
    Btype<:AbstractArray, 
    Ctype<:AbstractArray, 
    utype,
    ttype,
}
    if obs_noise isa Vector
        @assert length(obs_noise) == 1
        obs_noise = hcat(obs_noise) # Convert to matrix
    end
    
    return LinearStateSpaceProblem{
        Val(false), 
        Atype, 
        Btype, 
        Ctype, 
        typeof(noise), 
        typeof(obs_noise),
        utype,
        ttype,
        typeof(observables)
    }(
        A, # Evolution matrix
        B, # Noise matrix
        C, # Observation matrix
        noise, # Latent noise distribution
        obs_noise, # Observation noise matrix
        u0, # Initial condition
        tspan, # Timespan to use
        observables # Observed data to use, if any
    )
end

# Default is NoiseConditionalFilter
function CommonSolve.init(
    prob::LinearStateSpaceProblem, 
    args...; 
    vectype=identity, 
    kwargs...
)
    return StateSpaceCache(prob, NoiseConditionalFilter(), vectype)
end

function CommonSolve.init(
    prob::LinearStateSpaceProblem,
    solver::SciMLBase.SciMLAlgorithm,
    args...;
    vectype=identity,
    kwargs...
) 
    return StateSpaceCache(prob, solver, vectype)
end

function _solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype}, 
    ::NoiseConditionalFilter,
    args...;
    vectype=identity,
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype<:AbstractMatrix, utype, ttype, otype<:Nothing}
    # Preallocate values
    T = prob.tspan[2]
    A,B,C = prob.A, prob.B, prob.C
    L = size(C, 1) # Rows of observations

    u = vectype(Vector{utype}(undef, T)) # Latent states
    n = vectype(Vector{utype}(undef, T)) # Latent noise
    z = vectype(Vector{utype}(undef, T)) # Observables generated

    u[1] = prob.u0
    z[1] = C * u[1] + noise(prob.obs_noise, 1)
    n[1] = noise(prob.noise, 1)  # This noise term is never used

    # Simulate it, homie
    for t in 2:T
        n[t] = noise(prob.noise, t)
        u[t] = A * u[t-1] + B * n[t] # Call latent_noise here
        z[t] = C*u[t] + noise(prob.obs_noise, t)
    end

    return StateSpaceSolution(copy(z), copy(u), copy(n), nothing)
end

function _solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype}, 
    ::NoiseConditionalFilter,
    args...; 
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype<:AbstractMatrix, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2]
    A,B,C = prob.A, prob.B, prob.C
    L = size(C, 1) # Rows of observations

    u = vectype(Vector{utype}(undef, T)) # Latent states
    n = vectype(Vector{utype}(undef, T)) # Latent noise
    z = vectype(Vector{utype}(undef, T)) # Observables generated

    u[1] = prob.u0
    z[1] = C * u[1] + noise(prob.obs_noise, 1)
    n[1] = noise(prob.noise, 1) # This noise term is never used
    loglik = 0.0

    # Simulate it, homie
    for t in 2:T
        n[t] = noise(prob.noise, t)
        u[t] = A * u[t-1] + B * n[t] # Call latent_noise here
        z[t] = C*u[t] + noise(prob.obs_noise, t)
        err = z[t] - prob.observables[t]
        loglik += logpdf(MvNormal(R), err)
    end

    return StateSpaceSolution(copy(z), copy(u), copy(n), loglik)
end
