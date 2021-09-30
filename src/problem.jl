"""
    StateSpaceProblem{isinplace, dType, uType,tType,noiseType,oType, lType} <: AbstractStateSpaceProblem{isinplace}

A state space problem contains all the information necessary to simulate a state space
process.
"""
struct StateSpaceProblem{isinplace, dType, uType,tType,noiseType,oType, lType} <: AbstractStateSpaceProblem{isinplace}
    f::Function # Evolution
    g::Function # Noise function
    h::Function # Observation function
    D::dType # Noise distribution
    u0::uType # Initial condition
    tspan::tType # Timespan to use
    noise::noiseType # The noise to use
    observables::oType # Data to use, if any
    likelihood::lType # Likelihood function, if any
end

"""
    StateSpaceProblem(f, g, h, D, u0, tspan, noise, observables, likelihood=((x...) -> 0.0))
"""
function StateSpaceProblem(f, g, h, D, u0, tspan, noise, observables, likelihood=((x...) -> 0.0))
    return StateSpaceProblem{
        Val(false), # Default to out-of-place
        typeof(D),
        typeof(u0),
        typeof(tspan),
        typeof(noise),
        typeof(observables),
        typeof(likelihood)
    }(f, g, h, D, u0, SciMLBase.promote_tspan(tspan), noise, observables, likelihood)
end

"""
    index_or_nothing(x::AbstractArray, i::Int)
    index_or_nothing(x::Nothing, i::Int)

Extracts index `i` from `x`, if `x` is a subtype of `AbstractArray`. Otherwise,
this function returns `nothing`.
"""
index_or_nothing(x::AbstractArray, i::Int) = x[i]
index_or_nothing(x::Nothing, i::Int) = nothing

"""
    transition(prob::StateSpaceProblem, u, params, t)

Calculates the next latent state, given by `f(u[t-1], params, t-1) + g(u[t-1], params, t-1)`.
"""
function transition(prob::StateSpaceProblem, u, params, t)
    return prob.f(u[t - 1], params, t-1) .+ prob.g(u[t - 1], params, t-1, index_or_nothing(prob.noise, t-1))
end

"""
    _solve(prob::StateSpaceProblem, params)

Returns a `StateSpaceSolution` using the provided parameters `params`.
Calculates the likelihood if the likelihood function is available.
"""
function _solve(prob::StateSpaceProblem, params)
    t0, T = prob.tspan

    # Initial noise
    z0 = prob.h(prob.u0, params, 1, nothing)

    # Preallocate arrays
    u = Vector{typeof(prob.u0)}(undef, T+1) # Should be square, not vectored?
    z = Vector{typeof(z0)}(undef, T+1) # Should be square, not vectored?

    u[1] = prob.u0
    z[1] = z0

    # Accumulate likelihood
    loglik = 0.0

    for t in 2:T+1
        u[t] = transition(prob, u, params, t)
        z[t] = prob.h(u[t], params, t, prob.noise)
        loglik += prob.likelihood(z[t], params, t, prob.observables)
    end

    return StateSpaceSolution(copy(z),copy(u),prob.noise,nothing,loglik)
end