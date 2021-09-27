struct StateSpaceProblem{isinplace, dType, uType,tType,noiseType,oType} <: AbstractStateSpaceProblem{isinplace}
    f::Function # Evolution
    g::Function # Noise function
    h::Function # Observation function
    D::dType # Noise distribution
    u0::uType # Initial condition
    tspan::tType # Timespan to use
    noise::noiseType # The noise to use
    observables::oType # Data to use, if any
end

function StateSpaceProblem(f, g, h, D, u0, tspan, noise, observables)
    return StateSpaceProblem(f, g, h, D, u0, SciMLBase.promote_tspan(tspan), noise, observables)
end

# Likelhood remains 0 if no obserables.  Need to verify no overhead
maybe_logpdf(observables::Nothing, D, z, i) = 0.0
maybe_logpdf(observables, D, z, i) = logpdf(D, observables[i] .- z)

function _solve(prob::StateSpaceProblem, params)
    t0, T = prob.tspan

    # Initial noise
    z0 = h(prob.u0, params, 0)

    # Preallocate arrays
    u = zeros(typeof(prob.u0), T+1) # Should be square, not vectored?
    z = zeros(typeof(z0), T+1) # Should be square, not vectored?

    u[1] = prob.u0
    z[1] = z0

    # Accumulate likelihood
    loglik = 0.0

    for t in 2:T+1
        u[t] = prob.f(u[t - 1], p, t-1) .+ prob.g(u[t - 1], p, t-1) * prob.noise[t-1]
        z[t] = prob.h(u[t], p, t)
        loglik += maybe_logpdf(prob.observables, prob.D, z[t], t-1)  # z_0 doesn't enter likelihood        
    end

    return StateSpaceSolution(copy(z),copy(u),noise,nothing,loglik)
end