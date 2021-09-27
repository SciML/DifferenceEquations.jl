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
    return StateSpaceProblem{
        Val(false),
        typeof(D),
        typeof(u0),
        typeof(tspan),
        typeof(noise),
        typeof(observables)
    }(f, g, h, D, u0, SciMLBase.promote_tspan(tspan), noise, observables)
end

# Likelhood remains 0 if no obserables.  Need to verify no overhead
maybe_logpdf(observables::Nothing, D, z, i) = 0.0
function maybe_logpdf(observables, D, z, i)
    # println(logpdf(D, observables[i] .- z))
    return logpdf(D, observables[i] .- z)
end

# function transition(prob::StateSpaceProblem{A,B,C,D,Nothing,F}, u, params, t) where {A,B,C,D,F}
#     return prob.f(u[t - 1], params, t-1)
# end

index_or_nothing(x::AbstractArray, i::Int) = x[i]
index_or_nothing(x::Nothing, i::Int) = nothing
function transition(prob::StateSpaceProblem, u, params, t)
    return prob.f(u[t - 1], params, t-1) .+ prob.g(u[t - 1], params, t-1, index_or_nothing(prob.noise, t-1))
end

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
        # u[t] = prob.f(u[t - 1], params, t-1) .+ prob.g(u[t - 1], params, t-1) * prob.noise[t-1]
        z[t] = prob.h(u[t], params, t, prob.noise)
        loglik += maybe_logpdf(prob.observables, prob.D, z[t], t-1)  # z_0 doesn't enter likelihood        
    end

    return StateSpaceSolution(copy(z),copy(u),prob.noise,nothing,loglik)
end