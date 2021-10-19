"""
    StateSpaceProblem{isinplace, dType, uType,tType,noiseType,oType, lType} <: AbstractStateSpaceProblem{isinplace}

A state space problem contains all the information necessary to simulate a state space
process.
"""
struct StateSpaceProblem{isinplace, dType, uType,tType,noiseType,oType} <: AbstractStateSpaceProblem{isinplace}
    f::Function # Evolution
    g::Function # Noise function
    h::Function # Observation function
    latent_noise::dType # Noise distribution
    u0::uType # Initial condition
    tspan::tType # Timespan to use
    noise::noiseType # The observation noise to use. No noise → perfect obs
    observables::oType # Data to use, if any
end

noisetype(prob::StateSpaceProblem{isinplace, dType}) where {isinplace, dType} = dType
hasobservables(prob::StateSpaceProblem{a,b,c,d,e,f}) where {a,b,c,d,e,f} = ~(f <: Nothing)

"""
    StateSpaceProblem(f, g, h, D, u0, tspan, noise, observables, likelihood=((x...) -> 0.0))
"""
function StateSpaceProblem(f, g, h, latent_noise, u0, tspan, noise=nothing, observables=nothing)
    return StateSpaceProblem{
        Val(false), # Default to out-of-place
        typeof(latent_noise),
        typeof(u0),
        typeof(tspan),
        typeof(noise),
        typeof(observables)
    }(f, g, h, latent_noise, u0, SciMLBase.promote_tspan(tspan), noise, observables)
end

"""
    index_or_nothing(x::AbstractArray, i::Int)
    index_or_nothing(x::Nothing, i::Int)

Extracts index `i` from `x`, if `x` is a subtype of `AbstractArray`. Otherwise,
this function returns `nothing`.
"""
index_or_nothing(x::AbstractArray, i::Int) = x[i]
index_or_nothing(x::Nothing, i::Int) = nothing

latent_noise(prob::StateSpaceProblem{inplace, <:AbstractArray}, t) where {inplace} = prob.latent_noise[t]
latent_noise(prob::StateSpaceProblem{inplace, <:Distribution}, t) where {inplace} = rand(prob.latent_noise)
latent_noise(prob::StateSpaceProblem{inplace, <:AbstractMatrix}, t) where {inplace} = prob.latent_noise * randn(size(prob.u0))
latent_noise(prob::StateSpaceProblem, t) = randn(size(prob.u0))

# No observations, return zero likelihood
function likelihood(
    prob::StateSpaceProblem,
    alg::ConditionalGaussian,
    z_t,
    obs_t::Nothing
)
    return 0
end

# Have observables, dispatch on inner likelihood
function likelihood(
    prob::StateSpaceProblem{isinplace, dType, uType,tType,noiseType,oType},
    alg::ConditionalGaussian,
    z_t,
    obs_t
) where {isinplace, dType, uType,tType,noiseType,oType}
    return _likelihood(prob, alg, z_t, obs_t, noiseType)
end

function _likelihood(
    prob::StateSpaceProblem,
    alg::ConditionalGaussian,
    z_t,
    obs_t,
    noiseType::Type{Nothing}
)
    # Unclear whether this should be default behavior
    return 0
end

function _likelihood(
    prob::StateSpaceProblem,
    alg::ConditionalGaussian,
    z_t,
    obs_t::Number,
    noiseType::Type{<:UnivariateDistribution}
)
    # Unclear whether this should be default behavior
    return logpdf(prob.noise, obs_t - only(z_t))
end

function _likelihood(
    prob::StateSpaceProblem,
    alg::ConditionalGaussian,
    z_t,
    obs_t::Number,
    noiseType::Type{<:MultivariateDistribution}
)
    # Unclear whether this should be default behavior
    return logpdf(prob.noise, [obs_t] .- z_t)
end

function _likelihood(
    prob::StateSpaceProblem,
    alg::ConditionalGaussian,
    z_t,
    obs_t,
    noiseType
)
    # If given a noise distribution, provide the log pdf w.r.t. the
    # distribution.
    throw(ArgumentError("No implementation for prob = StateSpaceProblem, alg = ConditionalGaussian, and noiseType = $noiseType"))
end

"""
    _solve(prob::StateSpaceProblem, params)

Returns a `StateSpaceSolution` using the provided parameters `params`.
Calculates the likelihood if the likelihood function is available.
"""
function _solve(
    prob::StateSpaceProblem{isinplace, dType, uType,tType,noiseType,oType}, 
    alg::ConditionalGaussian, 
    params
) where {isinplace, dType, uType,tType,noiseType,oType <: Nothing} # No observables
    t0, T = prob.tspan

    # Initial noise
    z0 = prob.h(prob.u0, params, 1)

    # Preallocate arrays
    u = Vector{typeof(prob.u0)}(undef, T+1)
    z = Vector{typeof(z0)}(undef, T+1)
    n = noisetype(prob) <: Vector ? 
        Vector{eltype(prob.noise)}(undef, T+1) :
        Vector{typeof(prob.u0)}(undef, T+1) # Store sampled transition noise

    u[1] = prob.u0
    z[1] = z0

    for t in 2:T+1
        n[t-1] = latent_noise(prob, t-1)
        u[t] = prob.f(u[t - 1], params, t-1) + prob.g(u[t - 1], params, t-1) .* n[t-1] # Should this be n[t] or n[t-1]?
        z[t] = prob.h(u[t], params, t)
    end

    return StateSpaceSolution(copy(z),copy(u),prob.noise,nothing,nothing,n)
end
