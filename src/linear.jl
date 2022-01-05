"""
u(t+1) = A u(t) + B w(t+1)
z(t) = C u(t)
z_tilde{t} = z(t) + v(t+1)
"""
struct LinearStateSpaceProblem{
    isinplace, 
    Atype<:AbstractArray, 
    Btype<:AbstractArray, 
    Ctype<:AbstractArray, 
    wtype, 
    Rtype, # Distributions only
    utype,
    ttype,
    otype
} <: AbstractStateSpaceProblem{isinplace}
    A::Atype # Evolution matrix
    B::Btype # Noise matrix
    C::Ctype # Observation matrix
    noise::wtype # Latent noises
    obs_noise::Rtype # Observation noise / measurement error distribution
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
    obs_noise = (h0 = C * u0; MvNormal(zeros(eltype(h0), length(h0)), I)), # Assume the default measurement error is MvNormal with identity covariance
    observables = nothing,
    noise = nothing,
) where {
    Atype<:AbstractArray, 
    Btype<:AbstractArray, 
    Ctype<:AbstractArray, 
    utype,
    ttype,
}
    
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
    kwargs...
)
    return StateSpaceCache(prob, NoiseConditionalFilter())
end

function CommonSolve.init(
    prob::LinearStateSpaceProblem,
    solver::SciMLBase.SciMLAlgorithm,
    args...;
    kwargs...
) 
    return StateSpaceCache(prob, solver)
end

function _solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype}, 
    ::NoiseConditionalFilter,
    args...;
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype<:Nothing}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    A, B, C = prob.A, prob.B, prob.C

    u = Zygote.Buffer(Vector{utype}(undef, T)) # Latent states
    n1 = prob.noise[1]
    n = Zygote.Buffer(Vector{typeof(n1)}(undef, T)) # Latent noise
    z1 = C * prob.u0
    z = Zygote.Buffer(Vector{typeof(z1)}(undef, T)) # Observables generated

    # Initialize
    u[1] = prob.u0
    z[1] = C * u[1]

    for t in 2:T
        t_n = t - 1 + prob.tspan[1]
        n[t] = prob.noise[t_n]
        u[t] = A * u[t - 1] + B * n[t]
        z[t] = C * u[t]
    end

    return StateSpaceSolution(copy(z), copy(u), copy(n), nothing, nothing)
end

function _solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype}, 
    ::NoiseConditionalFilter,
    args...;
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    A, B, C = prob.A, prob.B, prob.C

    u = Zygote.Buffer(Vector{utype}(undef, T)) # Latent states
    n1 = prob.noise[1]
    n = Zygote.Buffer(Vector{typeof(n1)}(undef, T)) # Latent noise
    z1 = C * prob.u0
    z = Zygote.Buffer(Vector{typeof(z1)}(undef, T)) # Observables generated

    # Initialize
    u[1] = prob.u0
    z[1] = C * u[1]

    loglik = 0.0
    for t in 2:T
        t_n = t - 1 + prob.tspan[1]
        n[t] = prob.noise[t_n]
        u[t] = A * u[t - 1] + B * n[t]
        z[t] = C * u[t]
        loglik += logpdf(prob.obs_noise, prob.observables[t_n] - z[t])
    end

    return StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
end
