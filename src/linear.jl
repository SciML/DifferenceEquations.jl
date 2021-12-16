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
    R::Rtype # Observation noise matrix
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
    R = diagm(ones(size(C,1))),
    observables = nothing,
    noise = StandardGaussian(size(B, 1)),
) where {
    Atype<:AbstractArray, 
    Btype<:AbstractArray, 
    Ctype<:AbstractArray, 
    utype,
    ttype,
}
    v = diagm(ones(size(C,1)))

    if R isa Vector
        @assert length(R) == 1
        R = hcat(R) # Convert to matrix
    end
    
    return LinearStateSpaceProblem{
        Val(false), 
        Atype, 
        Btype, 
        Ctype, 
        typeof(noise), 
        typeof(R),
        utype,
        ttype,
        typeof(observables)
    }(
        A, # Evolution matrix
        B, # Noise matrix
        C, # Observation matrix
        noise, # Latent noise distribution
        v, # Observation noise matrix
        u0, # Initial condition
        tspan, # Timespan to use
        observables # Observed data to use, if any
    )
end


CommonSolve.init(prob::LinearStateSpaceProblem, args...; kwargs...) = StateSpaceCache(prob, ConditionalGaussian())
CommonSolve.init(prob::LinearStateSpaceProblem, solver::SciMLBase.SciMLAlgorithm, args...; kwargs...) = StateSpaceCache(prob, solver)

function _solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype}, 
    ::ConditionalGaussian,
    args...; 
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype<:AbstractMatrix, utype, ttype, otype<:Nothing}
    # Preallocate values
    T = prob.tspan[2]
    A,B,C,R = prob.A, prob.B, prob.C, prob.R
    L = size(C, 1) # Rows of observations

    u = Vector{utype}(undef, T+1) # Latent states
    n = Vector{utype}(undef, T+1) # Latent noise
    z = Vector{utype}(undef, T+1) # Observables generated

    u[1] = prob.u0
    z[1] = C * u[1] + R*randn(L)
    n[1] = noise(prob.noise, 1)  # This noise term is never used

    # Simulate it, homie
    for t in 2:T+1
        n[t] = noise(prob.noise, t)
        u[t] = A * u[t-1] + B * n[t] # Call latent_noise here
        z[t] = C*u[t] + R*randn(L)
    end

    return StateSpaceSolution(z, u, n, nothing)
end

function CommonSolve.solve!(
    prob::LinearStateSpaceProblem{isinplace, Atype, Btype, Ctype, wtype, Rtype, utype, ttype, otype}, 
    ::ConditionalGaussian,
    args...; 
    kwargs...
) where {isinplace, Atype, Btype, Ctype, wtype, Rtype<:AbstractMatrix, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2]
    A,B,C,R = prob.A, prob.B, prob.C, prob.R
    L = size(C, 1) # Rows of observations

    u = Vector{utype}(undef, T+1) # Latent states
    n = Vector{utype}(undef, T+1) # Latent noise
    z = Vector{utype}(undef, T+1) # Observables generated

    u[1] = prob.u0
    z[1] = C * u[1] + R*randn(L)
    n[1] = noise(prob.noise, 1) # This noise term is never used
    loglik = 0.0

    # Simulate it, homie
    for t in 2:T
        n[t] = noise(prob.noise, t)
        u[t] = A * u[t-1] + B * n[t] # Call latent_noise here
        z[t] = C*u[t] + R*randn(L)
        err = z[t] - prob.observables[t]
        loglik += logpdf(MvNormal(R), err)
    end

    return StateSpaceSolution(z, u, n, loglik)
end