"""
u_f(t+1) = A_1 u_f(t) .+ B * noise(t+1)
u(t+1) = A_0 + A_1 u(t) + quad(A_2, u_f(t)) .+ B noise(t+1)
z(t) = C_0 + C_1 u(t) + quad(C_2, u_f(t))
z_tilde(t) = z(t) + v(t+1)
"""
function quad(A::AbstractArray{<:Number,3}, x)
    return map(j -> dot(x, view(A, j, :, :), x), 1:size(A, 1))
end

function ChainRulesCore.rrule(::typeof(quad), A::AbstractArray{<:Number,3}, x)
    res = quad(A, x)
    function quad_pb(Δres)
        ΔA = similar(A)
        Δx = zeros(length(x))
        tmp = x * x'
        n = size(A, 1)
        for i in 1:n
            ΔA[i, :, :] .= tmp .* Δres[i]
            Δx += (A[i, :, :] + A[i, :, :]') * x .* Δres[i]
        end
        return NoTangent(), ΔA, Δx
    end
    return res, quad_pb
end

struct QuadraticStateSpaceProblem{
    isinplace, 
    A_0type<:AbstractArray,
    A_1type<:AbstractArray,
    A_2type<:AbstractArray, 
    Btype<:AbstractArray, 
    C_0type<:AbstractArray,
    C_1type<:AbstractArray,
    C_2type<:AbstractArray, 
    wtype, 
    Rtype, # Distributions only
    utype,
    ttype,
    otype
} <: AbstractStateSpaceProblem{isinplace}
    A_0::A_0type
    A_1::A_1type
    A_2::A_2type # Evolution matrix
    B::Btype # Noise matrix
    C_0::C_0type
    C_1::C_1type
    C_2::C_2type # Observation matrix
    noise::wtype # Latent noises
    obs_noise::Rtype # Observation noise / measurement error distribution
    u0::utype # Initial condition
    tspan::ttype # Timespan to use
    observables::otype # Observed data to use, if any
end

function QuadraticStateSpaceProblem(
    A_0::A_0type,
    A_1::A_1type,
    A_2::A_2type,
    B::Btype,
    C_0::C_0type,
    C_1::C_1type,
    C_2::C_2type,
    u0::utype,
    tspan::ttype;
    obs_noise = (h0 = C_1 * u0; MvNormal(zeros(eltype(h0), length(h0)), I)), # Assume the default measurement error is MvNormal with identity covariance
    observables = nothing,
    noise = nothing,
) where {
    A_0type<:AbstractArray,
    A_1type<:AbstractArray,
    A_2type<:AbstractArray, 
    Btype<:AbstractArray, 
    C_0type<:AbstractArray,
    C_1type<:AbstractArray,
    C_2type<:AbstractArray,
    utype,
    ttype,
}
    
    return QuadraticStateSpaceProblem{
        Val(false), 
        A_0type,
        A_1type,
        A_2type, 
        Btype, 
        C_0type,
        C_1type,
        C_2type, 
        typeof(noise), 
        typeof(obs_noise),
        utype,
        ttype,
        typeof(observables)
    }(
        A_0::A_0type,
        A_1::A_1type,
        A_2::A_2type, # Evolution matrix
        B::Btype, # Noise matrix
        C_0::C_0type,
        C_1::C_1type,
        C_2::C_2type, # Observation matrix
        noise, # Latent noise distribution
        obs_noise, # Observation noise matrix
        u0, # Initial condition
        tspan, # Timespan to use
        observables # Observed data to use, if any
    )
end

# Default is NoiseConditionalFilter
function CommonSolve.init(
    prob::QuadraticStateSpaceProblem, 
    args...; 
    kwargs...
)
    return StateSpaceCache(prob, NoiseConditionalFilter())
end

function CommonSolve.init(
    prob::QuadraticStateSpaceProblem,
    solver::SciMLBase.SciMLAlgorithm,
    args...;
    kwargs...
) 
    return StateSpaceCache(prob, solver)
end

function _solve!(
    prob::QuadraticStateSpaceProblem{isinplace, A_0type, A_1type, A_2type, Btype, C_0type, C_1type, C_2type, wtype, Rtype, utype, ttype, otype}, 
    ::NoiseConditionalFilter,
    args...;
    kwargs...
) where {isinplace, A_0type, A_1type, A_2type, Btype, C_0type, C_1type, C_2type, wtype, Rtype, utype, ttype, otype<:Nothing}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    A_0, A_1, A_2, B, C_0, C_1, C_2 = prob.A_0, prob.A_1, prob.A_2, prob.B, prob.C_0, prob.C_1, prob.C_2

    u_f = Zygote.Buffer(Vector{typeof(prob.u0)}(undef, T))
    u = Zygote.Buffer(Vector{typeof(prob.u0)}(undef, T))
    z0 = C_0 + C_1 * prob.u0 + quad(C_2, prob.u0)
    z = Zygote.Buffer(Vector{typeof(z0)}(undef, T))
    u[1] = prob.u0
    u_f[1] = prob.u0
    z[1] = z0
    
    for t in 2:T
        t_n = t - 1 + prob.tspan[1]
        u_f[t] = A_1 * u_f[t - 1] .+ B * prob.noise[t_n]
        u[t] = A_0 + A_1 * u[t - 1] + quad(A_2, u_f[t - 1]) .+ B * prob.noise[t_n]
        z[t] = C_0 + C_1 * u[t] + quad(C_2, u_f[t])
    end

    return StateSpaceSolution(copy(z), copy(u), prob.noise, nothing, nothing)
end

function _solve!(
    prob::QuadraticStateSpaceProblem{isinplace, A_0type, A_1type, A_2type, Btype, C_0type, C_1type, C_2type, wtype, Rtype, utype, ttype, otype}, 
    ::NoiseConditionalFilter,
    args...;
    kwargs...
) where {isinplace, A_0type, A_1type, A_2type, Btype, C_0type, C_1type, C_2type, wtype, Rtype, utype, ttype, otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    A_0, A_1, A_2, B, C_0, C_1, C_2 = prob.A_0, prob.A_1, prob.A_2, prob.B, prob.C_0, prob.C_1, prob.C_2

    u_f = Zygote.Buffer(Vector{typeof(prob.u0)}(undef, T))
    u = Zygote.Buffer(Vector{typeof(prob.u0)}(undef, T))
    z0 = C_0 + C_1 * prob.u0 + quad(C_2, prob.u0)
    z = Zygote.Buffer(Vector{typeof(z0)}(undef, T))
    u[1] = prob.u0
    u_f[1] = prob.u0
    z[1] = z0
    
    loglik = 0.0
    for t in 2:T
        t_n = t - 1 + prob.tspan[1]
        u_f[t] = A_1 * u_f[t - 1] .+ B * prob.noise[t_n]
        u[t] = A_0 + A_1 * u[t - 1] + quad(A_2, u_f[t - 1]) .+ B * prob.noise[t_n]
        z[t] = C_0 + C_1 * u[t] + quad(C_2, u_f[t])
        loglik += logpdf(prob.obs_noise, prob.observables[t_n] - z[t])
    end

    return StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
end
