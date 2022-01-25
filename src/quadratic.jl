"""
u_f(t+1) = A_1 u_f(t) .+ B * noise(t+1)
u(t+1) = A_0 + A_1 u(t) + quad(A_2, u_f(t)) .+ B noise(t+1)
z(t) = C_0 + C_1 u(t) + quad(C_2, u_f(t))
z_tilde(t) = z(t) + v(t+1)
"""
function quad(A::AbstractArray{<:Number,3}, x)
    return map(j -> dot(x, view(A, j, :, :), x), 1:size(A, 1))
end

# y += quad(A, x)
# The quad_muladd! uses on a vector of matrices for A
function quad_muladd!(y, A, x)
    @inbounds for j in 1:size(A, 1)
        @views y[j] += dot(x, A[j], x)
    end
    return y
end

# quadratic form pullback
function quad_pb(Δres::AbstractVector, A::AbstractArray{<:Number,3}, x::AbstractVector)
    ΔA = similar(A)
    Δx = zeros(length(x))
    tmp = x * x'
    for i in 1:size(A, 1)
        ΔA[i, :, :] .= tmp .* Δres[i]
        Δx += (A[i, :, :] + A[i, :, :]') * x .* Δres[i]
    end
    return ΔA, Δx
end

# inplace version with accumulation and using the cache of A[i] + A[i]', etc.
function quad_muladd_pb!(ΔA_vec, Δx, Δres, A_vec_sum, x)
    tmp = x * x'
    for (i, A_sum) in enumerate(A_vec_sum)  # @views @inbounds  ADD
        ΔA_vec[i] .+= tmp .* Δres[i]
        Δx += A_sum * x .* Δres[i]
    end
    return nothing
end

struct QuadraticStateSpaceProblem{isinplace,A_0type<:AbstractArray,A_1type<:AbstractArray,
                                  A_2type<:AbstractArray,Btype<:AbstractArray,
                                  C_0type<:AbstractArray,C_1type<:AbstractArray,
                                  C_2type<:AbstractArray,wtype,Rtype, # Distributions only
                                  utype,ttype,otype} <: AbstractStateSpaceProblem{isinplace}
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

function QuadraticStateSpaceProblem(A_0::A_0type, A_1::A_1type, A_2::A_2type, B::Btype,
                                    C_0::C_0type, C_1::C_1type, C_2::C_2type, u0::utype,
                                    tspan::ttype;
                                    obs_noise = (h0 = C_1 * u0;
                                                 MvNormal(zeros(eltype(h0), length(h0)), I)), # Assume the default measurement error is MvNormal with identity covariance
                                    observables = nothing,
                                    noise = nothing) where {A_0type<:AbstractArray,
                                                            A_1type<:AbstractArray,
                                                            A_2type<:AbstractArray,
                                                            Btype<:AbstractArray,
                                                            C_0type<:AbstractArray,
                                                            C_1type<:AbstractArray,
                                                            C_2type<:AbstractArray,utype,ttype}
    return QuadraticStateSpaceProblem{Val(false),A_0type,A_1type,A_2type,Btype,C_0type,C_1type,
                                      C_2type,typeof(noise),typeof(obs_noise),utype,ttype,
                                      typeof(observables)}(A_0::A_0type, A_1::A_1type, A_2::A_2type, # Evolution matrix
                                                           B::Btype, # Noise matrix
                                                           C_0::C_0type, C_1::C_1type, C_2::C_2type, # Observation matrix
                                                           noise, # Latent noise distribution
                                                           obs_noise, # Observation noise matrix
                                                           u0, # Initial condition
                                                           tspan, # Timespan to use
                                                           observables)
end

# Default is NoiseConditionalFilter
function CommonSolve.init(prob::QuadraticStateSpaceProblem, args...; kwargs...)
    return StateSpaceCache(prob, NoiseConditionalFilter())
end

function CommonSolve.init(prob::QuadraticStateSpaceProblem, solver::SciMLBase.SciMLAlgorithm,
                          args...; kwargs...)
    return StateSpaceCache(prob, solver)
end

function _solve!(prob::QuadraticStateSpaceProblem{isinplace,A_0type,A_1type,A_2type,Btype,C_0type,
                                                  C_1type,C_2type,wtype,Rtype,utype,ttype,otype},
                 ::NoiseConditionalFilter, args...;
                 kwargs...) where {isinplace,A_0type,A_1type,A_2type,Btype,C_0type,C_1type,C_2type,
                                   wtype,Rtype,utype,ttype,otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    t_0 = 1 + prob.tspan[1]
    @unpack A_0, A_1, A_2, B, C_0, C_1, C_2 = prob

    C_2_vec = [C_2[i, :, :] for i in 1:size(C_2, 1)] # should be the native datastructure
    A_2_vec = [A_2[i, :, :] for i in 1:size(A_2, 1)] # should be the native datastructure

    u_f = [zero(prob.u0) for _ in 1:T]  # TODO: move to internal algorithm cache
    u = [zero(prob.u0) for _ in 1:T] # TODO: move to internal algorithm cache
    z = [zero(C_0) for _ in 1:T] # TODO: move to internal algorithm cache

    u[1] .= prob.u0
    u_f[1] .= prob.u0
    z[1] .= C_0
    mul!(z[1], C_1, prob.u0, 1, 1)
    quad_muladd!(z[1], C_2_vec, prob.u0) #z0 .+= quad(C_2, prob.u0)

    loglik = 0.0
    @inbounds @views for t in 2:T
        mul!(u_f[t], A_1, u_f[t - 1])
        mul!(u_f[t], B, prob.noise[:, t - 1], 1, 1)

        u[t] .= A_0
        mul!(u[t], A_1, u[t - 1], 1, 1)
        quad_muladd!(u[t], A_2_vec, u_f[t - 1]) # u[t] .+= quad(A_2, u_f[t - 1])
        mul!(u[t], B, prob.noise[:, t - 1], 1, 1)

        z[t] .= C_0
        mul!(z[t], C_1, u[t], 1, 1)
        quad_muladd!(z[t], C_2_vec, u_f[t]) # z[t] .+= quad(C_2, u_f[t])
        loglik += logpdf(prob.obs_noise, prob.observables[:, t - 1] - z[t])
    end

    return StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)
end

function ChainRulesCore.rrule(::typeof(_solve!),
                              prob::QuadraticStateSpaceProblem{isinplace,A_0type,A_1type,A_2type,
                                                               Btype,C_0type,C_1type,C_2type,wtype,
                                                               Rtype,utype,ttype,otype},
                              ::NoiseConditionalFilter, args...;
                              kwargs...) where {isinplace,A_0type,A_1type,A_2type,Btype,C_0type,
                                                C_1type,C_2type,wtype,Rtype,utype,ttype,otype}
    # Preallocate values
    T = prob.tspan[2] - prob.tspan[1] + 1
    t_0 = 1 + prob.tspan[1]
    @unpack A_0, A_1, A_2, B, C_0, C_1, C_2 = prob

    C_2_vec = [C_2[i, :, :] for i in 1:size(C_2, 1)] # should be the native datastructure
    A_2_vec = [A_2[i, :, :] for i in 1:size(A_2, 1)] # should be the native datastructure

    u_f = [zero(prob.u0) for _ in 1:T]  # TODO: move to internal algorithm cache
    u = [zero(prob.u0) for _ in 1:T] # TODO: move to internal algorithm cache
    z = [zero(C_0) for _ in 1:T] # TODO: move to internal algorithm cache

    u[1] .= prob.u0
    u_f[1] .= prob.u0
    z[1] .= C_0
    mul!(z[1], C_1, prob.u0, 1, 1)
    quad_muladd!(z[1], C_2_vec, prob.u0) #z0 .+= quad(C_2, prob.u0)

    loglik = 0.0
    @inbounds @views for t in 2:T
        mul!(u_f[t], A_1, u_f[t - 1])
        mul!(u_f[t], B, prob.noise[:, t - 1], 1, 1)

        u[t] .= A_0
        mul!(u[t], A_1, u[t - 1], 1, 1)
        quad_muladd!(u[t], A_2_vec, u_f[t - 1]) # u[t] .+= quad(A_2, u_f[t - 1])
        mul!(u[t], B, prob.noise[:, t - 1], 1, 1)

        z[t] .= C_0
        mul!(z[t], C_1, u[t], 1, 1)
        quad_muladd!(z[t], C_2_vec, u_f[t]) # z[t] .+= quad(C_2, u_f[t])
        loglik += logpdf(prob.obs_noise, prob.observables[:, t - 1] - z[t])
    end

    sol = StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)

    function solve_pb(Δsol)
        Δlogpdf = Δsol.loglikelihood
        if iszero(Δlogpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end
        ΔA_0 = zero(A_0)
        ΔA_1 = zero(A_1)
        ΔA_2_vec = [zero(A) for A in A_2_vec] # should be native datastructure
        ΔA_2 = zero(A_2)

        ΔB = zero(B)
        ΔC_0 = zero(C_0)
        ΔC_1 = zero(C_1)
        ΔC_2_vec = [zero(A) for A in C_2_vec] # should be native datastructure
        ΔC_2 = zero(C_2)

        Δnoise = similar(prob.noise)
        Δu = [zero(prob.u0) for _ in 1:T]
        Δu_f = [zero(prob.u0) for _ in 1:T]
        Δu_f_temp = [zero(prob.u0) for _ in 1:T]
        A_2_vec_sum = [(A + A') for A in A_2_vec] # prep the sum since we will use it repeatedly
        C_2_vec_sum = [(A + A') for A in C_2_vec] # prep the sum since we will use it repeatedly

        @views @inbounds for t in T:-1:2
            Δz = Δlogpdf * (prob.observables[:, t - 1] - z[t]) ./ abs2.(prob.obs_noise.σ) # More generally, it should be Σ^-1 * (z_obs - z)
            # tmp1, tmp2 = quad_pb(Δz, C_2, u_f[t])
            # ΔC_2 += tmp1
            # Δu_f[t] .+= tmp2
            quad_muladd_pb!(ΔC_2_vec, Δu_f[t], Δz, C_2_vec_sum, u_f[t])

            mul!(Δu[t], C_1', Δz, 1, 1)
            tmp3, tmp4 = quad_pb(Δu[t], A_2, u_f[t - 1])

            quad_muladd_pb!(ΔA_2_vec, Δu_f_temp[t - 1], Δu[t], A_2_vec_sum, u_f[t - 1])

            ΔA_2 += tmp3
            Δu_f[t - 1] .+= tmp4
            Δu[t - 1] .= A_1' * Δu[t]
            mul!(Δu_f[t - 1], A_1', Δu_f[t], 1, 1)
            Δnoise[:, t - 1] = B' * (Δu[t] .+ Δu_f[t])
            # Now, deal with the coefficients
            ΔA_0 += Δu[t]
            mul!(ΔA_1, Δu[t], u[t - 1]', 1, 1)
            mul!(ΔA_1, Δu_f[t], u_f[t - 1]', 1, 1)
            ΔB += (Δu[t] + Δu_f[t]) * prob.noise[:, t - 1]'
            ΔC_0 += Δz
            mul!(ΔC_1, Δz, u[t]', 1, 1)
        end

        # Remove once the vector of matrices or column-major organized 3-tensor is the native datastructure for C_2/A_2
        for (i, ΔA_2_slice) in enumerate(ΔA_2_vec)
            ΔA_2[i, :, :] .= ΔA_2_slice
        end
        for (i, ΔC_2_slice) in enumerate(ΔC_2_vec)
            ΔC_2[i, :, :] .= ΔC_2_slice
        end

        return (NoTangent(),
                Tangent{typeof(prob)}(; A_0 = ΔA_0, A_1 = ΔA_1, A_2 = ΔA_2, B = ΔB, C_0 = ΔC_0,
                                      C_1 = ΔC_1, C_2 = ΔC_2, u0 = Δu[1] + Δu_f[1], noise = Δnoise),
                NoTangent(), map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
