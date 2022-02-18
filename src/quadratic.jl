"""
u_f(t+1) = A_1 u_f(t) .+ B * noise(t+1)
u(t+1) = A_0 + A_1 u(t) + quad(A_2, u_f(t)) .+ B noise(t+1)
z(t) = C_0 + C_1 u(t) + quad(C_2, u_f(t))
z_tilde(t) = z(t) + v(t+1)
"""

# old quad and adjoint replaced by inplace accumulation versions.
# function quad(A::AbstractArray{<:Number,3}, x)
#     return map(j -> dot(x, view(A, j, :, :), x), 1:size(A, 1))
# end
# # quadratic form pullback
# function quad_pb(Δres::AbstractVector, A::AbstractArray{<:Number,3}, x::AbstractVector)
#     ΔA = similar(A)
#     Δx = zeros(length(x))
#     tmp = x * x'
#     for i in 1:size(A, 1)
#         ΔA[i, :, :] .= tmp .* Δres[i]
#         Δx += (A[i, :, :] + A[i, :, :]') * x .* Δres[i]
#     end
#     return ΔA, Δx
# end

# y += quad(A, x)
# The quad_muladd! uses on a vector of matrices for A
function quad_muladd!(y, A, x)
    @inbounds for j in 1:size(A, 1)
        @views y[j] += dot(x, A[j], x)
    end
    return y
end

# inplace version with accumulation and using the cache of A[i] + A[i]', etc.
function quad_muladd_pb!(ΔA_vec, Δx, Δres, A_vec_sum, x)
    tmp = x * x'
    @inbounds for (i, A_sum) in enumerate(A_vec_sum)  # @views @inbounds  ADD
        ΔA_vec[i] .+= tmp .* Δres[i]
        Δx .+= A_sum * x .* Δres[i]
    end
    return nothing
end

Base.@kwdef struct QuadraticStateSpaceProblemCache{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,
                                                   T15,T16,T17,T18,T19}
    u::T1
    z::T2
    innovation::T3
    u_f::T4
    A_2_vec::T5 # remove after native datastructure
    C_2_vec::T6 # remove after native datastructure

    # Only for rrule
    A_2_vec_sum::T7
    C_2_vec_sum::T8
    Δu::T9
    Δu_f::T10
    Δnoise::T11
    ΔA_2_vec::T12
    ΔC_2_vec::T13
    temp_N_N::T14
    temp_M_M::T15
    temp_M_N::T16
    temp_N_M::T17
    temp_M::T18
    temp_N::T19
end

function QuadraticStateSpaceProblemCache{DT}(N, M, N_z, T,
                                             ::Val{AllocateAD} = Val(true)) where {DT,AllocateAD}
    return QuadraticStateSpaceProblemCache(; u = [Vector{DT}(undef, N) for _ in 1:T],
                                           u_f = [Vector{DT}(undef, N) for _ in 1:T],
                                           z = [Vector{DT}(undef, N_z) for _ in 1:T],
                                           innovation = [Vector{DT}(undef, N_z) for _ in 1:T],
                                           A_2_vec = [Matrix{DT}(undef, N, N) for _ in 1:N],
                                           C_2_vec = [Matrix{DT}(undef, N, N) for _ in 1:N_z],

                                           # Only on the rrule
                                           A_2_vec_sum = AllocateAD ?
                                                         [Matrix{DT}(undef, N, N) for _ in 1:N] :
                                                         nothing,
                                           C_2_vec_sum = AllocateAD ?
                                                         [Matrix{DT}(undef, N, N) for _ in 1:N_z] :
                                                         nothing,
                                           Δu = AllocateAD ? [Vector{DT}(undef, N) for _ in 1:T] :
                                                nothing,
                                           Δu_f = AllocateAD ? [Vector{DT}(undef, N) for _ in 1:T] :
                                                  nothing,
                                           Δnoise = AllocateAD ? Matrix{DT}(undef, M, T - 1) :
                                                    nothing,
                                           ΔA_2_vec = AllocateAD ?
                                                      [Matrix{DT}(undef, N, N) for _ in 1:N] :
                                                      nothing,
                                           ΔC_2_vec = AllocateAD ?
                                                      [Matrix{DT}(undef, N, N) for _ in 1:N_z] :
                                                      nothing,
                                           temp_N_N = AllocateAD ? Matrix{DT}(undef, N, N) :
                                                      nothing,
                                           temp_M_M = AllocateAD ? Matrix{DT}(undef, M, M) :
                                                      nothing,
                                           temp_M_N = AllocateAD ? Matrix{DT}(undef, M, N) :
                                                      nothing,
                                           temp_N_M = AllocateAD ? Matrix{DT}(undef, N, M) :
                                                      nothing,
                                           temp_M = AllocateAD ? Vector{DT}(undef, M) : nothing,
                                           temp_N = AllocateAD ? Vector{DT}(undef, N) : nothing)
end

struct QuadraticStateSpaceProblem{isinplace,A_0type<:AbstractArray,A_1type<:AbstractArray,
                                  A_2type<:AbstractArray,Btype<:AbstractArray,
                                  C_0type<:AbstractArray,C_1type<:AbstractArray,
                                  C_2type<:AbstractArray,wtype,Rtype, # Distributions only
                                  utype,ttype,otype,ctype} <: AbstractStateSpaceProblem{isinplace}
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
    cache::ctype # cache    
end

function QuadraticStateSpaceProblem(A_0::A_0type, A_1::A_1type, A_2::A_2type, B::Btype,
                                    C_0::C_0type, C_1::C_1type, C_2::C_2type, u0::utype,
                                    tspan::ttype;
                                    obs_noise = (h0 = C_1 * u0;
                                                 MvNormal(zeros(eltype(h0), length(h0)), I)), # Assume the default measurement error is MvNormal with identity covariance
                                    observables = nothing, noise = nothing,
                                    cache = QuadraticStateSpaceProblemCache{eltype(u0)}(length(u0),
                                                                                        length(obs_noise),
                                                                                        size(observables,
                                                                                             1),
                                                                                        size(observables,
                                                                                             2) + 1)) where {A_0type<:AbstractArray,
                                                                                                             A_1type<:AbstractArray,
                                                                                                             A_2type<:AbstractArray,
                                                                                                             Btype<:AbstractArray,
                                                                                                             C_0type<:AbstractArray,
                                                                                                             C_1type<:AbstractArray,
                                                                                                             C_2type<:AbstractArray,
                                                                                                             utype,
                                                                                                             ttype}
    return QuadraticStateSpaceProblem{Val(false),A_0type,A_1type,A_2type,Btype,C_0type,C_1type,
                                      C_2type,typeof(noise),typeof(obs_noise),utype,ttype,
                                      typeof(observables),typeof(cache)}(A_0::A_0type, A_1::A_1type,
                                                                         A_2::A_2type, # Evolution matrix
                                                                         B::Btype, # Noise matrix
                                                                         C_0::C_0type, C_1::C_1type,
                                                                         C_2::C_2type, # Observation matrix
                                                                         noise, # Latent noise distribution
                                                                         obs_noise, # Observation noise matrix
                                                                         u0, # Initial condition
                                                                         tspan, # Timespan to use
                                                                         observables, cache)
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
    @unpack A_0, A_1, A_2, B, C_0, C_1, C_2 = prob
    @unpack u, u_f, z, innovation, C_2_vec, A_2_vec = prob.cache  # problem cache
    @assert length(u) >= T && length(z) >= T && length(innovation) >= T # ensure enough space allocated

    # TODO: This should be the native datastrcture of A_2 and C_2.  Remove when it is
    for i in 1:size(C_2, 1)
        C_2_vec[i] .= C_2[i, :, :]
    end
    for i in 1:size(A_2, 1)
        A_2_vec[i] .= A_2[i, :, :]
    end

    u[1] .= prob.u0
    u_f[1] .= prob.u0
    z[1] .= C_0
    mul!(z[1], C_1, prob.u0, 1, 1)
    quad_muladd!(z[1], C_2_vec, prob.u0) #z0 .+= quad(C_2, prob.u0)

    loglik = 0.0
    @inbounds @views for t in 2:T
        mul!(u_f[t], A_1, u_f[t - 1])
        mul!(u_f[t], B, view(prob.noise, :, t - 1), 1, 1)

        u[t] .= A_0
        mul!(u[t], A_1, u[t - 1], 1, 1)
        quad_muladd!(u[t], A_2_vec, u_f[t - 1]) # u[t] .+= quad(A_2, u_f[t - 1])
        mul!(u[t], B, view(prob.noise, :, t - 1), 1, 1)

        z[t] .= C_0
        mul!(z[t], C_1, u[t], 1, 1)
        quad_muladd!(z[t], C_2_vec, u_f[t]) # z[t] .+= quad(C_2, u_f[t])
        innovation[t] .= prob.observables[:, t - 1] - z[t]
        loglik += logpdf(prob.obs_noise, innovation[t])
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
    @unpack A_0, A_1, A_2, B, C_0, C_1, C_2 = prob
    @unpack u, u_f, z, innovation, C_2_vec, A_2_vec = prob.cache  # problem cache
    @assert length(u) >= T && length(z) >= T && length(innovation) >= T # ensure enough space allocated

    # TODO: This should be the native datastrcture of A_2 and C_2.  Remove when it is
    for i in 1:size(C_2, 1)
        C_2_vec[i] .= C_2[i, :, :]
    end
    for i in 1:size(A_2, 1)
        A_2_vec[i] .= A_2[i, :, :]
    end

    u[1] .= prob.u0
    u_f[1] .= prob.u0
    z[1] .= C_0
    mul!(z[1], C_1, prob.u0, 1, 1)
    quad_muladd!(z[1], C_2_vec, prob.u0) #z0 .+= quad(C_2, prob.u0)

    loglik = 0.0
    @inbounds @views for t in 2:T
        mul!(u_f[t], A_1, u_f[t - 1])
        mul!(u_f[t], B, view(prob.noise, :, t - 1), 1, 1)

        u[t] .= A_0
        mul!(u[t], A_1, u[t - 1], 1, 1)
        quad_muladd!(u[t], A_2_vec, u_f[t - 1]) # u[t] .+= quad(A_2, u_f[t - 1])
        mul!(u[t], B, view(prob.noise, :, t - 1), 1, 1)

        z[t] .= C_0
        mul!(z[t], C_1, u[t], 1, 1)
        quad_muladd!(z[t], C_2_vec, u_f[t]) # z[t] .+= quad(C_2, u_f[t])
        innovation[t] .= prob.observables[:, t - 1] - z[t]
        loglik += logpdf(prob.obs_noise, innovation[t])
    end

    sol = StateSpaceSolution(nothing, nothing, nothing, nothing, loglik)

    function solve_pb(Δsol)
        #@unpack Δnoise,  = prob.cache  # problem cache
        @unpack ΔA_2_vec, ΔC_2_vec, Δu, Δu_f, temp_N_N, A_2_vec_sum, C_2_vec_sum = prob.cache  # problem cache
        Δlogpdf = Δsol.loglikelihood
        if iszero(Δlogpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end

        # zero out anything in the cache just in case.  Check if necessary
        foreach(fill_zero!, ΔA_2_vec)
        foreach(fill_zero!, ΔC_2_vec)
        foreach(fill_zero!, Δu)
        foreach(fill_zero!, Δu_f)
        # fill_zero!(Δnoise)

        # prep sum since used repeatedly
        for (i, A) in enumerate(A_2_vec)
            copy!(A_2_vec_sum[i], A)
            transpose!(temp_N_N, A)
            A_2_vec_sum[i] .+= temp_N_N
        end
        for (i, C) in enumerate(C_2_vec)
            copy!(C_2_vec_sum[i], C)
            transpose!(temp_N_N, C)
            C_2_vec_sum[i] .+= temp_N_N
        end

        Δnoise = similar(prob.noise)

        # Could move into cache later for huge problems
        ΔA_0 = zero(A_0)
        ΔA_1 = zero(A_1)
        ΔA_2 = zero(A_2)
        ΔB = zero(B)
        ΔC_0 = zero(C_0)
        ΔC_1 = zero(C_1)
        ΔC_2 = zero(C_2)

        # Δnoise = similar(prob.noise)
        @show size(Δnoise), size(prob.noise)
        @assert size(Δnoise) == size(prob.noise)

        @views @inbounds for t in T:-1:2
            Δz = Δlogpdf * innovation[t] ./ diag(prob.obs_noise.Σ) # More generally, it should be Σ^-1 * (z_obs - z)

            # inplace adoint of quadratic form with accumulation
            quad_muladd_pb!(ΔC_2_vec, Δu_f[t], Δz, C_2_vec_sum, u_f[t])
            mul!(Δu[t], C_1', Δz, 1, 1)

            quad_muladd_pb!(ΔA_2_vec, Δu_f[t - 1], Δu[t], A_2_vec_sum, u_f[t - 1])
            mul!(Δu[t - 1], A_1', Δu[t])
            mul!(Δu_f[t - 1], A_1', Δu_f[t], 1, 1)
            Δu_f_sum = Δu[t] .+ Δu_f[t]
            mul!(Δnoise[:, t - 1], B', Δu_f_sum)
            # Now, deal with the coefficients
            ΔA_0 += Δu[t]
            mul!(ΔA_1, Δu[t], u[t - 1]', 1, 1)
            mul!(ΔA_1, Δu_f[t], u_f[t - 1]', 1, 1)
            mul!(ΔB, Δu_f_sum, prob.noise[:, t - 1]', 1, 1)
            ΔC_0 += Δz
            mul!(ΔC_1, Δz, u[t]', 1, 1)
        end

        # Remove once the vector of matrices or column-major organized 3-tensor is the native datastructure for C_2/A_2
        @views @inbounds for (i, ΔA_2_slice) in enumerate(ΔA_2_vec)
            ΔA_2[i, :, :] .= ΔA_2_slice
        end
        @views @inbounds for (i, ΔC_2_slice) in enumerate(ΔC_2_vec)
            ΔC_2[i, :, :] .= ΔC_2_slice
        end

        return (NoTangent(),
                Tangent{typeof(prob)}(; A_0 = ΔA_0, A_1 = ΔA_1, A_2 = ΔA_2, B = ΔB, C_0 = ΔC_0,
                                      C_1 = ΔC_1, C_2 = ΔC_2, u0 = Δu[1] + Δu_f[1], noise = Δnoise),
                NoTangent(), map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
