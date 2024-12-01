# This file contains utilities for use in the algorithms to make the code more generic and able to handle different model variations (e.g., no observables, no observation equation, etc.)

# Temporary.  Eventually, move to use sciml NoiseProcess with better rng support/etc.
get_concrete_noise(prob, noise, B, T) = noise # maybe do a promotion to an AbstractVectorOfVector type
get_concrete_noise(prob, noise, B::Nothing, T) = nothing # if no noise matrix given, do not create noise
get_concrete_noise(prob, noise::Nothing, B::Nothing, T) = nothing # if no noise matrix given, do not create noise
get_concrete_noise(prob, noise::Nothing, B, T) = randn(eltype(B), size(B, 2), T) # default is unit Gaussian
get_concrete_noise(prob, noise::UnivariateDistribution, B, T) = rand(noise, size(B, 2), T) # iid
get_concrete_noise(prob, noise::UnivariateDistribution, B::Nothing, T) = nothing

# Utility functions to conditionally check size if not-nothing
maybe_check_size(m::AbstractMatrix, index, val) = (size(m, index) == val)
maybe_check_size(m::Nothing, index, val) = true

function maybe_check_size(m1::AbstractArray, index1, m2::AbstractArray, index2)
    (size(m1, index1) ==
     size(m2, index2))
end
maybe_check_size(m1::Nothing, index1, m2, index2) = true
maybe_check_size(m1, index1, m2::Nothing, index2) = true
maybe_check_size(m1::Nothing, index1, m2::Nothing, index2) = true

Base.@propagate_inbounds @inline function maybe_logpdf(observables_noise::Distribution,
        observables::AbstractMatrix, t,
        z::AbstractVector, s)
    logpdf(observables_noise,
        view(observables,
            :,
            t) -
        z[s])
end
# Don't accumulate likelihoods if no observations or observatino noise
maybe_logpdf(observables_noise, observable, t, z, s) = 0.0

# If no noise process is given, don't add in noise in simulation
Base.@propagate_inbounds @inline function maybe_muladd!(x, B, noise, t)
    mul!(x, B, view(noise, :, t), 1, 1)
end
maybe_muladd!(x, B::Nothing, noise, t) = nothing

Base.@propagate_inbounds @inline maybe_muladd!(x, A, B) = mul!(x, A, B, 1, 1)
maybe_muladd!(x, A::Nothing, B) = nothing

# need transpose versions for gradients
Base.@propagate_inbounds @inline maybe_muladd_transpose!(x, C, Δz) = mul!(x, C', Δz, 1, 1)
maybe_muladd_transpose!(x, C::Nothing, Δz) = nothing
Base.@propagate_inbounds @inline function maybe_muladd_transpose!(ΔB::AbstractMatrix,
        Δu_temp,
        noise::AbstractMatrix, t)
    mul!(ΔB, Δu_temp, view(noise, :, t)', 1, 1)
    return nothing
end
maybe_muladd_transpose!(ΔB, Δu_temp, noise, t) = nothing
Base.@propagate_inbounds @inline maybe_mul!(x, t, A, y, s) = mul!(x[t], A, y[s])
maybe_mul!(x::Nothing, t, A, y, s) = nothing
# Need transpose versions for rrule
Base.@propagate_inbounds @inline maybe_mul_transpose!(x, t, A, y, s) = mul!(x[t], A', y[s])
maybe_mul_transpose!(x::Nothing, t, A, y, s) = nothing
Base.@propagate_inbounds @inline function maybe_mul_transpose!(Δnoise, t, B, y)
    mul!(view(Δnoise, :, t),
        B', y)
end
maybe_mul_transpose!(Δnoise::Nothing, t, B, y) = nothing

# Utilities to get distribution for logpdf from observation error argument
make_observables_noise(observables_noise::Nothing) = nothing
make_observables_noise(observables_noise::AbstractMatrix) = MvNormal(observables_noise)
function make_observables_noise(observables_noise::AbstractVector)
    MvNormal(Diagonal(observables_noise))
end

# Utilities to get covariance matrix from observation error argument for kalman filter.  e.g. vector is diagonal, etc.
make_observables_covariance_matrix(observables_noise::AbstractMatrix) = observables_noise
function make_observables_covariance_matrix(observables_noise::AbstractVector)
    Diagonal(observables_noise)
end

#Add in observation noise to the output if simulated (i.e, observables not given) and there is observation_noise provided
function maybe_add_observation_noise!(z, observables_noise::Distribution,
        observables::Nothing)
    # add noise to the vector of vectors componentwise
    for z_val in z
        z_val .+= rand(observables_noise)
    end
    return nothing
end
maybe_add_observation_noise!(z, observables_noise, observables) = nothing  #otherwise do nothing

#Maybe add observation noise, if observables and their adjoints given
Base.@propagate_inbounds @inline function maybe_add_Δ!(Δz, Δsol_z::AbstractVector, t)
    Δz .+= Δsol_z[t]
    return nothing
end
maybe_add_Δ!(Δz, Δsol_z, t) = nothing

Base.@propagate_inbounds @inline function maybe_add_Δ_slice!(Δnoise::AbstractMatrix,
        ΔW::AbstractMatrix, t)
    Δnoise[:, t] .+= view(ΔW, :, t)
    return nothing
end
maybe_add_Δ_slice!(Δz, Δsol_A, t) = nothing

# Don't add logpdf to observables unless provided
# TODO: check if this can be repalced with the following and if it has a performance regression for diagonal noise covariance
# ldiv!(Δz, observables_noise.Σ.chol, innovation[t])
# rmul!(Δlogpdf, Δz)
Base.@propagate_inbounds @inline function maybe_add_Δ_logpdf!(Δz::AbstractArray{<:Real, 1},
        Δlogpdf::Number,
        observables::AbstractArray{
            <:Real,
            2},
        z::AbstractArray{T, 1},
        t,
        observables_noise_cov::AbstractArray{
            <:Real,
            1
        }) where {
        T
}
    Δz .= Δlogpdf * (view(observables, :, t - 1) - z[t]) ./
          observables_noise_cov
    return nothing
end
# Otherwise do nothing
function maybe_add_Δ_logpdf!(Δz, Δlogpdf, observables, z, t, observables_noise_cov)
    nothing
end

# Only allocate if observation equation
allocate_z(prob, C, u0, T) = [zeros(size(C, 1)) for _ in 1:T]
allocate_z(prob, C::Nothing, u0, T) = nothing

# Maybe zero
maybe_zero(A::AbstractArray) = zero(A)
maybe_zero(A::Nothing) = nothing
maybe_zero(A::AbstractArray, i::Int64) = zero(A[i])
maybe_zero(A::Nothing, i) = nothing

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
    tmp = x * x'  # could add in a temp here
    @inbounds for (i, A_sum) in enumerate(A_vec_sum)  # @views @inbounds  ADD
        ΔA_vec[i] .+= tmp .* Δres[i]
        Δx .+= A_sum * x .* Δres[i]
    end
    return nothing
end
