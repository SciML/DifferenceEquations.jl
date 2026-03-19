# Utilities for algorithms to handle different model variations
# (e.g., no observables, no observation equation, etc.)

# =============================================================================
# Noise handling — returns vector of vectors
# =============================================================================

# Pass-through: already a vector of vectors
get_concrete_noise(prob, noise::AbstractVector{<:AbstractVector}, B, T) = noise

# Matrix noise: convert to vector of vectors
function get_concrete_noise(prob, noise::AbstractMatrix, B, T)
    return [noise[:, t] for t in 1:size(noise, 2)]
end

# No noise matrix: no noise regardless of noise argument
get_concrete_noise(prob, noise, B::Nothing, T) = nothing
get_concrete_noise(prob, noise::Nothing, B::Nothing, T) = nothing
# Disambiguations: B=nothing takes precedence
get_concrete_noise(prob, noise::AbstractVector{<:AbstractVector}, B::Nothing, T) = nothing
get_concrete_noise(prob, noise::AbstractMatrix, B::Nothing, T) = nothing

# Generate random noise as vector of vectors
function get_concrete_noise(prob, noise::Nothing, B, T)
    return [randn(eltype(B), size(B, 2)) for _ in 1:T]
end

# iid noise from distribution as vector of vectors
function get_concrete_noise(prob, noise::UnivariateDistribution, B, T)
    return [rand(noise, size(B, 2)) for _ in 1:T]
end

# Disambiguation: no noise matrix takes precedence over distribution
get_concrete_noise(prob, noise::UnivariateDistribution, B::Nothing, T) = nothing

# =============================================================================
# Copy noise into cache buffers
# =============================================================================

"""
    copy_noise_to_cache!(cache_noise, noise)

Copy concrete noise into preallocated cache noise buffers.
"""
function copy_noise_to_cache!(cache_noise, noise)
    @inbounds for t in eachindex(cache_noise)
        assign!!(cache_noise[t], noise[t])
    end
    return cache_noise
end
copy_noise_to_cache!(cache_noise, ::Nothing) = nothing
copy_noise_to_cache!(::Nothing, ::Nothing) = nothing

# =============================================================================
# Observables handling — support both matrix and vec-of-vecs
# =============================================================================

"""
    get_observable(observables::AbstractMatrix, t)

Get observation at time t from matrix-format observables.
"""
Base.@propagate_inbounds @inline get_observable(observables::AbstractMatrix, t) =
    view(observables, :, t)

"""
    get_observable(observables::AbstractVector{<:AbstractVector}, t)

Get observation at time t from vector-of-vectors observables.
"""
Base.@propagate_inbounds @inline get_observable(observables::AbstractVector{<:AbstractVector}, t) =
    observables[t]

# =============================================================================
# Conditional size checking
# =============================================================================

maybe_check_size(m::AbstractMatrix, index::Integer, val::Integer) = (size(m, index) == val)
maybe_check_size(m::AbstractVector, index::Integer, val::Integer) = (index == 1 ? length(m) == val : true)
maybe_check_size(m::Nothing, index::Integer, val::Integer) = true

function maybe_check_size(
        m1::AbstractArray, index1::Integer,
        m2::AbstractArray, index2::Integer
    )
    return size(m1, index1) == size(m2, index2)
end
maybe_check_size(m1::Nothing, index1::Integer, m2, index2::Integer) = true
maybe_check_size(m1, index1::Integer, m2::Nothing, index2::Integer) = true
maybe_check_size(m1::Nothing, index1::Integer, m2::Nothing, index2::Integer) = true

# Size check for vector of vectors
function maybe_check_size(m::AbstractVector{<:AbstractVector}, index::Integer, val::Integer)
    if index == 1
        return isempty(m) || length(m[1]) == val
    elseif index == 2
        return length(m) == val
    end
    return true
end

# =============================================================================
# Conditional log-likelihood computation
# =============================================================================

"""
    maybe_logpdf(observables_noise, observables, t, z, s)

Compute log-likelihood contribution if observations and noise are provided.
Supports both matrix and vector-of-vectors observables formats.
"""
Base.@propagate_inbounds @inline function maybe_logpdf(
        observables_noise::Distribution,
        observables::AbstractMatrix, t,
        z::AbstractVector, s
    )
    return logpdf(observables_noise, view(observables, :, t) - z[s])
end

Base.@propagate_inbounds @inline function maybe_logpdf(
        observables_noise::Distribution,
        observables::AbstractVector{<:AbstractVector}, t,
        z::AbstractVector, s
    )
    return logpdf(observables_noise, observables[t] - z[s])
end

# Don't accumulate likelihoods if no observations or observation noise
maybe_logpdf(observables_noise, observable, t, z, s) = 0.0

# =============================================================================
# Observation noise distribution construction
# =============================================================================

make_observables_noise(observables_noise::Nothing) = nothing
make_observables_noise(observables_noise::AbstractMatrix) = MvNormal(observables_noise)
function make_observables_noise(observables_noise::AbstractVector)
    return MvNormal(Diagonal(observables_noise))
end

# Covariance matrix for Kalman filter
make_observables_covariance_matrix(observables_noise::AbstractMatrix) = observables_noise
function make_observables_covariance_matrix(observables_noise::AbstractVector)
    return Diagonal(observables_noise)
end

# =============================================================================
# Observation noise simulation
# =============================================================================

function maybe_add_observation_noise!(
        z, observables_noise::Distribution,
        observables::Nothing
    )
    for z_val in z
        z_val .+= rand(observables_noise)
    end
    return nothing
end
maybe_add_observation_noise!(z, observables_noise, observables) = nothing

# =============================================================================
# Legacy helpers (kept for backward compatibility during transition)
# =============================================================================

# Conditional matrix-vector multiply-add
Base.@propagate_inbounds @inline function maybe_muladd!(x, B, noise, t)
    return mul!(x, B, view(noise, :, t), 1, 1)
end
maybe_muladd!(x, B::Nothing, noise, t) = nothing

Base.@propagate_inbounds @inline maybe_muladd!(x, A, B) = mul!(x, A, B, 1, 1)
maybe_muladd!(x, A::Nothing, B) = nothing

Base.@propagate_inbounds @inline maybe_mul!(x, t, A, y, s) = mul!(x[t], A, y[s])
maybe_mul!(x::Nothing, t, A, y, s) = nothing

# Only allocate if observation equation
allocate_z(prob, C, u0, T) = [zeros(size(C, 1)) for _ in 1:T]
allocate_z(prob, C::Nothing, u0, T) = nothing

# =============================================================================
# Quadratic form helpers (legacy, kept for reference)
# =============================================================================

# y += quad(A, x) using vector of matrices
function quad_muladd!(y, A, x)
    @inbounds for j in 1:size(A, 1)
        @views y[j] += dot(x, A[j], x)
    end
    return y
end
