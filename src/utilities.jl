# Utilities for algorithms to handle different model variations
# (e.g., no observables, no observation equation, etc.)

# =============================================================================
# Noise handling — vector of vectors only
# =============================================================================

# Pass-through: already a vector of vectors
get_concrete_noise(prob, noise::AbstractVector{<:AbstractVector}, B, T) = noise

# No noise matrix: no noise regardless of noise argument
get_concrete_noise(prob, noise, B::Nothing, T) = nothing
get_concrete_noise(prob, noise::Nothing, B::Nothing, T) = nothing
# Disambiguation: B=nothing takes precedence
get_concrete_noise(prob, noise::AbstractVector{<:AbstractVector}, B::Nothing, T) = nothing

# Generate random noise as vector of vectors
function get_concrete_noise(prob, noise::Nothing, B, T)
    return [randn(eltype(B), size(B, 2)) for _ in 1:T]
end

# =============================================================================
# Copy noise into cache buffers
# =============================================================================

"""
    copy_noise_to_cache!(cache_noise, noise)

Copy concrete noise into preallocated cache noise buffers.
"""
function copy_noise_to_cache!(cache_noise, noise)
    @inbounds for t in eachindex(cache_noise)
        cache_noise[t] = assign!!(cache_noise[t], noise[t])
    end
    return cache_noise
end
copy_noise_to_cache!(cache_noise, ::Nothing) = nothing
copy_noise_to_cache!(::Nothing, ::Nothing) = nothing

# =============================================================================
# Observables handling — vector of vectors only
# =============================================================================

"""
    get_observable(observables::AbstractVector{<:AbstractVector}, t)

Get observation at time t from vector-of-vectors observables.
"""
Base.@propagate_inbounds @inline get_observable(
    observables::AbstractVector{<:AbstractVector}, t
) = observables[t]

# =============================================================================
# Conditional size checking
# =============================================================================

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
# Observation noise covariance
# =============================================================================

# Covariance matrix for Kalman filter and loglik computation.
# observables_noise must be an AbstractMatrix (e.g., Diagonal(d), Symmetric(H*H'), or Matrix).
make_observables_covariance_matrix(observables_noise::AbstractMatrix) = observables_noise
function make_observables_covariance_matrix(observables_noise::AbstractVector)
    return error(
        "observables_noise must be an AbstractMatrix (e.g., Diagonal(d)). " *
            "Got a Vector. Use Diagonal(d) to construct a diagonal covariance matrix."
    )
end

# =============================================================================
# Observation noise simulation (for DirectIteration without observables)
# =============================================================================

"""
    _add_observation_noise!(z, F_chol)

Add observation noise to simulated observations using a pre-computed Cholesky factor.
`F_chol` is an upper-triangular Cholesky factor (R = U'U), so L = U'.
"""
function _add_observation_noise!(z, F_chol)
    M = size(F_chol, 1)
    for z_val in z
        if !isnothing(z_val)
            z_val .+= F_chol.L * randn(M)
        end
    end
    return nothing
end
