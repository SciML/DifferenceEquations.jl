# Generic state-space model callbacks for DirectIteration solver
# The solver loop is in linear.jl — these methods define the model-specific behavior.

# =============================================================================
# NoiseSpec sentinel — provides size(_, 2) and eltype for get_concrete_noise
# =============================================================================

struct NoiseSpec{T}
    n_shocks::Int
end
NoiseSpec(n_shocks::Int, ::Type{T}) where {T} = NoiseSpec{T}(n_shocks)
Base.size(ns::NoiseSpec, i::Int) = i == 2 ? ns.n_shocks : 1
Base.eltype(::NoiseSpec{T}) where {T} = T

# =============================================================================
# Model interface methods for GenericStateSpaceProblem
# =============================================================================

function _noise_matrix(prob::GenericStateSpaceProblem)
    return prob.n_shocks > 0 ? NoiseSpec(prob.n_shocks, eltype(prob.u0)) : nothing
end

_init_model_state!!(::GenericStateSpaceProblem, cache) = nothing

@inline function _transition!!(x_next, x, w, prob::GenericStateSpaceProblem, cache, t)
    return prob.transition(x_next, x, w, prob.p, t - 2)  # 0-based time
end

@inline function _observation!!(y, x, prob::GenericStateSpaceProblem, cache, t)
    return prob.observation(y, x, prob.p, t - 1)  # 0-based time
end
