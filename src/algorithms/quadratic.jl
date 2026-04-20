# Quadratic state-space model dispatches for DirectIteration solver
# Two variants: unpruned (quad on x) and pruned (quad on linear-part u_f)
# Both plug into the generic _solve_direct_iteration! loop via these methods.

# --- Noise matrix extraction ---
_noise_matrix(prob::AnyQuadraticProblem) = prob.B

# --- Model initialization ---
_init_model_state!!(::QuadraticStateSpaceProblem, cache) = nothing

function _init_model_state!!(prob::PrunedQuadraticStateSpaceProblem, cache)
    cache.u_f[1] = assign!!(cache.u_f[1], prob.u0)
    return nothing
end

# --- Observation flag (shared with linear, already defined) ---
# _has_observations(sol) = !isnothing(sol.z)  # defined in linear.jl

# --- Quadratic form helper ---
# Computes q[i] = v' * A_2[i, :, :] * v for each output dimension
@inline function _add_quadratic!!(y, A_2, v)
    if ismutable(y)
        @inbounds for i in 1:length(y)
            y[i] += dot(v, view(A_2, i, :, :), v)
        end
        return y
    else
        n = length(y)
        return y + typeof(y)(ntuple(i -> dot(v, view(A_2, i, :, :), v), n))
    end
end

# =============================================================================
# Unpruned quadratic: quad(A_2, x)
# =============================================================================

@inline function _transition!!(x_next, x, w, prob::QuadraticStateSpaceProblem, cache, t)
    (; A_0, A_1, A_2, B) = prob
    x_next = copyto!!(x_next, A_0)
    x_next = mul!!(x_next, A_1, x, 1.0, 1.0)
    x_next = _add_quadratic!!(x_next, A_2, x)
    x_next = muladd!!(x_next, B, w)
    return x_next
end

@inline function _observation!!(y, x, prob::QuadraticStateSpaceProblem, cache, t)
    (; C_0, C_1, C_2) = prob
    y = copyto!!(y, C_0)
    y = mul!!(y, C_1, x, 1.0, 1.0)
    y = _add_quadratic!!(y, C_2, x)
    return y
end

# =============================================================================
# Pruned quadratic: quad(A_2, u_f) where u_f tracks the linear-part state
# =============================================================================

@inline function _transition!!(x_next, x, w, prob::PrunedQuadraticStateSpaceProblem, cache, t)
    (; A_0, A_1, A_2, B) = prob
    u_f_prev = cache.u_f[t - 1]
    # Advance u_f: u_f[t] = A_1 * u_f[t-1] + B * w
    u_f_new = mul!!(cache.u_f[t], A_1, u_f_prev)
    u_f_new = muladd!!(u_f_new, B, w)
    cache.u_f[t] = u_f_new
    # Full transition: x_next = A_0 + A_1*x + quad(A_2, u_f_prev) + B*w
    x_next = copyto!!(x_next, A_0)
    x_next = mul!!(x_next, A_1, x, 1.0, 1.0)
    x_next = _add_quadratic!!(x_next, A_2, u_f_prev)
    x_next = muladd!!(x_next, B, w)
    return x_next
end

@inline function _observation!!(y, x, prob::PrunedQuadraticStateSpaceProblem, cache, t)
    (; C_0, C_1, C_2) = prob
    u_f = cache.u_f[t]
    y = copyto!!(y, C_0)
    y = mul!!(y, C_1, x, 1.0, 1.0)
    y = _add_quadratic!!(y, C_2, u_f)
    return y
end

# --- Pruned quadratic: save_everystep=false overloads (ping-pong u_f) ---

function _init_model_state!!(prob::PrunedQuadraticStateSpaceProblem, cache, ::Val{false})
    cache.u_f[1] = assign!!(cache.u_f[1], prob.u0)
    return nothing
end

@inline function _transition!!(
        x_next, x, w, prob::PrunedQuadraticStateSpaceProblem, cache, t, ::Val{false}
    )
    (; A_0, A_1, A_2, B) = prob
    uf_prev_idx = _u_idx_pingpong(t - 1)
    uf_curr_idx = _u_idx_pingpong(t)
    u_f_prev = cache.u_f[uf_prev_idx]
    u_f_new = mul!!(cache.u_f[uf_curr_idx], A_1, u_f_prev)
    u_f_new = muladd!!(u_f_new, B, w)
    cache.u_f[uf_curr_idx] = u_f_new
    x_next = copyto!!(x_next, A_0)
    x_next = mul!!(x_next, A_1, x, 1.0, 1.0)
    x_next = _add_quadratic!!(x_next, A_2, u_f_prev)
    x_next = muladd!!(x_next, B, w)
    return x_next
end

@inline function _observation!!(
        y, x, prob::PrunedQuadraticStateSpaceProblem, cache, t, ::Val{false}
    )
    (; C_0, C_1, C_2) = prob
    u_f = cache.u_f[_u_idx_pingpong(t)]
    y = copyto!!(y, C_0)
    y = mul!!(y, C_1, x, 1.0, 1.0)
    y = _add_quadratic!!(y, C_2, u_f)
    return y
end
