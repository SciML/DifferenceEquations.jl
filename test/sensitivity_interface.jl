# Minimal sensitivity interface MWE
# Tests Enzyme AD through SciML-like struct construction + solve! pattern.
# Serves as reproducible MWE for SciML/Enzyme developers.

using LinearAlgebra, Test, Enzyme, EnzymeTestUtils

# =============================================================================
# Minimal problem type (immutable — Enzyme handles struct construction fine
# when all args are Duplicated)
# =============================================================================

struct MinimalProblem{AT, UT}
    A::AT
    u0::UT
end

struct MinimalCache{UT}
    u::Vector{UT}
end

function alloc_minimal_cache(u0, T)
    return MinimalCache([similar(u0) for _ in 1:T])
end

function zero_minimal_cache!(cache)
    for t in eachindex(cache.u)
        cache.u[t] .= 0
    end
    return cache
end

# =============================================================================
# Solve: u[t+1] = A * u[t], in-place mutation of cache
# =============================================================================

function minimal_solve!(prob::MinimalProblem, cache::MinimalCache)
    cache.u[1] .= prob.u0
    for t in 1:(length(cache.u) - 1)
        mul!(cache.u[t + 1], prob.A, cache.u[t])
    end
    return nothing
end

# =============================================================================
# Wrapper functions for Enzyme AD
# =============================================================================

# Forward: constructs problem, solves, returns mutated cache (not nothing).
# Rule: a function that mutates an argument must return that argument.
function minimal_solve_wrapper!(A, u0, cache)
    prob = MinimalProblem(A, u0)
    zero_minimal_cache!(cache)
    minimal_solve!(prob, cache)
    return cache
end

# Scalar: constructs problem, solves, returns scalar for reverse mode.
function minimal_loss(A, u0, cache)::Float64
    prob = MinimalProblem(A, u0)
    zero_minimal_cache!(cache)
    minimal_solve!(prob, cache)
    return sum(cache.u[end])
end

# =============================================================================
# Tests
# =============================================================================

@testset "Minimal sensitivity interface - sanity" begin
    A = [0.8 0.1; -0.1 0.7]
    u0 = [1.0, 0.5]
    cache = alloc_minimal_cache(u0, 4)

    minimal_solve_wrapper!(A, u0, cache)
    @test cache.u[1] ≈ u0
    @test cache.u[4] ≈ A^3 * u0

    loglik = minimal_loss(A, u0, cache)
    @test loglik ≈ sum(A^3 * u0)
end

@testset "Minimal sensitivity - forward (in-place, validates cache tangents)" begin
    A = [0.8 0.1; -0.1 0.7]
    u0 = [1.0, 0.5]

    test_forward(
        minimal_solve_wrapper!, Const,
        (copy(A), Duplicated),
        (copy(u0), Duplicated),
        (alloc_minimal_cache(u0, 4), Duplicated)
    )
end

@testset "Minimal sensitivity - reverse (scalar loglik)" begin
    A = [0.8 0.1; -0.1 0.7]
    u0 = [1.0, 0.5]

    test_reverse(
        minimal_loss, Active,
        (copy(A), Duplicated),
        (copy(u0), Duplicated),
        (alloc_minimal_cache(u0, 4), Duplicated)
    )
end
