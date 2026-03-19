using LinearAlgebra, Test, Enzyme, EnzymeTestUtils, StaticArrays, Random
using DifferenceEquations
using DifferenceEquations: _direct_iteration_loglik!, alloc_direct_loglik_cache,
    zero_direct_loglik_cache!!

# =============================================================================
# Test setup
# =============================================================================

const N_di = 3  # State dimension
const M_di = 2  # Observation dimension
const K_di = 2  # State noise dimension
const L_di = 2  # Observation noise dimension
const T_di = 5  # Number of observation time steps

Random.seed!(42)
A_raw = randn(N_di, N_di)
const A_di = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
const B_di = 0.1 * randn(N_di, K_di)
const C_di = randn(M_di, N_di)
const H_di = 0.1 * randn(M_di, L_di)

const u0_di = zeros(N_di)

# Generate synthetic observations with noise
Random.seed!(123)
const noise_di = [randn(K_di) for _ in 1:T_di]

# Simulate to get observations
x_sim = [zeros(N_di) for _ in 1:(T_di + 1)]
z_sim = [zeros(M_di) for _ in 1:(T_di + 1)]
x_sim[1] = copy(u0_di)
z_sim[1] = C_di * x_sim[1]
for t in 2:(T_di + 1)
    x_sim[t] = A_di * x_sim[t - 1] + B_di * noise_di[t - 1]
    z_sim[t] = C_di * x_sim[t]
end
const y_di = [z_sim[t + 1] + H_di * randn(L_di) for t in 1:T_di]

const T_total_di = T_di + 1

function make_di_cache(A, B, C, H, u0, T_total)
    return alloc_direct_loglik_cache(u0, A, B, C, H, T_total)
end

# =============================================================================
# Scalar wrapper for Enzyme AD testing
# =============================================================================

@inline function scalar_di_loglik!(A, B, C, u0, noise, observables, H, cache)
    zero_direct_loglik_cache!!(cache)
    return _direct_iteration_loglik!(A, B, C, u0, noise, observables, H, cache)
end

# =============================================================================
# Basic functionality test
# =============================================================================

@testset "DirectIteration loglik - basic sanity" begin
    cache = make_di_cache(A_di, B_di, C_di, H_di, u0_di, T_total_di)
    zero_direct_loglik_cache!!(cache)
    loglik = _direct_iteration_loglik!(A_di, B_di, C_di, u0_di, noise_di, y_di, H_di, cache)

    @test isfinite(loglik)
end

# =============================================================================
# Mutable arrays - EnzymeTestUtils validation
# =============================================================================

@testset "EnzymeTestUtils - DirectIteration mutable (model Const)" begin
    cache = make_di_cache(A_di, B_di, C_di, H_di, u0_di, T_total_di)

    test_forward(scalar_di_loglik!, Const,
        (copy(A_di), Const),
        (copy(B_di), Const),
        (copy(C_di), Const),
        (copy(u0_di), Duplicated),
        ([copy(n) for n in noise_di], Duplicated),
        ([copy(y) for y in y_di], Duplicated),
        (copy(H_di), Const),
        (cache, Duplicated))

    test_reverse(scalar_di_loglik!, Const,
        (copy(A_di), Const),
        (copy(B_di), Const),
        (copy(C_di), Const),
        (copy(u0_di), Duplicated),
        ([copy(n) for n in noise_di], Duplicated),
        ([copy(y) for y in y_di], Duplicated),
        (copy(H_di), Const),
        (make_di_cache(A_di, B_di, C_di, H_di, u0_di, T_total_di), Duplicated))
end

@testset "EnzymeTestUtils - DirectIteration mutable (model Duplicated)" begin
    N_small, M_small, K_small, L_small, T_small = 2, 2, 2, 2, 2

    A_small = [0.8 0.1; -0.1 0.7]
    B_small = [0.1 0.0; 0.0 0.1]
    C_small = [1.0 0.0; 0.0 1.0]
    H_small = [0.1 0.0; 0.0 0.1]

    u0_small = zeros(N_small)
    noise_small = [[0.1, -0.1], [0.2, 0.05]]
    y_small = [[0.5, 0.3], [0.2, 0.1]]

    cache = make_di_cache(A_small, B_small, C_small, H_small, u0_small, T_small + 1)

    test_forward(scalar_di_loglik!, Const,
        (copy(A_small), Duplicated),
        (copy(B_small), Duplicated),
        (copy(C_small), Duplicated),
        (copy(u0_small), Const),
        ([copy(n) for n in noise_small], Duplicated),
        ([copy(y) for y in y_small], Duplicated),
        (copy(H_small), Const),
        (cache, Duplicated))

    test_reverse(scalar_di_loglik!, Const,
        (copy(A_small), Duplicated),
        (copy(B_small), Duplicated),
        (copy(C_small), Duplicated),
        (copy(u0_small), Const),
        ([copy(n) for n in noise_small], Duplicated),
        ([copy(y) for y in y_small], Duplicated),
        (copy(H_small), Const),
        (make_di_cache(A_small, B_small, C_small, H_small, u0_small, T_small + 1),
            Duplicated))
end

# =============================================================================
# Static arrays - EnzymeTestUtils validation
# =============================================================================

@testset "EnzymeTestUtils - DirectIteration static (model Const)" begin
    A_static = SMatrix{N_di, N_di}(A_di)
    B_static = SMatrix{N_di, K_di}(B_di)
    C_static = SMatrix{M_di, N_di}(C_di)
    H_static = SMatrix{M_di, L_di}(H_di)
    u0_static = SVector{N_di}(u0_di)
    noise_static = [SVector{K_di}(noise_di[t]) for t in 1:T_di]
    y_static = [SVector{M_di}(y_di[t]) for t in 1:T_di]

    cache = make_di_cache(A_static, B_static, C_static, H_static, u0_static, T_total_di)

    test_forward(scalar_di_loglik!, Const,
        (A_static, Const),
        (B_static, Const),
        (C_static, Const),
        (u0_static, Duplicated),
        (noise_static, Duplicated),
        (y_static, Duplicated),
        (H_static, Const),
        (cache, Duplicated))

    # Note: Reverse mode with StaticArrays has known gradient accumulation issues
    # in Enzyme. Forward mode validates correctness; mutable reverse passes.
end

@testset "EnzymeTestUtils - DirectIteration static (model Duplicated)" begin
    N_small, M_small, K_small, L_small, T_small = 2, 2, 2, 2, 2

    A_static = SMatrix{N_small, N_small}([0.8 0.1; -0.1 0.7])
    B_static = SMatrix{N_small, K_small}([0.1 0.0; 0.0 0.1])
    C_static = SMatrix{M_small, N_small}([1.0 0.0; 0.0 1.0])
    H_static = SMatrix{M_small, L_small}([0.1 0.0; 0.0 0.1])

    u0_static = SVector{N_small}(zeros(N_small))
    noise_static = [SVector{K_small}([0.1, -0.1]), SVector{K_small}([0.2, 0.05])]
    y_static = [SVector{M_small}([0.5, 0.3]), SVector{M_small}([0.2, 0.1])]

    cache = make_di_cache(A_static, B_static, C_static, H_static, u0_static, T_small + 1)

    test_forward(scalar_di_loglik!, Const,
        (A_static, Duplicated),
        (B_static, Duplicated),
        (C_static, Duplicated),
        (u0_static, Const),
        (noise_static, Duplicated),
        (y_static, Duplicated),
        (H_static, Const),
        (cache, Duplicated))
end

# =============================================================================
# Regression test
# =============================================================================

@testset "DirectIteration loglik - regression test" begin
    A_reg = [0.9 0.1; -0.1 0.9]
    B_reg = [0.1 0.0; 0.0 0.1]
    C_reg = [1.0 0.0; 0.0 1.0]
    H_reg = [0.1 0.0; 0.0 0.1]

    u0_reg = [0.0, 0.0]
    noise_reg = [[0.1, -0.1], [0.2, 0.05], [0.0, 0.1]]
    y_reg = [[0.5, -0.3], [0.8, -0.1], [0.6, 0.2]]

    cache = make_di_cache(A_reg, B_reg, C_reg, H_reg, u0_reg, 4)
    zero_direct_loglik_cache!!(cache)
    loglik = _direct_iteration_loglik!(A_reg, B_reg, C_reg, u0_reg, noise_reg, y_reg,
        H_reg, cache)

    @test isfinite(loglik)
    @test loglik < 0

    # Verify it's consistent across calls (caller zeros cache)
    zero_direct_loglik_cache!!(cache)
    loglik2 = _direct_iteration_loglik!(A_reg, B_reg, C_reg, u0_reg, noise_reg, y_reg,
        H_reg, cache)
    @test loglik ≈ loglik2 rtol = 1e-12
end
