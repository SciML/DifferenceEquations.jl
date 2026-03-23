using LinearAlgebra, Test, Enzyme, EnzymeTestUtils, StaticArrays, Random
using DifferenceEquations
using DifferenceEquations: init, solve!, StateSpaceWorkspace

include("enzyme_test_utils.jl")

# =============================================================================
# Test setup — generate observations using the package's own solve()
# =============================================================================

const N_kf = 3  # State dimension
const M_kf = 2  # Observation dimension
const K_kf = 2  # State noise dimension
const L_kf = 2  # Observation noise dimension
const T_kf = 5  # Number of observation time steps

Random.seed!(42)
A_raw = randn(N_kf, N_kf)
const A_kf = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
const B_kf = 0.1 * randn(N_kf, K_kf)
const C_kf = randn(M_kf, N_kf)
const H_kf = 0.1 * randn(M_kf, L_kf)
const R_kf = H_kf * H_kf'

const mu_0_kf = zeros(N_kf)
const Sigma_0_kf = Matrix{Float64}(I, N_kf, N_kf)

# Generate observations using package's solve() + manual observation noise
Random.seed!(123)
const x0_kf = mu_0_kf + cholesky(Sigma_0_kf).L * randn(N_kf)
const noise_kf = [randn(K_kf) for _ in 1:T_kf]
const obs_noise_kf = [randn(L_kf) for _ in 1:T_kf]

const sim_sol_kf = solve(LinearStateSpaceProblem(
    A_kf, B_kf, x0_kf, (0, T_kf); C = C_kf, noise = noise_kf))
const y_kf = [sim_sol_kf.z[t + 1] + H_kf * obs_noise_kf[t] for t in 1:T_kf]

# =============================================================================
# Helpers
# =============================================================================

function make_kalman_prob(A, B, C, R, mu_0, Sigma_0, y)
    return LinearStateSpaceProblem(
        A, B, zeros(eltype(A), size(A, 1)), (0, length(y));
        C, u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y)
end

function make_kalman_cache(A, B, C, R, mu_0, Sigma_0, y)
    prob = make_kalman_prob(A, B, C, R, mu_0, Sigma_0, y)
    return init(prob, KalmanFilter()).cache
end

# =============================================================================
# Wrapper functions for Enzyme AD
# All array arguments MUST be Duplicated (no Const) — Enzyme can't handle
# mixed Const/Duplicated activity in struct fields.
# =============================================================================

# Forward: returns solution struct (validates tangents of u, P, z, logpdf)
function kalman_solve!(A, B, C, mu_0, Sigma_0, R, y, cache)
    prob = make_kalman_prob(A, B, C, R, mu_0, Sigma_0, y)
    ws = StateSpaceWorkspace(prob, KalmanFilter(), cache)
    return solve!(ws)
end

# Scalar: validates gradient of logpdf
function kalman_loglik(A, B, C, mu_0, Sigma_0, R, y, cache)
    prob = make_kalman_prob(A, B, C, R, mu_0, Sigma_0, y)
    ws = StateSpaceWorkspace(prob, KalmanFilter(), cache)
    return solve!(ws).logpdf
end

# Scalar with vech parameterization for posdef Sigma_0 and R
function kalman_loglik_vech(A, B, C, mu_0, sigma_0_vech, r_vech, y, cache,
        n_state, n_obs)
    Sigma_0 = make_posdef_from_vech(sigma_0_vech, n_state)
    R = make_posdef_from_vech(r_vech, n_obs)
    return kalman_loglik(A, B, C, mu_0, Sigma_0, R, y, cache)
end

# =============================================================================
# Basic sanity test
# =============================================================================

@testset "Kalman loglik via solve!() - sanity" begin
    cache = make_kalman_cache(A_kf, B_kf, C_kf, R_kf, mu_0_kf, Sigma_0_kf, y_kf)
    loglik = kalman_loglik(A_kf, B_kf, C_kf, mu_0_kf, Sigma_0_kf, R_kf, y_kf, cache)
    @test isfinite(loglik)
    @test loglik < 0

    # Verify consistency: calling twice gives same result (cache reuse)
    loglik2 = kalman_loglik(A_kf, B_kf, C_kf, mu_0_kf, Sigma_0_kf, R_kf, y_kf, cache)
    @test loglik ≈ loglik2 rtol = 1e-12
end

# =============================================================================
# Mutable arrays — all Duplicated (small model, N=M=K=L=2, T=2)
# =============================================================================

@testset "EnzymeTestUtils - Kalman forward (in-place, all Duplicated)" begin
    A_s = [0.8 0.1; -0.1 0.7]
    B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]
    R_s = [0.01 0.0; 0.0 0.01]
    mu_0_s = zeros(2)
    Sigma_0_s = Matrix{Float64}(I, 2, 2)
    y_s = [[0.5, 0.3], [0.2, 0.1]]

    test_forward(kalman_solve!, Const,
        (copy(A_s), Duplicated),
        (copy(B_s), Duplicated),
        (copy(C_s), Duplicated),
        (copy(mu_0_s), Duplicated),
        (copy(Sigma_0_s), Duplicated),
        (copy(R_s), Duplicated),
        ([copy(y) for y in y_s], Duplicated),
        (make_kalman_cache(A_s, B_s, C_s, R_s, mu_0_s, Sigma_0_s, y_s), Duplicated))
end

@testset "EnzymeTestUtils - Kalman reverse (scalar logpdf via vech, all Duplicated)" begin
    # Reverse mode uses vech parameterization for Sigma_0 and R to guarantee
    # positive-definiteness under finite difference perturbations.
    # Direct R/Sigma_0 perturbations can violate posdef, causing DomainError in log(det(S)).
    A_s = [0.8 0.1; -0.1 0.7]
    B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]
    R_s = [0.01 0.0; 0.0 0.01]
    mu_0_s = zeros(2)
    Sigma_0_s = Matrix{Float64}(I, 2, 2)
    y_s = [[0.5, 0.3], [0.2, 0.1]]

    sigma_0_v = make_vech_for(Sigma_0_s)
    r_v = make_vech_for(R_s)

    test_reverse(kalman_loglik_vech, Active,
        (copy(A_s), Duplicated),
        (copy(B_s), Duplicated),
        (copy(C_s), Duplicated),
        (copy(mu_0_s), Duplicated),
        (copy(sigma_0_v), Duplicated),
        (copy(r_v), Duplicated),
        ([copy(y) for y in y_s], Duplicated),
        (make_kalman_cache(A_s, B_s, C_s, R_s, mu_0_s, Sigma_0_s, y_s), Duplicated),
        (2, Const),
        (2, Const))
end

# =============================================================================
# Vech parameterization for posdef Sigma_0 and R (small model)
# =============================================================================

@testset "EnzymeTestUtils - Kalman reverse with vech (all Duplicated)" begin
    A_s = [0.8 0.1; -0.1 0.7]
    B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]
    R_s = [0.01 0.0; 0.0 0.01]
    mu_0_s = zeros(2)
    Sigma_0_s = Matrix{Float64}(I, 2, 2)
    y_s = [[0.5, 0.3], [0.2, 0.1]]

    sigma_0_v = make_vech_for(Sigma_0_s)
    r_v = make_vech_for(R_s)

    test_reverse(kalman_loglik_vech, Active,
        (copy(A_s), Duplicated),
        (copy(B_s), Duplicated),
        (copy(C_s), Duplicated),
        (copy(mu_0_s), Duplicated),
        (copy(sigma_0_v), Duplicated),
        (copy(r_v), Duplicated),
        ([copy(y) for y in y_s], Duplicated),
        (make_kalman_cache(A_s, B_s, C_s, R_s, mu_0_s, Sigma_0_s, y_s), Duplicated),
        (2, Const),
        (2, Const))
end

# =============================================================================
# Rectangular B (N≠K) — validates mul_aat!! workaround for Enzyme syrk bug
# https://github.com/EnzymeAD/Enzyme.jl/issues/2355
# =============================================================================

@testset "EnzymeTestUtils - Kalman rectangular B forward (all Duplicated)" begin
    A_rect = [0.3 0.1 0.0 0.05 0.02;
              -0.1 0.3 0.05 0.0 0.01;
              0.02 -0.05 0.3 0.1 0.0;
              0.0 0.02 -0.1 0.3 0.05;
              0.01 0.0 0.02 -0.05 0.3]
    B_rect = 0.1 * [1.0 0.5; 0.3 -0.2; 0.7 0.1; -0.4 0.6; 0.2 -0.3]
    C_rect = [1.0 0.0 0.5 0.0 0.0; 0.0 1.0 0.0 0.5 0.0; 0.0 0.0 1.0 0.0 0.5]
    R_rect = 0.01 * Matrix{Float64}(I, 3, 3)
    mu_0_rect = zeros(5)
    Sigma_0_rect = Matrix{Float64}(I, 5, 5)
    y_rect = [[0.5, 0.3, 0.1], [0.2, -0.1, 0.4], [0.8, 0.4, -0.2]]

    test_forward(kalman_solve!, Const,
        (copy(A_rect), Duplicated),
        (copy(B_rect), Duplicated),
        (copy(C_rect), Duplicated),
        (copy(mu_0_rect), Duplicated),
        (copy(Sigma_0_rect), Duplicated),
        (copy(R_rect), Duplicated),
        ([copy(y) for y in y_rect], Duplicated),
        (make_kalman_cache(A_rect, B_rect, C_rect, R_rect, mu_0_rect, Sigma_0_rect,
            y_rect), Duplicated))
end

@testset "EnzymeTestUtils - Kalman rectangular B reverse via vech (all Duplicated)" begin
    N_rect, M_rect = 5, 3
    A_rect = [0.3 0.1 0.0 0.05 0.02;
              -0.1 0.3 0.05 0.0 0.01;
              0.02 -0.05 0.3 0.1 0.0;
              0.0 0.02 -0.1 0.3 0.05;
              0.01 0.0 0.02 -0.05 0.3]
    B_rect = 0.1 * [1.0 0.5; 0.3 -0.2; 0.7 0.1; -0.4 0.6; 0.2 -0.3]
    C_rect = [1.0 0.0 0.5 0.0 0.0; 0.0 1.0 0.0 0.5 0.0; 0.0 0.0 1.0 0.0 0.5]
    R_rect = 0.01 * Matrix{Float64}(I, M_rect, M_rect)
    mu_0_rect = zeros(N_rect)
    Sigma_0_rect = Matrix{Float64}(I, N_rect, N_rect)
    y_rect = [[0.5, 0.3, 0.1], [0.2, -0.1, 0.4], [0.8, 0.4, -0.2]]

    sigma_0_v = make_vech_for(Sigma_0_rect)
    r_v = make_vech_for(R_rect)

    test_reverse(kalman_loglik_vech, Active,
        (copy(A_rect), Duplicated),
        (copy(B_rect), Duplicated),
        (copy(C_rect), Duplicated),
        (copy(mu_0_rect), Duplicated),
        (copy(sigma_0_v), Duplicated),
        (copy(r_v), Duplicated),
        ([copy(y) for y in y_rect], Duplicated),
        (make_kalman_cache(A_rect, B_rect, C_rect, R_rect, mu_0_rect, Sigma_0_rect,
            y_rect), Duplicated),
        (N_rect, Const),
        (M_rect, Const))
end

# =============================================================================
# Regression test
# =============================================================================

@testset "Kalman loglik - regression test" begin
    A_reg = [0.9 0.1; -0.1 0.9]
    B_reg = [0.1 0.0; 0.0 0.1]
    C_reg = [1.0 0.0; 0.0 1.0]
    R_reg = [0.01 0.0; 0.0 0.01]
    mu_0_reg = [0.0, 0.0]
    Sigma_0_reg = [1.0 0.0; 0.0 1.0]
    y_reg = [[0.5, -0.3], [0.8, -0.1], [0.6, 0.2]]

    cache = make_kalman_cache(A_reg, B_reg, C_reg, R_reg, mu_0_reg, Sigma_0_reg, y_reg)
    loglik = kalman_loglik(A_reg, B_reg, C_reg, mu_0_reg, Sigma_0_reg, R_reg, y_reg, cache)

    @test isfinite(loglik)
    @test loglik < 0
    @test length(cache.u) == 4  # T+1 time points
end
