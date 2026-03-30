# Enzyme AD tests for KalmanFilter
# prob passed as Duplicated — observables get zero shadow automatically.
# GC disabled to avoid Enzyme reverse-mode GC corruption (#2355).

GC.gc()
GC.enable(false)

using LinearAlgebra, Test, Enzyme, EnzymeTestUtils, StaticArrays, Random
using DifferenceEquations
using DifferenceEquations: init, solve!, StateSpaceWorkspace
using FiniteDifferences: central_fdm

include("enzyme_test_utils.jl")  # vech helpers only

# max_range needed: FD perturbation of observables_noise inside prob can push
# the matrix non-positive-definite, causing DomainError in logdet_chol.
const _fdm_kf = central_fdm(5, 1; max_range = 1.0e-3)

# --- Test setup ---

const N_kf = 3
const M_kf = 2
const K_kf = 2
const L_kf = 2
const T_kf = 5

Random.seed!(42)
A_raw = randn(N_kf, N_kf)
const A_kf = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
const B_kf = 0.1 * randn(N_kf, K_kf)
const C_kf = randn(M_kf, N_kf)
const H_kf = 0.1 * randn(M_kf, L_kf)
const R_kf = H_kf * H_kf'
const mu_0_kf = zeros(N_kf)
const Sigma_0_kf = Matrix{Float64}(I, N_kf, N_kf)

Random.seed!(123)
const x0_kf = mu_0_kf + cholesky(Sigma_0_kf).L * randn(N_kf)
const noise_kf = [randn(K_kf) for _ in 1:T_kf]
const obs_noise_kf = [randn(L_kf) for _ in 1:T_kf]
const sim_sol_kf = solve(
    LinearStateSpaceProblem(
        A_kf, B_kf, x0_kf, (0, T_kf); C = C_kf, noise = noise_kf
    )
)
const y_kf = [sim_sol_kf.z[t + 1] + H_kf * obs_noise_kf[t] for t in 1:T_kf]

# --- Helpers ---

function make_kalman_prob(A, B, C, R, mu_0, Sigma_0, y)
    return LinearStateSpaceProblem(
        A, B, zeros(eltype(A), size(A, 1)), (0, length(y));
        C, u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y
    )
end

# --- Wrappers — prob as single Duplicated arg ---

function kalman_solve_prob!(prob, sol, cache)
    ws = StateSpaceWorkspace(prob, KalmanFilter(), sol, cache)
    solve!(ws)
    return (sol.u, sol.P, sol.z)
end

function kalman_loglik_prob(prob, sol, cache)::Float64
    ws = StateSpaceWorkspace(prob, KalmanFilter(), sol, cache)
    return solve!(ws).logpdf
end

# Vech: separate args (y stays Duplicated — remake doesn't work with Enzyme shadows)
function kalman_solve_vech!(
        A, B, C, mu_0, sigma_0_vech, r_vech, y, sol, cache,
        n_state, n_obs
    )
    Sigma_0 = make_posdef_from_vech(sigma_0_vech, n_state)
    R = make_posdef_from_vech(r_vech, n_obs)
    prob = LinearStateSpaceProblem(
        A, B, zeros(eltype(A), size(A, 1)), (0, length(y));
        C, u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y
    )
    ws = StateSpaceWorkspace(prob, KalmanFilter(), sol, cache)
    solve!(ws)
    return (sol.u, sol.P, sol.z)
end

function kalman_loglik_vech(
        A, B, C, mu_0, sigma_0_vech, r_vech, y, sol, cache,
        n_state, n_obs
    )::Float64
    Sigma_0 = make_posdef_from_vech(sigma_0_vech, n_state)
    R = make_posdef_from_vech(r_vech, n_obs)
    prob = LinearStateSpaceProblem(
        A, B, zeros(eltype(A), size(A, 1)), (0, length(y));
        C, u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y
    )
    ws = StateSpaceWorkspace(prob, KalmanFilter(), sol, cache)
    return solve!(ws).logpdf
end

# --- Basic sanity test ---

@testset "Kalman loglik via solve!() - sanity" begin
    prob = make_kalman_prob(A_kf, B_kf, C_kf, R_kf, mu_0_kf, Sigma_0_kf, y_kf)
    ws = init(prob, KalmanFilter())
    loglik = kalman_loglik_prob(prob, ws.output, ws.cache)
    @test isfinite(loglik)
    @test loglik < 0

    loglik2 = kalman_loglik_prob(prob, ws.output, ws.cache)
    @test loglik ≈ loglik2 rtol = 1.0e-12
end

# --- Forward — prob as Duplicated (small model, N=M=K=L=2, T=2) ---

@testset "EnzymeTestUtils - Kalman forward (prob Duplicated)" begin
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]; R_s = [0.01 0.0; 0.0 0.01]
    mu_0_s = zeros(2); Sigma_0_s = Matrix{Float64}(I, 2, 2)
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    prob = make_kalman_prob(A_s, B_s, C_s, R_s, mu_0_s, Sigma_0_s, y_s)
    ws = init(prob, KalmanFilter())

    test_forward(
        kalman_solve_prob!, Const,
        (prob, Duplicated),
        (ws.output, Duplicated), (ws.cache, Duplicated);
        fdm = _fdm_kf,
    )
end

# --- Reverse via vech (all Duplicated) ---

@testset "EnzymeTestUtils - Kalman reverse via vech (all Duplicated)" begin
    _fdm_vech = central_fdm(5, 1)
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]; R_s = [0.01 0.0; 0.0 0.01]
    mu_0_s = zeros(2); Sigma_0_s = Matrix{Float64}(I, 2, 2)
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    sigma_0_v = make_vech_for(Sigma_0_s)
    r_v = make_vech_for(R_s)
    prob = make_kalman_prob(A_s, B_s, C_s, R_s, mu_0_s, Sigma_0_s, y_s)
    ws = init(prob, KalmanFilter())

    test_reverse(
        kalman_loglik_vech, Active,
        (copy(A_s), Duplicated), (copy(B_s), Duplicated),
        (copy(C_s), Duplicated), (copy(mu_0_s), Duplicated),
        (copy(sigma_0_v), Duplicated), (copy(r_v), Duplicated),
        ([copy(y) for y in y_s], Duplicated),
        (deepcopy(ws.output), Duplicated), (deepcopy(ws.cache), Duplicated),
        (2, Const), (2, Const);
        fdm = _fdm_vech,
    )
end

# --- Forward — rectangular B (N!=K) — validates mul_aat!! workaround ---

@testset "EnzymeTestUtils - Kalman rectangular B forward (prob Duplicated)" begin
    A_r = [
        0.3 0.1 0.0 0.05 0.02; -0.1 0.3 0.05 0.0 0.01;
        0.02 -0.05 0.3 0.1 0.0; 0.0 0.02 -0.1 0.3 0.05;
        0.01 0.0 0.02 -0.05 0.3
    ]
    B_r = 0.1 * [1.0 0.5; 0.3 -0.2; 0.7 0.1; -0.4 0.6; 0.2 -0.3]
    C_r = [1.0 0.0 0.5 0.0 0.0; 0.0 1.0 0.0 0.5 0.0; 0.0 0.0 1.0 0.0 0.5]
    R_r = 0.01 * Matrix{Float64}(I, 3, 3)
    mu_0_r = zeros(5); Sigma_0_r = Matrix{Float64}(I, 5, 5)
    y_r = [[0.5, 0.3, 0.1], [0.2, -0.1, 0.4], [0.8, 0.4, -0.2]]
    prob = make_kalman_prob(A_r, B_r, C_r, R_r, mu_0_r, Sigma_0_r, y_r)
    ws = init(prob, KalmanFilter())

    test_forward(
        kalman_solve_prob!, Const,
        (prob, Duplicated),
        (ws.output, Duplicated), (ws.cache, Duplicated);
        fdm = _fdm_kf,
    )
end

# --- Reverse — rectangular B via vech ---

@testset "EnzymeTestUtils - Kalman rectangular B reverse via vech (all Duplicated)" begin
    _fdm_vech = central_fdm(5, 1)
    N_r, M_r = 5, 3
    A_r = [
        0.3 0.1 0.0 0.05 0.02; -0.1 0.3 0.05 0.0 0.01;
        0.02 -0.05 0.3 0.1 0.0; 0.0 0.02 -0.1 0.3 0.05;
        0.01 0.0 0.02 -0.05 0.3
    ]
    B_r = 0.1 * [1.0 0.5; 0.3 -0.2; 0.7 0.1; -0.4 0.6; 0.2 -0.3]
    C_r = [1.0 0.0 0.5 0.0 0.0; 0.0 1.0 0.0 0.5 0.0; 0.0 0.0 1.0 0.0 0.5]
    R_r = 0.01 * Matrix{Float64}(I, M_r, M_r)
    mu_0_r = zeros(N_r); Sigma_0_r = Matrix{Float64}(I, N_r, N_r)
    y_r = [[0.5, 0.3, 0.1], [0.2, -0.1, 0.4], [0.8, 0.4, -0.2]]
    sigma_0_v = make_vech_for(Sigma_0_r)
    r_v = make_vech_for(R_r)
    prob = make_kalman_prob(A_r, B_r, C_r, R_r, mu_0_r, Sigma_0_r, y_r)
    ws = init(prob, KalmanFilter())

    test_reverse(
        kalman_loglik_vech, Active,
        (copy(A_r), Duplicated), (copy(B_r), Duplicated),
        (copy(C_r), Duplicated), (copy(mu_0_r), Duplicated),
        (copy(sigma_0_v), Duplicated), (copy(r_v), Duplicated),
        ([copy(y) for y in y_r], Duplicated),
        (deepcopy(ws.output), Duplicated), (deepcopy(ws.cache), Duplicated),
        (N_r, Const), (M_r, Const);
        fdm = _fdm_vech,
    )
end

# --- Non-diagonal R via vech (genuinely off-diagonal) ---

@testset "EnzymeTestUtils - Kalman non-diagonal R forward (vech)" begin
    _fdm_vech = central_fdm(5, 1)
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]
    R_offdiag = [0.02 0.005; 0.005 0.01]
    r_v = make_vech_for(R_offdiag)
    mu_0_s = zeros(2); Sigma_0_s = Matrix{Float64}(I, 2, 2)
    sigma_0_v = make_vech_for(Sigma_0_s)
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    prob = make_kalman_prob(A_s, B_s, C_s, R_offdiag, mu_0_s, Sigma_0_s, y_s)
    ws = init(prob, KalmanFilter())

    test_forward(
        kalman_solve_vech!, Const,
        (copy(A_s), Duplicated), (copy(B_s), Duplicated),
        (copy(C_s), Duplicated), (copy(mu_0_s), Duplicated),
        (copy(sigma_0_v), Duplicated), (copy(r_v), Duplicated),
        ([copy(y) for y in y_s], Duplicated),
        (ws.output, Duplicated), (ws.cache, Duplicated),
        (2, Const), (2, Const);
        fdm = _fdm_vech,
    )
end

@testset "EnzymeTestUtils - Kalman non-diagonal R reverse (vech)" begin
    _fdm_vech = central_fdm(5, 1)
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]
    R_offdiag = [0.02 0.005; 0.005 0.01]
    r_v = make_vech_for(R_offdiag)
    mu_0_s = zeros(2); Sigma_0_s = Matrix{Float64}(I, 2, 2)
    sigma_0_v = make_vech_for(Sigma_0_s)
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    prob = make_kalman_prob(A_s, B_s, C_s, R_offdiag, mu_0_s, Sigma_0_s, y_s)
    ws = init(prob, KalmanFilter())

    test_reverse(
        kalman_loglik_vech, Active,
        (copy(A_s), Duplicated), (copy(B_s), Duplicated),
        (copy(C_s), Duplicated), (copy(mu_0_s), Duplicated),
        (copy(sigma_0_v), Duplicated), (copy(r_v), Duplicated),
        ([copy(y) for y in y_s], Duplicated),
        (deepcopy(ws.output), Duplicated), (deepcopy(ws.cache), Duplicated),
        (2, Const), (2, Const);
        fdm = _fdm_vech,
    )
end

# --- Regression test ---

@testset "Kalman loglik - regression test" begin
    A_reg = [0.9 0.1; -0.1 0.9]; B_reg = [0.1 0.0; 0.0 0.1]
    C_reg = [1.0 0.0; 0.0 1.0]; R_reg = [0.01 0.0; 0.0 0.01]
    mu_0_reg = [0.0, 0.0]; Sigma_0_reg = [1.0 0.0; 0.0 1.0]
    y_reg = [[0.5, -0.3], [0.8, -0.1], [0.6, 0.2]]
    prob = make_kalman_prob(A_reg, B_reg, C_reg, R_reg, mu_0_reg, Sigma_0_reg, y_reg)
    ws = init(prob, KalmanFilter())

    loglik = kalman_loglik_prob(prob, ws.output, ws.cache)
    @test isfinite(loglik)
    @test loglik < 0
    @test length(ws.output.u) == 4
end

GC.enable(true)
