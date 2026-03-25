# ForwardDiff AD tests for Kalman filter
# Tests gradient correctness against central finite differences.

using LinearAlgebra, Test, ForwardDiff, StaticArrays, Random
using DifferenceEquations

include("forwarddiff_test_utils.jl")

# =============================================================================
# Problem setup
# =============================================================================

const N_kf_fd = 3
const M_kf_fd = 2
const K_kf_fd = 2
const T_kf_fd = 5

Random.seed!(42)
A_raw_kf_fd = randn(N_kf_fd, N_kf_fd)
const A_kf_fd = 0.5 * A_raw_kf_fd / maximum(abs.(eigvals(A_raw_kf_fd)))
const B_kf_fd = 0.1 * randn(N_kf_fd, K_kf_fd)
const C_kf_fd = randn(M_kf_fd, N_kf_fd)
const H_kf_fd = 0.1 * randn(M_kf_fd, M_kf_fd)
const R_kf_fd = H_kf_fd * H_kf_fd' + 0.01 * I
const mu_0_kf_fd = zeros(N_kf_fd)
const Sigma_0_kf_fd = Matrix{Float64}(I, N_kf_fd, N_kf_fd)

Random.seed!(123)
const x0_kf_fd = randn(N_kf_fd)
const noise_sim_kf_fd = [randn(K_kf_fd) for _ in 1:T_kf_fd]
const sim_sol_kf_fd = solve(LinearStateSpaceProblem(
    A_kf_fd, B_kf_fd, x0_kf_fd, (0, T_kf_fd); C = C_kf_fd, noise = noise_sim_kf_fd))
const y_kf_fd = [sim_sol_kf_fd.z[t + 1] + H_kf_fd * randn(M_kf_fd) for t in 1:T_kf_fd]

# =============================================================================
# Mutable arrays — ForwardDiff gradient tests
# =============================================================================

function kalman_loglik_fd(A, B, C, mu_0, Sigma_0, R, y)
    T_el = promote_type(eltype(A), eltype(B), eltype(C),
        eltype(mu_0), eltype(Sigma_0), eltype(R))
    prob = LinearStateSpaceProblem(
        promote_array(T_el, A), promote_array(T_el, B),
        zeros(T_el, size(A, 1)), (0, length(y));
        C = promote_array(T_el, C),
        u0_prior_mean = promote_array(T_el, mu_0),
        u0_prior_var = promote_array(T_el, Sigma_0),
        observables_noise = promote_array(T_el, R),
        observables = y)
    sol = solve(prob, KalmanFilter())
    return sol.logpdf
end

@testset "ForwardDiff - Kalman Filter (mutable)" begin
    @testset "primal sanity" begin
        loglik_val = kalman_loglik_fd(A_kf_fd, B_kf_fd, C_kf_fd,
            mu_0_kf_fd, Sigma_0_kf_fd, R_kf_fd, y_kf_fd)
        @test isfinite(loglik_val)
        @test loglik_val < 0
    end

    @testset "gradient w.r.t. A" begin
        f = a_vec -> kalman_loglik_fd(reshape(a_vec, N_kf_fd, N_kf_fd),
            B_kf_fd, C_kf_fd, mu_0_kf_fd, Sigma_0_kf_fd, R_kf_fd, y_kf_fd)
        x0 = vec(copy(A_kf_fd))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end

    @testset "gradient w.r.t. B" begin
        f = b_vec -> kalman_loglik_fd(A_kf_fd, reshape(b_vec, N_kf_fd, K_kf_fd),
            C_kf_fd, mu_0_kf_fd, Sigma_0_kf_fd, R_kf_fd, y_kf_fd)
        x0 = vec(copy(B_kf_fd))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end

    @testset "gradient w.r.t. C" begin
        f = c_vec -> kalman_loglik_fd(A_kf_fd, B_kf_fd,
            reshape(c_vec, M_kf_fd, N_kf_fd), mu_0_kf_fd, Sigma_0_kf_fd, R_kf_fd, y_kf_fd)
        x0 = vec(copy(C_kf_fd))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end

    @testset "gradient w.r.t. mu_0" begin
        f = m_vec -> kalman_loglik_fd(A_kf_fd, B_kf_fd, C_kf_fd,
            m_vec, Sigma_0_kf_fd, R_kf_fd, y_kf_fd)
        x0 = copy(mu_0_kf_fd)
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end
end

# =============================================================================
# StaticArrays — ForwardDiff gradient tests
# =============================================================================

const A_kf_fd_s = SMatrix{N_kf_fd, N_kf_fd}(A_kf_fd)
const B_kf_fd_s = SMatrix{N_kf_fd, K_kf_fd}(B_kf_fd)
const C_kf_fd_s = SMatrix{M_kf_fd, N_kf_fd}(C_kf_fd)
const R_kf_fd_s = SMatrix{M_kf_fd, M_kf_fd}(R_kf_fd)
const mu_0_kf_fd_s = SVector{N_kf_fd}(mu_0_kf_fd)
const Sigma_0_kf_fd_s = SMatrix{N_kf_fd, N_kf_fd}(Sigma_0_kf_fd)
const y_kf_fd_s = [SVector{M_kf_fd}(yi) for yi in y_kf_fd]

function kalman_loglik_fd_static(A_vec, B, C, mu_0, Sigma_0, R, y,
        ::Val{N}, ::Val{M}, ::Val{K}) where {N, M, K}
    T_el = eltype(A_vec)
    A = SMatrix{N, N}(reshape(A_vec, N, N))
    prob = LinearStateSpaceProblem(
        A, SMatrix{N, K}(T_el.(B)),
        SVector{N}(zeros(T_el, N)), (0, length(y));
        C = SMatrix{M, N}(T_el.(C)),
        u0_prior_mean = SVector{N}(T_el.(mu_0)),
        u0_prior_var = SMatrix{N, N}(T_el.(Sigma_0)),
        observables_noise = SMatrix{M, M}(T_el.(R)),
        observables = y)
    sol = solve(prob, KalmanFilter())
    return sol.logpdf
end

@testset "ForwardDiff - Kalman Filter (static)" begin
    @testset "gradient w.r.t. A" begin
        f = a_vec -> kalman_loglik_fd_static(a_vec, B_kf_fd_s, C_kf_fd_s,
            mu_0_kf_fd_s, Sigma_0_kf_fd_s, R_kf_fd_s, y_kf_fd_s,
            Val(N_kf_fd), Val(M_kf_fd), Val(K_kf_fd))
        x0 = collect(vec(Matrix(A_kf_fd)))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end

    @testset "gradient w.r.t. C" begin
        f = c_vec -> begin
            T_el = eltype(c_vec)
            prob = LinearStateSpaceProblem(
                SMatrix{N_kf_fd, N_kf_fd}(T_el.(A_kf_fd)),
                SMatrix{N_kf_fd, K_kf_fd}(T_el.(B_kf_fd)),
                SVector{N_kf_fd}(zeros(T_el, N_kf_fd)), (0, length(y_kf_fd_s));
                C = SMatrix{M_kf_fd, N_kf_fd}(reshape(c_vec, M_kf_fd, N_kf_fd)),
                u0_prior_mean = SVector{N_kf_fd}(T_el.(mu_0_kf_fd)),
                u0_prior_var = SMatrix{N_kf_fd, N_kf_fd}(T_el.(Sigma_0_kf_fd)),
                observables_noise = SMatrix{M_kf_fd, M_kf_fd}(T_el.(R_kf_fd)),
                observables = y_kf_fd_s)
            sol = solve(prob, KalmanFilter())
            return sol.logpdf
        end
        x0 = collect(vec(Matrix(C_kf_fd)))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end
end
