# ForwardDiff AD tests for DirectIteration (loglik path)
# Tests gradient correctness against central finite differences.

using LinearAlgebra, Test, ForwardDiff, StaticArrays, Random
using DifferenceEquations

include("forwarddiff_test_utils.jl")

# =============================================================================
# Problem setup
# =============================================================================

const N_di_fd = 2
const M_di_fd = 2
const K_di_fd = 2
const T_di_fd = 5

const A_di_fd = [0.8 0.1; -0.1 0.7]
const B_di_fd = [0.1 0.0; 0.0 0.1]
const C_di_fd = [1.0 0.0; 0.0 1.0]
const H_di_fd = [0.1 0.0; 0.0 0.1]
const u0_di_fd = zeros(N_di_fd)

Random.seed!(42)
const noise_di_fd = [randn(K_di_fd) for _ in 1:T_di_fd]
const sim_sol_di_fd = solve(LinearStateSpaceProblem(
    A_di_fd, B_di_fd, u0_di_fd, (0, T_di_fd); C = C_di_fd, noise = noise_di_fd))
const y_di_fd = [sim_sol_di_fd.z[t + 1] + H_di_fd * randn(M_di_fd) for t in 1:T_di_fd]

# =============================================================================
# Mutable arrays — ForwardDiff gradient tests
# =============================================================================

function di_loglik_fd(A, B, C, u0, noise, y, H)
    T_el = promote_type(eltype(A), eltype(B), eltype(C), eltype(u0), eltype(H))
    R = promote_array(T_el, H) * promote_array(T_el, H)'
    prob = LinearStateSpaceProblem(
        promote_array(T_el, A), promote_array(T_el, B),
        promote_array(T_el, u0), (0, length(y));
        C = promote_array(T_el, C),
        observables_noise = R,
        observables = y, noise = noise)
    sol = solve(prob, DirectIteration())
    return sol.logpdf
end

@testset "ForwardDiff - DirectIteration loglik (mutable)" begin
    @testset "primal sanity" begin
        loglik_val = di_loglik_fd(A_di_fd, B_di_fd, C_di_fd, u0_di_fd,
            noise_di_fd, y_di_fd, H_di_fd)
        @test isfinite(loglik_val)
    end

    @testset "gradient w.r.t. A" begin
        f = a_vec -> di_loglik_fd(reshape(a_vec, N_di_fd, N_di_fd),
            B_di_fd, C_di_fd, u0_di_fd, noise_di_fd, y_di_fd, H_di_fd)
        x0 = vec(copy(A_di_fd))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end

    @testset "gradient w.r.t. u0" begin
        f = u_vec -> di_loglik_fd(A_di_fd, B_di_fd, C_di_fd,
            u_vec, noise_di_fd, y_di_fd, H_di_fd)
        x0 = [0.1, -0.1]
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end

    @testset "gradient w.r.t. H" begin
        f = h_vec -> di_loglik_fd(A_di_fd, B_di_fd, C_di_fd, u0_di_fd,
            noise_di_fd, y_di_fd, reshape(h_vec, M_di_fd, M_di_fd))
        x0 = vec(copy(H_di_fd))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end
end

# =============================================================================
# StaticArrays — ForwardDiff gradient tests
# =============================================================================

const noise_di_fd_s = [SVector{K_di_fd}(n) for n in noise_di_fd]
const y_di_fd_s = [SVector{M_di_fd}(yi) for yi in y_di_fd]

@testset "ForwardDiff - DirectIteration loglik (static)" begin
    @testset "gradient w.r.t. A" begin
        f = a_vec -> begin
            T_el = eltype(a_vec)
            A_d = SMatrix{N_di_fd, N_di_fd}(reshape(a_vec, N_di_fd, N_di_fd))
            B_d = SMatrix{N_di_fd, K_di_fd}(T_el.(B_di_fd))
            C_d = SMatrix{M_di_fd, N_di_fd}(T_el.(C_di_fd))
            H_d = SMatrix{M_di_fd, M_di_fd}(T_el.(H_di_fd))
            prob = LinearStateSpaceProblem(A_d, B_d,
                SVector{N_di_fd}(zeros(T_el, N_di_fd)), (0, length(y_di_fd_s));
                C = C_d, observables_noise = H_d * H_d',
                observables = y_di_fd_s, noise = noise_di_fd_s)
            sol = solve(prob, DirectIteration())
            return sol.logpdf
        end
        x0 = collect(vec(Matrix(A_di_fd)))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end

    @testset "gradient w.r.t. H" begin
        f = h_vec -> begin
            T_el = eltype(h_vec)
            A_d = SMatrix{N_di_fd, N_di_fd}(T_el.(A_di_fd))
            B_d = SMatrix{N_di_fd, K_di_fd}(T_el.(B_di_fd))
            C_d = SMatrix{M_di_fd, N_di_fd}(T_el.(C_di_fd))
            H_d = SMatrix{M_di_fd, M_di_fd}(reshape(h_vec, M_di_fd, M_di_fd))
            prob = LinearStateSpaceProblem(A_d, B_d,
                SVector{N_di_fd}(zeros(T_el, N_di_fd)), (0, length(y_di_fd_s));
                C = C_d, observables_noise = H_d * H_d',
                observables = y_di_fd_s, noise = noise_di_fd_s)
            sol = solve(prob, DirectIteration())
            return sol.logpdf
        end
        x0 = collect(vec(Matrix(H_di_fd)))
        @test ForwardDiff.gradient(f, x0) ≈ fdm_gradient(f, x0) rtol = 1e-4
    end
end
