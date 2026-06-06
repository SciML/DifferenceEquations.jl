# ForwardDiff AD tests for ConditionalLikelihood
# Tests gradient correctness against FiniteDifferences.jl central FD.

using LinearAlgebra, Test, ForwardDiff, StaticArrays, Random
using DifferenceEquations
using FiniteDifferences: central_fdm, grad

include("forwarddiff_test_utils.jl")  # promote_array only

const _fdm_cl_fd = central_fdm(5, 1)

# =============================================================================
# Problem setup
# =============================================================================

const N_cl_fd = 2
const M_cl_fd = 2
const T_cl_fd = 10

const A_cl_fd = [0.8 0.1; -0.1 0.7]
const H_cl_fd = [0.1 0.0; 0.0 0.1]
const u0_cl_fd = zeros(N_cl_fd)

# Generate observables from an AR process
Random.seed!(42)
const y_cl_fd = let
    y = Vector{Vector{Float64}}(undef, T_cl_fd)
    x = zeros(N_cl_fd)
    for t in 1:T_cl_fd
        x = A_cl_fd * x + H_cl_fd * randn(N_cl_fd)
        y[t] = copy(x)
    end
    y
end

# =============================================================================
# Mutable arrays — ForwardDiff gradient tests
# =============================================================================

function cl_loglik_fd(A, u0, y, H)
    T_el = promote_type(eltype(A), eltype(u0), eltype(H))
    R = promote_array(T_el, H) * promote_array(T_el, H)'
    prob = LinearStateSpaceProblem(
        promote_array(T_el, A), nothing,
        promote_array(T_el, u0), (0, length(y));
        observables_noise = R,
        observables = y,
    )
    return solve(prob, ConditionalLikelihood()).logpdf
end

@testset "ForwardDiff - ConditionalLikelihood loglik (mutable)" begin
    @testset "primal sanity" begin
        loglik_val = cl_loglik_fd(A_cl_fd, u0_cl_fd, y_cl_fd, H_cl_fd)
        @test isfinite(loglik_val)
    end

    @testset "gradient w.r.t. A" begin
        f(a_vec) = cl_loglik_fd(
            reshape(a_vec, N_cl_fd, N_cl_fd), u0_cl_fd, y_cl_fd, H_cl_fd
        )
        x0 = vec(copy(A_cl_fd))
        @test ForwardDiff.gradient(f, x0) ≈ grad(_fdm_cl_fd, f, x0)[1] rtol = 1.0e-4
    end

    @testset "gradient w.r.t. u0" begin
        f(u_vec) = cl_loglik_fd(A_cl_fd, u_vec, y_cl_fd, H_cl_fd)
        x0 = [0.1, -0.1]
        @test ForwardDiff.gradient(f, x0) ≈ grad(_fdm_cl_fd, f, x0)[1] rtol = 1.0e-4
    end

    @testset "gradient w.r.t. H" begin
        f(h_vec) = cl_loglik_fd(
            A_cl_fd, u0_cl_fd, y_cl_fd, reshape(h_vec, M_cl_fd, M_cl_fd)
        )
        x0 = vec(copy(H_cl_fd))
        @test ForwardDiff.gradient(f, x0) ≈ grad(_fdm_cl_fd, f, x0)[1] rtol = 1.0e-4
    end
end

# =============================================================================
# Non-diagonal R — ForwardDiff gradient tests
# =============================================================================

const H_cl_fd_offdiag = [0.1 0.05; 0.02 0.08]

@testset "ForwardDiff - ConditionalLikelihood non-diagonal R (mutable)" begin
    @testset "primal sanity" begin
        loglik_val = cl_loglik_fd(A_cl_fd, u0_cl_fd, y_cl_fd, H_cl_fd_offdiag)
        @test isfinite(loglik_val)
    end

    @testset "gradient w.r.t. H (off-diagonal)" begin
        f(h_vec) = cl_loglik_fd(
            A_cl_fd, u0_cl_fd, y_cl_fd, reshape(h_vec, M_cl_fd, M_cl_fd)
        )
        x0 = vec(copy(H_cl_fd_offdiag))
        @test ForwardDiff.gradient(f, x0) ≈ grad(_fdm_cl_fd, f, x0)[1] rtol = 1.0e-4
    end

    @testset "gradient w.r.t. A (with off-diagonal R)" begin
        f(a_vec) = cl_loglik_fd(
            reshape(a_vec, N_cl_fd, N_cl_fd), u0_cl_fd, y_cl_fd, H_cl_fd_offdiag
        )
        x0 = vec(copy(A_cl_fd))
        @test ForwardDiff.gradient(f, x0) ≈ grad(_fdm_cl_fd, f, x0)[1] rtol = 1.0e-4
    end
end

# =============================================================================
# With C matrix — ForwardDiff gradient tests
# =============================================================================

const C_cl_fd = [1.0 0.0; 0.0 1.0]

function cl_loglik_fd_with_c(A, C, u0, y, H)
    T_el = promote_type(eltype(A), eltype(C), eltype(u0), eltype(H))
    R = promote_array(T_el, H) * promote_array(T_el, H)'
    prob = LinearStateSpaceProblem(
        promote_array(T_el, A), nothing,
        promote_array(T_el, u0), (0, length(y));
        C = promote_array(T_el, C),
        observables_noise = R,
        observables = y,
    )
    return solve(prob, ConditionalLikelihood()).logpdf
end

@testset "ForwardDiff - ConditionalLikelihood with C (mutable)" begin
    @testset "gradient w.r.t. A (with C)" begin
        f(a_vec) = cl_loglik_fd_with_c(
            reshape(a_vec, N_cl_fd, N_cl_fd), C_cl_fd, u0_cl_fd, y_cl_fd, H_cl_fd
        )
        x0 = vec(copy(A_cl_fd))
        @test ForwardDiff.gradient(f, x0) ≈ grad(_fdm_cl_fd, f, x0)[1] rtol = 1.0e-4
    end
end

# =============================================================================
# StaticArrays — ForwardDiff gradient tests
# =============================================================================

const y_cl_fd_s = [SVector{M_cl_fd}(yi) for yi in y_cl_fd]

@testset "ForwardDiff - ConditionalLikelihood loglik (static)" begin
    @testset "gradient w.r.t. A" begin
        function _cl_static_A(a_vec)
            T_el = eltype(a_vec)
            A_d = SMatrix{N_cl_fd, N_cl_fd}(reshape(a_vec, N_cl_fd, N_cl_fd))
            H_d = SMatrix{M_cl_fd, M_cl_fd}(T_el.(H_cl_fd))
            prob = LinearStateSpaceProblem(
                A_d, nothing,
                SVector{N_cl_fd}(zeros(T_el, N_cl_fd)), (0, length(y_cl_fd_s));
                observables_noise = H_d * H_d',
                observables = y_cl_fd_s,
            )
            return solve(prob, ConditionalLikelihood()).logpdf
        end
        x0 = collect(vec(Matrix(A_cl_fd)))
        @test ForwardDiff.gradient(_cl_static_A, x0) ≈
            grad(_fdm_cl_fd, _cl_static_A, x0)[1] rtol = 1.0e-4
    end

    @testset "gradient w.r.t. H" begin
        function _cl_static_H(h_vec)
            T_el = eltype(h_vec)
            A_d = SMatrix{N_cl_fd, N_cl_fd}(T_el.(A_cl_fd))
            H_d = SMatrix{M_cl_fd, M_cl_fd}(reshape(h_vec, M_cl_fd, M_cl_fd))
            prob = LinearStateSpaceProblem(
                A_d, nothing,
                SVector{N_cl_fd}(zeros(T_el, N_cl_fd)), (0, length(y_cl_fd_s));
                observables_noise = H_d * H_d',
                observables = y_cl_fd_s,
            )
            return solve(prob, ConditionalLikelihood()).logpdf
        end
        x0 = collect(vec(Matrix(H_cl_fd)))
        @test ForwardDiff.gradient(_cl_static_H, x0) ≈
            grad(_fdm_cl_fd, _cl_static_H, x0)[1] rtol = 1.0e-4
    end
end

# =============================================================================
# Generic nonlinear StateSpaceProblem — ForwardDiff gradient
# =============================================================================

@testset "ForwardDiff - ConditionalLikelihood generic nonlinear (mutable)" begin
    T_nl = 15
    sigma_e_nl = 0.3

    Random.seed!(99)
    y_nl = let
        y = Vector{Vector{Float64}}(undef, T_nl)
        x = 0.0
        for t in 1:T_nl
            x = 0.8 * x + 0.05 * x^2 + sigma_e_nl * randn()
            y[t] = [x]
        end
        y
    end

    nl_f!! = (x_next, x, w, p, t) -> begin
        (; rho, alpha) = p
        val = rho * x[1] + alpha * x[1]^2
        if ismutable(x_next)
            x_next[1] = val
            return x_next
        else
            return typeof(x)(val)
        end
    end

    function cl_nl_loglik(param_vec, y, sigma_e)
        T_el = eltype(param_vec)
        p = (; rho = param_vec[1], alpha = param_vec[2])
        prob = StateSpaceProblem(
            nl_f!!, nothing, [zero(T_el)], (0, length(y)), p;
            n_shocks = 0, n_obs = 0,
            observables = y,
            observables_noise = Diagonal([T_el(sigma_e^2)]),
        )
        return solve(prob, ConditionalLikelihood()).logpdf
    end

    @testset "primal sanity" begin
        @test isfinite(cl_nl_loglik([0.8, 0.05], y_nl, sigma_e_nl))
    end

    @testset "gradient w.r.t. (rho, alpha)" begin
        f(p_vec) = cl_nl_loglik(p_vec, y_nl, sigma_e_nl)
        x0 = [0.8, 0.05]
        @test ForwardDiff.gradient(f, x0) ≈ grad(_fdm_cl_fd, f, x0)[1] rtol = 1.0e-4
    end
end
