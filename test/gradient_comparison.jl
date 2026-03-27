# Apples-to-apples gradient comparison: ForwardDiff vs Enzyme BatchDuplicated vs Enzyme Reverse
# All methods compute the SAME quantity: full gradient of loglik w.r.t. vec(A) (N² components).

using LinearAlgebra, Test, ForwardDiff, Enzyme, Random
using Enzyme: make_zero, make_zero!
using DifferenceEquations
using DifferenceEquations: init, solve!, StateSpaceWorkspace, fill_zero!!

include("forwarddiff_test_utils.jl")

# =============================================================================
# Kalman problem setup
# =============================================================================

const N_gc = 2
const M_gc = 2
const K_gc = 2
const T_gc = 3
const CHUNK_gc = 2  # batch size for Enzyme BatchDuplicated

const A_gc = [0.8 0.1; -0.1 0.7]
const B_gc = [0.1 0.0; 0.0 0.1]
const C_gc = [1.0 0.0; 0.0 1.0]
const R_gc = [0.01 0.0; 0.0 0.01]
const mu_0_gc = zeros(N_gc)
const Sigma_0_gc = Matrix{Float64}(I, N_gc, N_gc)
const y_gc = [[0.5, 0.3], [0.2, 0.1], [0.8, 0.4]]

# Enzyme workspace (pre-allocated Float64)
function _make_gc_workspace()
    prob = LinearStateSpaceProblem(
        A_gc, B_gc, zeros(N_gc), (0, T_gc); C = C_gc,
        u0_prior_mean = mu_0_gc, u0_prior_var = Sigma_0_gc,
        observables_noise = R_gc, observables = y_gc
    )
    ws = init(prob, KalmanFilter())
    return ws.output, ws.cache
end

# =============================================================================
# 1. ForwardDiff gradient w.r.t. vec(A)
# =============================================================================

function _kf_loglik_fd(A_vec)
    T_el = eltype(A_vec)
    A = reshape(A_vec, N_gc, N_gc)
    prob = LinearStateSpaceProblem(
        A, promote_array(T_el, B_gc),
        zeros(T_el, N_gc), (0, T_gc);
        C = promote_array(T_el, C_gc),
        u0_prior_mean = promote_array(T_el, mu_0_gc),
        u0_prior_var = promote_array(T_el, Sigma_0_gc),
        observables_noise = promote_array(T_el, R_gc),
        observables = y_gc
    )
    sol = solve(prob, KalmanFilter())
    return sol.logpdf
end

# =============================================================================
# 2. Enzyme BatchDuplicated forward — full gradient via chunked forward passes
# =============================================================================

function _kf_loglik_enzyme!(A, B, C, mu_0, Sigma_0, R, y, sol_out, cache)
    prob = LinearStateSpaceProblem(
        A, B, zeros(eltype(A), size(A, 1)), (0, length(y)); C,
        u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y
    )
    ws = StateSpaceWorkspace(prob, KalmanFilter(), sol_out, cache)
    return solve!(ws).logpdf
end

function enzyme_batched_forward_gradient_kf!(
        grad_out, A, B, C, mu_0, Sigma_0, R, y,
        sol_out, cache, chunk_size,
        dAs, dBs, dCs, dmu0s, dSig0s, dRs, dys, dsols, dcaches
    )
    N_params = length(vec(A))
    for chunk_start in 1:chunk_size:N_params
        chunk_end = min(chunk_start + chunk_size - 1, N_params)
        actual = chunk_end - chunk_start + 1

        # Zero all shadows
        for k in 1:chunk_size
            fill_zero!!(dAs[k]); fill_zero!!(dBs[k]); fill_zero!!(dCs[k])
            fill_zero!!(dmu0s[k]); fill_zero!!(dSig0s[k]); fill_zero!!(dRs[k])
            for t in eachindex(dys[k])
                dys[k][t] = fill_zero!!(dys[k][t])
            end
            make_zero!(dsols[k]); make_zero!(dcaches[k])
        end

        # Seed directions: standard basis vectors for vec(A)
        for k in 1:actual
            dAs[k][chunk_start + k - 1] = 1.0
        end

        result = autodiff(
            Forward, _kf_loglik_enzyme!,
            BatchDuplicated(A, dAs),
            BatchDuplicated(B, dBs),
            BatchDuplicated(C, dCs),
            BatchDuplicated(mu_0, dmu0s),
            BatchDuplicated(Sigma_0, dSig0s),
            BatchDuplicated(R, dRs),
            BatchDuplicated(y, dys),
            BatchDuplicated(sol_out, dsols),
            BatchDuplicated(cache, dcaches)
        )

        # Result is ((d1, d2, ...),) for scalar return
        derivs = values(result[1])
        for k in 1:actual
            grad_out[chunk_start + k - 1] = derivs[k]
        end
    end
    return grad_out
end

# =============================================================================
# 3. Enzyme Reverse — full gradient, extract dA
# =============================================================================

function enzyme_reverse_gradient_kf!(
        A, B, C, mu_0, Sigma_0, R, y,
        sol_out, cache, dA, dB, dC, dmu_0, dSigma_0, dR, dy, dsol_out, dcache
    )
    make_zero!(dsol_out); make_zero!(dcache)
    fill_zero!!(dA); fill_zero!!(dB); fill_zero!!(dC)
    fill_zero!!(dmu_0); fill_zero!!(dSigma_0); fill_zero!!(dR)
    @inbounds for i in eachindex(dy)
        dy[i] = fill_zero!!(dy[i])
    end

    autodiff(
        Reverse, _kf_loglik_enzyme!, Active,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Duplicated(R, dR), Duplicated(y, dy),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache)
    )
    return vec(dA)
end

# =============================================================================
# Tests
# =============================================================================

@testset "Gradient comparison - Kalman loglik w.r.t. vec(A)" begin
    A_vec = vec(copy(A_gc))

    # Finite differences (baseline)
    grad_fin = fdm_gradient(_kf_loglik_fd, A_vec)

    # ForwardDiff
    grad_fd = ForwardDiff.gradient(_kf_loglik_fd, A_vec)

    # Enzyme BatchDuplicated forward
    sol_out_bf, cache_bf = _make_gc_workspace()
    N_params = length(A_vec)
    dAs = ntuple(_ -> make_zero(A_gc), CHUNK_gc)
    dBs = ntuple(_ -> make_zero(B_gc), CHUNK_gc)
    dCs = ntuple(_ -> make_zero(C_gc), CHUNK_gc)
    dmu0s = ntuple(_ -> make_zero(mu_0_gc), CHUNK_gc)
    dSig0s = ntuple(_ -> make_zero(Sigma_0_gc), CHUNK_gc)
    dRs = ntuple(_ -> make_zero(R_gc), CHUNK_gc)
    dys = ntuple(_ -> [make_zero(y_gc[1]) for _ in 1:T_gc], CHUNK_gc)
    dsols = ntuple(_ -> make_zero(sol_out_bf), CHUNK_gc)
    dcaches = ntuple(_ -> make_zero(cache_bf), CHUNK_gc)

    grad_enzyme_fwd = zeros(N_params)
    enzyme_batched_forward_gradient_kf!(
        grad_enzyme_fwd,
        A_gc, B_gc, C_gc, mu_0_gc, Sigma_0_gc, R_gc, y_gc,
        sol_out_bf, cache_bf, CHUNK_gc,
        dAs, dBs, dCs, dmu0s, dSig0s, dRs, dys, dsols, dcaches
    )

    # Enzyme Reverse
    sol_out_rv, cache_rv = _make_gc_workspace()
    dA_rv = make_zero(A_gc); dB_rv = make_zero(B_gc); dC_rv = make_zero(C_gc)
    dmu0_rv = make_zero(mu_0_gc); dSig0_rv = make_zero(Sigma_0_gc); dR_rv = make_zero(R_gc)
    dy_rv = [make_zero(y_gc[1]) for _ in 1:T_gc]
    dsol_rv = make_zero(sol_out_rv); dcache_rv = make_zero(cache_rv)

    grad_enzyme_rev = enzyme_reverse_gradient_kf!(
        A_gc, B_gc, C_gc, mu_0_gc, Sigma_0_gc, R_gc, y_gc,
        sol_out_rv, cache_rv, dA_rv, dB_rv, dC_rv, dmu0_rv, dSig0_rv, dR_rv,
        dy_rv, dsol_rv, dcache_rv
    )

    @testset "all methods finite" begin
        @test all(isfinite, grad_fin)
        @test all(isfinite, grad_fd)
        @test all(isfinite, grad_enzyme_fwd)
        @test all(isfinite, grad_enzyme_rev)
    end

    @testset "ForwardDiff matches finite differences" begin
        @test grad_fd ≈ grad_fin rtol = 1.0e-4
    end

    @testset "Enzyme BatchDuplicated forward matches finite differences" begin
        @test grad_enzyme_fwd ≈ grad_fin rtol = 1.0e-4
    end

    @testset "Enzyme reverse matches finite differences" begin
        @test grad_enzyme_rev ≈ grad_fin rtol = 1.0e-4
    end

    @testset "ForwardDiff matches Enzyme reverse (high precision)" begin
        @test grad_fd ≈ grad_enzyme_rev rtol = 1.0e-10
    end

    @testset "Enzyme BatchDuplicated forward matches Enzyme reverse (high precision)" begin
        @test grad_enzyme_fwd ≈ grad_enzyme_rev rtol = 1.0e-10
    end
end

# =============================================================================
# DirectIteration variant
# =============================================================================

const A_di_gc = [0.8 0.1; -0.1 0.7]
const B_di_gc = [0.1 0.0; 0.0 0.1]
const C_di_gc = [1.0 0.0; 0.0 1.0]
const H_di_gc = [0.1 0.0; 0.0 0.1]
const u0_di_gc = [0.1, -0.1]
const noise_di_gc = [[0.1, -0.1], [0.2, 0.05], [0.0, 0.1]]
const y_di_gc = [[0.5, 0.3], [0.2, 0.1], [0.8, 0.4]]

function _make_di_gc_workspace()
    R = H_di_gc * H_di_gc'
    prob = LinearStateSpaceProblem(
        A_di_gc, B_di_gc, u0_di_gc, (0, T_gc);
        C = C_di_gc, observables_noise = R, observables = y_di_gc, noise = noise_di_gc
    )
    ws = init(prob, DirectIteration())
    return ws.output, ws.cache
end

function _di_loglik_fd_gc(A_vec)
    T_el = eltype(A_vec)
    A = reshape(A_vec, N_gc, N_gc)
    H = promote_array(T_el, H_di_gc)
    R = H * H'
    prob = LinearStateSpaceProblem(
        A, promote_array(T_el, B_di_gc),
        promote_array(T_el, u0_di_gc), (0, T_gc);
        C = promote_array(T_el, C_di_gc),
        observables_noise = R,
        observables = y_di_gc, noise = noise_di_gc
    )
    sol = solve(prob, DirectIteration())
    return sol.logpdf
end

function _di_loglik_enzyme!(A, B, C, u0, noise, y, H, sol_out, cache)
    R = H * H'
    prob = LinearStateSpaceProblem(
        A, B, u0, (0, length(y));
        C, observables_noise = R, observables = y, noise
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return solve!(ws).logpdf
end

function enzyme_batched_forward_gradient_di!(
        grad_out, A, B, C, u0, noise, y, H,
        sol_out, cache, chunk_size,
        dAs, dBs, dCs, du0s, dnoises, dys, dHs, dsols, dcaches
    )
    N_params = length(vec(A))
    for chunk_start in 1:chunk_size:N_params
        chunk_end = min(chunk_start + chunk_size - 1, N_params)
        actual = chunk_end - chunk_start + 1

        for k in 1:chunk_size
            fill_zero!!(dAs[k]); fill_zero!!(dBs[k]); fill_zero!!(dCs[k])
            fill_zero!!(du0s[k]); fill_zero!!(dHs[k])
            for t in eachindex(dnoises[k])
                dnoises[k][t] = fill_zero!!(dnoises[k][t])
            end
            for t in eachindex(dys[k])
                dys[k][t] = fill_zero!!(dys[k][t])
            end
            make_zero!(dsols[k]); make_zero!(dcaches[k])
        end
        for k in 1:actual
            dAs[k][chunk_start + k - 1] = 1.0
        end

        result = autodiff(
            Forward, _di_loglik_enzyme!,
            BatchDuplicated(A, dAs),
            BatchDuplicated(B, dBs),
            BatchDuplicated(C, dCs),
            BatchDuplicated(u0, du0s),
            BatchDuplicated(noise, dnoises),
            BatchDuplicated(y, dys),
            BatchDuplicated(H, dHs),
            BatchDuplicated(sol_out, dsols),
            BatchDuplicated(cache, dcaches)
        )

        derivs = values(result[1])
        for k in 1:actual
            grad_out[chunk_start + k - 1] = derivs[k]
        end
    end
    return grad_out
end

@testset "Gradient comparison - DI loglik w.r.t. vec(A)" begin
    A_vec = vec(copy(A_di_gc))
    N_params = length(A_vec)

    grad_fin = fdm_gradient(_di_loglik_fd_gc, A_vec)
    grad_fd = ForwardDiff.gradient(_di_loglik_fd_gc, A_vec)

    # Enzyme BatchDuplicated forward
    sol_out_bf, cache_bf = _make_di_gc_workspace()
    dAs = ntuple(_ -> make_zero(A_di_gc), CHUNK_gc)
    dBs = ntuple(_ -> make_zero(B_di_gc), CHUNK_gc)
    dCs = ntuple(_ -> make_zero(C_di_gc), CHUNK_gc)
    du0s = ntuple(_ -> make_zero(u0_di_gc), CHUNK_gc)
    dnoises = ntuple(_ -> [make_zero(noise_di_gc[1]) for _ in 1:T_gc], CHUNK_gc)
    dys = ntuple(_ -> [make_zero(y_di_gc[1]) for _ in 1:T_gc], CHUNK_gc)
    dHs = ntuple(_ -> make_zero(H_di_gc), CHUNK_gc)
    dsols = ntuple(_ -> make_zero(sol_out_bf), CHUNK_gc)
    dcaches = ntuple(_ -> make_zero(cache_bf), CHUNK_gc)

    grad_enzyme_fwd = zeros(N_params)
    enzyme_batched_forward_gradient_di!(
        grad_enzyme_fwd,
        A_di_gc, B_di_gc, C_di_gc, u0_di_gc, noise_di_gc, y_di_gc, H_di_gc,
        sol_out_bf, cache_bf, CHUNK_gc,
        dAs, dBs, dCs, du0s, dnoises, dys, dHs, dsols, dcaches
    )

    # Enzyme Reverse
    sol_out_rv, cache_rv = _make_di_gc_workspace()
    dA_rv = make_zero(A_di_gc); dB_rv = make_zero(B_di_gc); dC_rv = make_zero(C_di_gc)
    du0_rv = make_zero(u0_di_gc); dH_rv = make_zero(H_di_gc)
    dnoise_rv = [make_zero(noise_di_gc[1]) for _ in 1:T_gc]
    dy_rv = [make_zero(y_di_gc[1]) for _ in 1:T_gc]
    dsol_rv = make_zero(sol_out_rv); dcache_rv = make_zero(cache_rv)

    autodiff(
        Reverse, _di_loglik_enzyme!, Active,
        Duplicated(A_di_gc, dA_rv), Duplicated(B_di_gc, dB_rv),
        Duplicated(C_di_gc, dC_rv), Duplicated(u0_di_gc, du0_rv),
        Duplicated(noise_di_gc, dnoise_rv),
        Duplicated(y_di_gc, dy_rv),
        Duplicated(H_di_gc, dH_rv),
        Duplicated(sol_out_rv, dsol_rv), Duplicated(cache_rv, dcache_rv)
    )
    grad_enzyme_rev = vec(dA_rv)

    @testset "all methods finite" begin
        @test all(isfinite, grad_fin)
        @test all(isfinite, grad_fd)
        @test all(isfinite, grad_enzyme_fwd)
        @test all(isfinite, grad_enzyme_rev)
    end

    @testset "ForwardDiff matches finite differences" begin
        @test grad_fd ≈ grad_fin rtol = 1.0e-4
    end

    @testset "Enzyme BatchDuplicated forward matches finite differences" begin
        @test grad_enzyme_fwd ≈ grad_fin rtol = 1.0e-4
    end

    @testset "Enzyme reverse matches finite differences" begin
        @test grad_enzyme_rev ≈ grad_fin rtol = 1.0e-4
    end

    @testset "all AD methods agree (high precision)" begin
        @test grad_fd ≈ grad_enzyme_rev rtol = 1.0e-10
        @test grad_enzyme_fwd ≈ grad_enzyme_rev rtol = 1.0e-10
    end
end
