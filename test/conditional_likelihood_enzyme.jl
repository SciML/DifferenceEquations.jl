# Enzyme AD tests for ConditionalLikelihood
# Forward: test_forward (EnzymeTestUtils) — sol/cache as Duplicated args
# Reverse: test_reverse (EnzymeTestUtils) — sol/cache as Duplicated args

using LinearAlgebra, Test, Enzyme, EnzymeTestUtils, Random
using DifferenceEquations
using DifferenceEquations: init, solve!, StateSpaceWorkspace
using FiniteDifferences: central_fdm

# vech utilities for non-diagonal R test (from enzyme_test_utils.jl)
include("enzyme_test_utils.jl")
# Uses: make_posdef_from_vech, make_vech_for, vech_length

# =============================================================================
# Test data
# =============================================================================

const N_cl_e = 2
const T_cl_e = 5

const A_cl_e = [0.8 0.1; -0.1 0.7]
const H_cl_e = [0.1 0.0; 0.0 0.1]
const u0_cl_e = zeros(N_cl_e)

Random.seed!(42)
const y_cl_e = [randn(N_cl_e) for _ in 1:T_cl_e]

const _fdm_cl = central_fdm(5, 1)

# =============================================================================
# Helpers
# =============================================================================

function make_cl_prob(A, u0, y, H)
    R = H * H'
    return LinearStateSpaceProblem(
        A, nothing, u0, (0, length(y));
        observables_noise = R, observables = y
    )
end

function make_cl_sol_cache(A, u0, y, H)
    ws = init(make_cl_prob(A, u0, y, H), ConditionalLikelihood())
    return ws.output, ws.cache
end

# =============================================================================
# Forward wrapper — returns output (not workspace)
# =============================================================================

function cl_forward!(A, u0, y, H, sol, cache)
    prob = make_cl_prob(A, u0, y, H)
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol, cache)
    solve!(ws)
    return sol.u
end

# =============================================================================
# Reverse wrapper — scalar return, sol/cache as Duplicated args
# =============================================================================

function cl_loglik(A, u0, y, H, sol, cache)::Float64
    prob = make_cl_prob(A, u0, y, H)
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol, cache)
    return solve!(ws).logpdf
end

function cl_loglik_with_c(A, C, u0, y, H, sol, cache)::Float64
    R = H * H'
    prob = LinearStateSpaceProblem(
        A, nothing, u0, (0, length(y));
        C = C, observables_noise = R, observables = y
    )
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol, cache)
    return solve!(ws).logpdf
end

function cl_loglik_vech(A, v_R, u0, y, sol, cache)::Float64
    R = make_posdef_from_vech(v_R, size(A, 1))
    prob = LinearStateSpaceProblem(
        A, nothing, u0, (0, length(y));
        observables_noise = R, observables = y
    )
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol, cache)
    return solve!(ws).logpdf
end

# =============================================================================
# Sanity
# =============================================================================

@testset "ConditionalLikelihood loglik via solve!() — Enzyme sanity" begin
    sol, cache = make_cl_sol_cache(A_cl_e, u0_cl_e, y_cl_e, H_cl_e)
    loglik = cl_loglik(A_cl_e, u0_cl_e, y_cl_e, H_cl_e, sol, cache)
    @test isfinite(loglik)
    loglik2 = cl_loglik(A_cl_e, u0_cl_e, y_cl_e, H_cl_e, sol, cache)
    @test loglik ≈ loglik2 rtol = 1.0e-12
end

# =============================================================================
# Forward (all Duplicated)
# =============================================================================

@testset "EnzymeTestUtils — CL forward (all Duplicated)" begin
    sol, cache = make_cl_sol_cache(A_cl_e, u0_cl_e, y_cl_e, H_cl_e)

    test_forward(
        cl_forward!, Const,
        (copy(A_cl_e), Duplicated), (copy(u0_cl_e), Duplicated),
        ([copy(yi) for yi in y_cl_e], Duplicated),
        (copy(H_cl_e), Duplicated),
        (sol, Duplicated), (cache, Duplicated);
        fdm = _fdm_cl,
    )
end

# =============================================================================
# Reverse — test_reverse with sol/cache as Duplicated
# =============================================================================

@testset "EnzymeTestUtils — CL reverse (all Duplicated)" begin
    sol, cache = make_cl_sol_cache(A_cl_e, u0_cl_e, y_cl_e, H_cl_e)

    test_reverse(
        cl_loglik, Active,
        (copy(A_cl_e), Duplicated), (copy(u0_cl_e), Duplicated),
        ([copy(yi) for yi in y_cl_e], Duplicated),
        (copy(H_cl_e), Duplicated),
        (deepcopy(sol), Duplicated), (deepcopy(cache), Duplicated);
        fdm = _fdm_cl,
    )
end

# =============================================================================
# Reverse — with C matrix
# =============================================================================

@testset "EnzymeTestUtils — CL reverse with C" begin
    C_cl = [1.0 0.0; 0.0 1.0]
    prob_c = LinearStateSpaceProblem(
        A_cl_e, nothing, u0_cl_e, (0, T_cl_e);
        C = C_cl, observables_noise = H_cl_e * H_cl_e', observables = y_cl_e
    )
    ws_c = init(prob_c, ConditionalLikelihood())

    test_reverse(
        cl_loglik_with_c, Active,
        (copy(A_cl_e), Duplicated),
        (copy(C_cl), Duplicated),
        (copy(u0_cl_e), Duplicated),
        ([copy(yi) for yi in y_cl_e], Duplicated),
        (copy(H_cl_e), Duplicated),
        (deepcopy(ws_c.output), Duplicated),
        (deepcopy(ws_c.cache), Duplicated);
        fdm = _fdm_cl,
    )
end

# =============================================================================
# Reverse — non-diagonal R via vech
# =============================================================================

@testset "EnzymeTestUtils — CL reverse non-diagonal R (vech)" begin
    H_offdiag = [0.1 0.05; 0.02 0.08]
    R_offdiag = H_offdiag * H_offdiag'
    v0 = make_vech_for(R_offdiag)

    prob_v = LinearStateSpaceProblem(
        A_cl_e, nothing, u0_cl_e, (0, T_cl_e);
        observables_noise = R_offdiag, observables = y_cl_e
    )
    ws_v = init(prob_v, ConditionalLikelihood())

    test_reverse(
        cl_loglik_vech, Active,
        (copy(A_cl_e), Duplicated),
        (copy(v0), Duplicated),
        (copy(u0_cl_e), Duplicated),
        ([copy(yi) for yi in y_cl_e], Duplicated),
        (deepcopy(ws_v.output), Duplicated),
        (deepcopy(ws_v.cache), Duplicated);
        fdm = _fdm_cl,
    )
end
