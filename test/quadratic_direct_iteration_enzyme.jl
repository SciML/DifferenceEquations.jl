using LinearAlgebra, Test, Enzyme, EnzymeTestUtils, Random
using DifferenceEquations
using DifferenceEquations: init, solve!, StateSpaceWorkspace

# =============================================================================
# Small test data (N=2, K=1, M=2, T=2)
# =============================================================================

Random.seed!(77)
const N_qe = 2; const K_qe = 1; const M_qe = 2; const T_qe = 2

const A_0_qe = 0.01 * randn(N_qe)
const A_1_qe_raw = randn(N_qe, N_qe)
const A_1_qe = 0.5 * A_1_qe_raw / maximum(abs.(eigvals(A_1_qe_raw)))
const A_2_qe = 0.01 * randn(N_qe, N_qe, N_qe)
const B_qe = 0.1 * randn(N_qe, K_qe)
const C_0_qe = 0.01 * randn(M_qe)
const C_1_qe = randn(M_qe, N_qe)
const C_2_qe = 0.01 * randn(M_qe, N_qe, N_qe)
const u0_qe = zeros(N_qe)
const noise_qe = [0.1 * randn(K_qe) for _ in 1:T_qe]

# =============================================================================
# Helper: allocate sol/cache from init
# =============================================================================

function make_quad_sol_cache(A_0, A_1, A_2, B, u0, noise; C_0, C_1, C_2)
    prob = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = init(prob, DirectIteration())
    return ws.output, ws.cache
end

function make_pruned_sol_cache(A_0, A_1, A_2, B, u0, noise; C_0, C_1, C_2)
    prob = PrunedQuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = init(prob, DirectIteration())
    return ws.output, ws.cache
end

# =============================================================================
# Unpruned wrapper functions
# =============================================================================

function quad_solve!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol, cache)
    prob = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return (sol.u, sol.z)
end

function quad_scalar!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol, cache)::Float64
    prob = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    return sum(solve!(ws).u[end])
end

# =============================================================================
# Pruned wrapper functions
# =============================================================================

function pruned_solve!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol, cache)
    prob = PrunedQuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return (sol.u, sol.z)
end

function pruned_scalar!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol, cache)::Float64
    prob = PrunedQuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    return sum(solve!(ws).u[end])
end

# =============================================================================
# Sanity tests
# =============================================================================

@testset "Unpruned quadratic solve! sanity" begin
    sol, cache = make_quad_sol_cache(
        A_0_qe, A_1_qe, A_2_qe, B_qe, u0_qe, noise_qe;
        C_0 = C_0_qe, C_1 = C_1_qe, C_2 = C_2_qe)
    val = quad_scalar!(
        A_0_qe, A_1_qe, A_2_qe, B_qe, C_0_qe, C_1_qe, C_2_qe,
        u0_qe, noise_qe, sol, cache)
    @test isfinite(val)

    val2 = quad_scalar!(
        A_0_qe, A_1_qe, A_2_qe, B_qe, C_0_qe, C_1_qe, C_2_qe,
        u0_qe, noise_qe, sol, cache)
    @test val ≈ val2 rtol = 1e-12
end

@testset "Pruned quadratic solve! sanity" begin
    sol, cache = make_pruned_sol_cache(
        A_0_qe, A_1_qe, A_2_qe, B_qe, u0_qe, noise_qe;
        C_0 = C_0_qe, C_1 = C_1_qe, C_2 = C_2_qe)
    val = pruned_scalar!(
        A_0_qe, A_1_qe, A_2_qe, B_qe, C_0_qe, C_1_qe, C_2_qe,
        u0_qe, noise_qe, sol, cache)
    @test isfinite(val)

    val2 = pruned_scalar!(
        A_0_qe, A_1_qe, A_2_qe, B_qe, C_0_qe, C_1_qe, C_2_qe,
        u0_qe, noise_qe, sol, cache)
    @test val ≈ val2 rtol = 1e-12
end

# =============================================================================
# Unpruned forward (all Duplicated)
# =============================================================================

@testset "EnzymeTestUtils - Unpruned quadratic forward (all Duplicated)" begin
    sol, cache = make_quad_sol_cache(
        A_0_qe, A_1_qe, A_2_qe, B_qe, u0_qe, noise_qe;
        C_0 = C_0_qe, C_1 = C_1_qe, C_2 = C_2_qe)

    test_forward(quad_solve!, Const,
        (copy(A_0_qe), Duplicated), (copy(A_1_qe), Duplicated),
        (copy(A_2_qe), Duplicated), (copy(B_qe), Duplicated),
        (copy(C_0_qe), Duplicated), (copy(C_1_qe), Duplicated),
        (copy(C_2_qe), Duplicated), (copy(u0_qe), Duplicated),
        ([copy(n) for n in noise_qe], Duplicated),
        (sol, Duplicated), (cache, Duplicated))
end

# =============================================================================
# Unpruned reverse (scalar sum(u[end]), all Duplicated)
# =============================================================================

@testset "EnzymeTestUtils - Unpruned quadratic reverse (scalar, all Duplicated)" begin
    sol, cache = make_quad_sol_cache(
        A_0_qe, A_1_qe, A_2_qe, B_qe, u0_qe, noise_qe;
        C_0 = C_0_qe, C_1 = C_1_qe, C_2 = C_2_qe)

    # Same cache gradient FD mismatch as linear reverse tests
    @test_broken test_reverse(quad_scalar!, Active,
        (copy(A_0_qe), Duplicated), (copy(A_1_qe), Duplicated),
        (copy(A_2_qe), Duplicated), (copy(B_qe), Duplicated),
        (copy(C_0_qe), Duplicated), (copy(C_1_qe), Duplicated),
        (copy(C_2_qe), Duplicated), (copy(u0_qe), Duplicated),
        ([copy(n) for n in noise_qe], Duplicated),
        (sol, Duplicated), (cache, Duplicated)) === nothing
end

# =============================================================================
# Pruned forward (all Duplicated)
# =============================================================================

@testset "EnzymeTestUtils - Pruned quadratic forward (all Duplicated)" begin
    sol, cache = make_pruned_sol_cache(
        A_0_qe, A_1_qe, A_2_qe, B_qe, u0_qe, noise_qe;
        C_0 = C_0_qe, C_1 = C_1_qe, C_2 = C_2_qe)

    test_forward(pruned_solve!, Const,
        (copy(A_0_qe), Duplicated), (copy(A_1_qe), Duplicated),
        (copy(A_2_qe), Duplicated), (copy(B_qe), Duplicated),
        (copy(C_0_qe), Duplicated), (copy(C_1_qe), Duplicated),
        (copy(C_2_qe), Duplicated), (copy(u0_qe), Duplicated),
        ([copy(n) for n in noise_qe], Duplicated),
        (sol, Duplicated), (cache, Duplicated))
end

# =============================================================================
# Pruned reverse (scalar sum(u[end]), all Duplicated)
# =============================================================================

@testset "EnzymeTestUtils - Pruned quadratic reverse (scalar, all Duplicated)" begin
    sol, cache = make_pruned_sol_cache(
        A_0_qe, A_1_qe, A_2_qe, B_qe, u0_qe, noise_qe;
        C_0 = C_0_qe, C_1 = C_1_qe, C_2 = C_2_qe)

    # Same cache gradient FD mismatch as linear reverse tests
    @test_broken test_reverse(pruned_scalar!, Active,
        (copy(A_0_qe), Duplicated), (copy(A_1_qe), Duplicated),
        (copy(A_2_qe), Duplicated), (copy(B_qe), Duplicated),
        (copy(C_0_qe), Duplicated), (copy(C_1_qe), Duplicated),
        (copy(C_2_qe), Duplicated), (copy(u0_qe), Duplicated),
        ([copy(n) for n in noise_qe], Duplicated),
        (sol, Duplicated), (cache, Duplicated)) === nothing
end
