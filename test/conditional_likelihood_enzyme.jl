# Enzyme AD tests for ConditionalLikelihood
# prob passed as Duplicated — observables get zero shadow automatically.
# GC disabled to avoid Enzyme reverse-mode GC corruption (#2355).

GC.gc()
GC.enable(false)

using LinearAlgebra, Test, Enzyme, EnzymeTestUtils, Random
using DifferenceEquations
using DifferenceEquations: init, solve!, StateSpaceWorkspace
using FiniteDifferences: central_fdm

include("enzyme_test_utils.jl")  # vech helpers

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

# max_range needed: FD perturbation of observables_noise inside prob can push
# the matrix non-positive-definite, causing DomainError in logdet_chol.
const _fdm_cl = central_fdm(5, 1; max_range = 1.0e-3)

# =============================================================================
# Wrappers — prob as single Duplicated arg
# =============================================================================

function cl_forward_prob!(prob, sol, cache)
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol, cache)
    solve!(ws)
    return sol.u
end

function cl_loglik_prob(prob, sol, cache)::Float64
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol, cache)
    return solve!(ws).logpdf
end

# Vech: separate args (y stays Duplicated — remake doesn't work with Enzyme shadows)
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
    prob = LinearStateSpaceProblem(
        A_cl_e, nothing, u0_cl_e, (0, T_cl_e);
        observables_noise = H_cl_e * H_cl_e', observables = y_cl_e
    )
    ws = init(prob, ConditionalLikelihood())
    loglik = cl_loglik_prob(prob, ws.output, ws.cache)
    @test isfinite(loglik)
    loglik2 = cl_loglik_prob(prob, ws.output, ws.cache)
    @test loglik ≈ loglik2 rtol = 1.0e-12
end

# =============================================================================
# Forward — prob as Duplicated
# =============================================================================

@testset "EnzymeTestUtils — CL forward (prob Duplicated)" begin
    prob = LinearStateSpaceProblem(
        A_cl_e, nothing, u0_cl_e, (0, T_cl_e);
        observables_noise = H_cl_e * H_cl_e', observables = y_cl_e
    )
    ws = init(prob, ConditionalLikelihood())

    test_forward(
        cl_forward_prob!, Const,
        (prob, Duplicated),
        (ws.output, Duplicated), (ws.cache, Duplicated);
        fdm = _fdm_cl,
    )
end

# =============================================================================
# Reverse — prob as Duplicated
# =============================================================================

@testset "EnzymeTestUtils — CL reverse (prob Duplicated)" begin
    prob = LinearStateSpaceProblem(
        A_cl_e, nothing, u0_cl_e, (0, T_cl_e);
        observables_noise = H_cl_e * H_cl_e', observables = y_cl_e
    )
    ws = init(prob, ConditionalLikelihood())

    test_reverse(
        cl_loglik_prob, Active,
        (prob, Duplicated),
        (deepcopy(ws.output), Duplicated), (deepcopy(ws.cache), Duplicated);
        fdm = _fdm_cl,
    )
end

# =============================================================================
# Reverse — with C matrix
# =============================================================================

@testset "EnzymeTestUtils — CL reverse with C (prob Duplicated)" begin
    C_cl = [1.0 0.0; 0.0 1.0]
    prob = LinearStateSpaceProblem(
        A_cl_e, nothing, u0_cl_e, (0, T_cl_e);
        C = C_cl, observables_noise = H_cl_e * H_cl_e', observables = y_cl_e
    )
    ws = init(prob, ConditionalLikelihood())

    test_reverse(
        cl_loglik_prob, Active,
        (prob, Duplicated),
        (deepcopy(ws.output), Duplicated), (deepcopy(ws.cache), Duplicated);
        fdm = _fdm_cl,
    )
end

# =============================================================================
# Reverse — non-diagonal R via vech
# =============================================================================

# Vech test: separate args (y as Duplicated — can't avoid due to struct storage).
# Tighter max_range needed for vech: make_posdef_from_vech has high curvature.
@testset "EnzymeTestUtils — CL reverse non-diagonal R (vech)" begin
    H_offdiag = [0.1 0.05; 0.02 0.08]
    R_offdiag = H_offdiag * H_offdiag'
    v0 = make_vech_for(R_offdiag)
    _fdm_vech = central_fdm(5, 1)

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
        fdm = _fdm_vech,
    )
end

GC.enable(true)
