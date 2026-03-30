# Enzyme AD tests for DirectIteration
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
const _fdm_di = central_fdm(5, 1; max_range = 1.0e-3)

# --- Test setup ---

const N_di = 3; const M_di = 2; const K_di = 2; const L_di = 2; const T_di = 5

Random.seed!(42)
A_raw_di = randn(N_di, N_di)
const A_di = 0.5 * A_raw_di / maximum(abs.(eigvals(A_raw_di)))
const B_di = 0.1 * randn(N_di, K_di)
const C_di = randn(M_di, N_di)
const H_di = 0.1 * randn(M_di, L_di)
const u0_di = zeros(N_di)

Random.seed!(123)
const noise_di = [randn(K_di) for _ in 1:T_di]
const obs_noise_di = [randn(L_di) for _ in 1:T_di]
const sim_sol_di = solve(
    LinearStateSpaceProblem(
        A_di, B_di, u0_di, (0, T_di); C = C_di, noise = noise_di
    )
)
const y_di = [sim_sol_di.z[t + 1] + H_di * obs_noise_di[t] for t in 1:T_di]

# --- Helpers ---

function make_di_prob(A, B, C, u0, noise, y, H)
    R = H * H'
    return LinearStateSpaceProblem(
        A, B, u0, (0, length(y));
        C, observables_noise = R, observables = y, noise
    )
end

function make_di_sol_cache(A, B, C, u0, noise, y, H)
    ws = init(make_di_prob(A, B, C, u0, noise, y, H), DirectIteration())
    return ws.output, ws.cache
end

# --- Wrappers — prob as single Duplicated arg ---

function di_solve_prob!(prob, sol, cache)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return (sol.u, sol.z)
end

function di_loglik_prob(prob, sol, cache)::Float64
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    return solve!(ws).logpdf
end

# Scalar wrappers for reverse mode (prob pattern)
function di_z_sum_prob(prob, sol, cache)::Float64
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return sol.z[2][1] + sol.z[3][2]
end

function di_u_sum_prob(prob, sol, cache)::Float64
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return sol.u[2][1] + sol.u[3][2]
end

# Vech: separate args (y stays Duplicated — remake doesn't work with Enzyme shadows)
function di_solve_vech!(A, B, C, u0, noise, y, r_v, n_obs, sol, cache)
    prob = LinearStateSpaceProblem(
        A, B, u0, (0, length(y));
        C, observables_noise = make_posdef_from_vech(r_v, n_obs), observables = y, noise
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return (sol.u, sol.z)
end

function di_loglik_vech(A, B, C, u0, noise, y, r_v, n_obs, sol, cache)::Float64
    prob = LinearStateSpaceProblem(
        A, B, u0, (0, length(y));
        C, observables_noise = make_posdef_from_vech(r_v, n_obs), observables = y, noise
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    return solve!(ws).logpdf
end

# --- Sanity test ---

@testset "DirectIteration loglik via solve!() - sanity" begin
    prob = make_di_prob(A_di, B_di, C_di, u0_di, noise_di, y_di, H_di)
    ws = init(prob, DirectIteration())
    loglik = di_loglik_prob(prob, ws.output, ws.cache)
    @test isfinite(loglik)

    loglik2 = di_loglik_prob(prob, ws.output, ws.cache)
    @test loglik ≈ loglik2 rtol = 1.0e-12
end

# --- Forward — prob as Duplicated ---

@testset "EnzymeTestUtils - DirectIteration forward (prob Duplicated)" begin
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]; H_s = [0.1 0.0; 0.0 0.1]
    u0_s = zeros(2); noise_s = [[0.1, -0.1], [0.2, 0.05]]
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    prob = make_di_prob(A_s, B_s, C_s, u0_s, noise_s, y_s, H_s)
    ws = init(prob, DirectIteration())

    test_forward(
        di_solve_prob!, Const,
        (prob, Duplicated),
        (ws.output, Duplicated), (ws.cache, Duplicated);
        fdm = _fdm_di,
    )
end

# --- Reverse — prob as Duplicated (logpdf) ---

@testset "EnzymeTestUtils - DirectIteration reverse (prob Duplicated, logpdf)" begin
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]; H_s = [0.1 0.0; 0.0 0.1]
    u0_s = zeros(2); noise_s = [[0.1, -0.1], [0.2, 0.05]]
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    prob = make_di_prob(A_s, B_s, C_s, u0_s, noise_s, y_s, H_s)
    ws = init(prob, DirectIteration())

    test_reverse(
        di_loglik_prob, Active,
        (prob, Duplicated),
        (deepcopy(ws.output), Duplicated), (deepcopy(ws.cache), Duplicated);
        fdm = _fdm_di,
    )
end

# --- Forward — rectangular H (prob as Duplicated) ---

@testset "EnzymeTestUtils - DirectIteration rectangular H forward (prob Duplicated)" begin
    A_r = [0.5 0.1 0.0; -0.1 0.5 0.05; 0.02 -0.05 0.5]
    B_r = 0.1 * [1.0 0.5; 0.3 -0.2; 0.7 0.1]
    C_r = [1.0 0.0 0.5; 0.0 1.0 0.0]
    H_r = 0.1 * [1.0 0.5 0.3; -0.2 0.7 0.1]
    u0_r = zeros(3); noise_r = [[0.1, -0.1], [0.2, 0.05]]
    y_r = [[0.5, 0.3], [0.2, -0.1]]
    prob = make_di_prob(A_r, B_r, C_r, u0_r, noise_r, y_r, H_r)
    ws = init(prob, DirectIteration())

    test_forward(
        di_solve_prob!, Const,
        (prob, Duplicated),
        (ws.output, Duplicated), (ws.cache, Duplicated);
        fdm = _fdm_di,
    )
end

# --- Non-diagonal R via vech parameterization ---

@testset "EnzymeTestUtils - DirectIteration non-diagonal R forward (vech)" begin
    _fdm_vech = central_fdm(5, 1)
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]
    u0_s = zeros(2); noise_s = [[0.1, -0.1], [0.2, 0.05]]
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    R_offdiag = [0.02 0.005; 0.005 0.01]
    r_v = make_vech_for(R_offdiag)
    sol, cache = make_di_sol_cache(
        A_s, B_s, C_s, u0_s, noise_s, y_s,
        [sqrt(0.02) 0.0; 0.0 sqrt(0.01)]
    )

    test_forward(
        di_solve_vech!, Const,
        (copy(A_s), Duplicated), (copy(B_s), Duplicated),
        (copy(C_s), Duplicated), (copy(u0_s), Duplicated),
        ([copy(n) for n in noise_s], Duplicated),
        ([copy(y) for y in y_s], Duplicated),
        (copy(r_v), Duplicated), (2, Const),
        (sol, Duplicated), (cache, Duplicated);
        fdm = _fdm_vech,
    )
end

@testset "EnzymeTestUtils - DirectIteration non-diagonal R reverse (vech)" begin
    _fdm_vech = central_fdm(5, 1)
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]
    u0_s = zeros(2); noise_s = [[0.1, -0.1], [0.2, 0.05]]
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    R_offdiag = [0.02 0.005; 0.005 0.01]
    r_v = make_vech_for(R_offdiag)
    sol, cache = make_di_sol_cache(
        A_s, B_s, C_s, u0_s, noise_s, y_s,
        [sqrt(0.02) 0.0; 0.0 sqrt(0.01)]
    )

    test_reverse(
        di_loglik_vech, Active,
        (copy(A_s), Duplicated), (copy(B_s), Duplicated),
        (copy(C_s), Duplicated), (copy(u0_s), Duplicated),
        ([copy(n) for n in noise_s], Duplicated),
        ([copy(y) for y in y_s], Duplicated),
        (copy(r_v), Duplicated), (2, Const),
        (deepcopy(sol), Duplicated), (deepcopy(cache), Duplicated);
        fdm = _fdm_vech,
    )
end

# --- Regression test ---

@testset "DirectIteration loglik - regression test" begin
    A_reg = [0.9 0.1; -0.1 0.9]; B_reg = [0.1 0.0; 0.0 0.1]
    C_reg = [1.0 0.0; 0.0 1.0]; H_reg = [0.1 0.0; 0.0 0.1]
    u0_reg = [0.0, 0.0]; noise_reg = [[0.1, -0.1], [0.2, 0.05], [0.0, 0.1]]
    y_reg = [[0.5, -0.3], [0.8, -0.1], [0.6, 0.2]]
    prob = make_di_prob(A_reg, B_reg, C_reg, u0_reg, noise_reg, y_reg, H_reg)
    ws = init(prob, DirectIteration())

    loglik = di_loglik_prob(prob, ws.output, ws.cache)
    @test isfinite(loglik)

    loglik2 = di_loglik_prob(prob, ws.output, ws.cache)
    @test loglik ≈ loglik2 rtol = 1.0e-12
end

# --- Edge-case helpers ---

function _alloc_u(u0, T)
    return [similar(u0) for _ in 1:T]
end
function _alloc_uz(u0, C, T)
    M = size(C, 1)
    return [similar(u0) for _ in 1:T], [zeros(eltype(u0), M) for _ in 1:T]
end
function _alloc_noise_cache(B, T)
    return [Vector{eltype(B)}(undef, size(B, 2)) for _ in 1:(T - 1)]
end

# No observables: B+C present, no obs/obs_noise
function di_no_obs_solve!(A, B, C, u0, noise, u_out, z_out, noise_cache)
    prob = LinearStateSpaceProblem(A, B, u0, (0, length(noise)); C, noise)
    sol = (; u = u_out, z = z_out)
    cache = (;
        noise = noise_cache, R = nothing, R_chol = nothing,
        innovation = nothing, innovation_solved = nothing,
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return (u_out, z_out)
end

# No noise: B=nothing, C present
function di_no_noise_solve!(A, C, u0, u_out, z_out, T)
    prob = LinearStateSpaceProblem(A, nothing, u0, (0, T); C)
    sol = (; u = u_out, z = z_out)
    cache = (;
        noise = nothing, R = nothing, R_chol = nothing,
        innovation = nothing, innovation_solved = nothing,
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return (u_out, z_out)
end

# No observation equation: B=nothing, C=nothing
function di_no_obs_eq_solve!(A, u0, u_out, T)
    prob = LinearStateSpaceProblem(A, nothing, u0, (0, T))
    sol = (; u = u_out, z = nothing)
    cache = (;
        noise = nothing, R = nothing, R_chol = nothing,
        innovation = nothing, innovation_solved = nothing,
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return u_out
end

# Noise but no observation equation: B present, C=nothing
function di_noise_no_obs_eq_solve!(A, B, u0, noise, u_out, noise_cache)
    prob = LinearStateSpaceProblem(A, B, u0, (0, length(noise)); noise)
    sol = (; u = u_out, z = nothing)
    cache = (;
        noise = noise_cache, R = nothing, R_chol = nothing,
        innovation = nothing, innovation_solved = nothing,
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return u_out
end

# Impulse response: B+C present, long trajectory
function di_impulse_solve!(A, B, C, u0, noise, u_out, z_out, noise_cache)
    prob = LinearStateSpaceProblem(A, B, u0, (0, length(noise)); C, noise)
    sol = (; u = u_out, z = z_out)
    cache = (;
        noise = noise_cache, R = nothing, R_chol = nothing,
        innovation = nothing, innovation_solved = nothing,
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    solve!(ws)
    return (u_out, z_out)
end

# --- Edge-case forward tests ---

@testset "EnzymeTestUtils - DirectIteration no observables forward" begin
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]
    u0_s = zeros(2); noise_s = [[0.1, -0.1], [0.2, 0.05]]
    T_a = length(noise_s) + 1
    u_out, z_out = _alloc_uz(u0_s, C_s, T_a)
    nc = _alloc_noise_cache(B_s, T_a)

    test_forward(
        di_no_obs_solve!, Const,
        (copy(A_s), Duplicated), (copy(B_s), Duplicated),
        (copy(C_s), Duplicated), (copy(u0_s), Duplicated),
        ([copy(n) for n in noise_s], Duplicated),
        (u_out, Duplicated), (z_out, Duplicated), (nc, Duplicated);
        fdm = _fdm_di,
    )
end

@testset "EnzymeTestUtils - DirectIteration no noise forward" begin
    A_s = [0.8 0.1; -0.1 0.7]; C_s = [1.0 0.0; 0.0 1.0]
    u0_s = [0.5, -0.3]; T_val = 3
    u_out, z_out = _alloc_uz(u0_s, C_s, T_val + 1)

    test_forward(
        di_no_noise_solve!, Const,
        (copy(A_s), Duplicated), (copy(C_s), Duplicated),
        (copy(u0_s), Duplicated),
        (u_out, Duplicated), (z_out, Duplicated),
        (T_val, Const);
        fdm = _fdm_di,
    )
end

@testset "EnzymeTestUtils - DirectIteration no observation equation forward" begin
    A_s = [0.8 0.1; -0.1 0.7]
    u0_s = [0.5, -0.3]; T_val = 3
    u_out = _alloc_u(u0_s, T_val + 1)

    test_forward(
        di_no_obs_eq_solve!, Const,
        (copy(A_s), Duplicated), (copy(u0_s), Duplicated),
        (u_out, Duplicated), (T_val, Const);
        fdm = _fdm_di,
    )
end

@testset "EnzymeTestUtils - DirectIteration noise no observation equation forward" begin
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    u0_s = zeros(2); noise_s = [[0.1, -0.1], [0.2, 0.05]]
    T_d = length(noise_s) + 1
    u_out = _alloc_u(u0_s, T_d)
    nc = _alloc_noise_cache(B_s, T_d)

    test_forward(
        di_noise_no_obs_eq_solve!, Const,
        (copy(A_s), Duplicated), (copy(B_s), Duplicated),
        (copy(u0_s), Duplicated),
        ([copy(n) for n in noise_s], Duplicated),
        (u_out, Duplicated), (nc, Duplicated);
        fdm = _fdm_di,
    )
end

@testset "EnzymeTestUtils - DirectIteration impulse response forward" begin
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]; u0_s = zeros(2)
    noise_s = [[1.0, 0.0]]; append!(noise_s, [[0.0, 0.0] for _ in 2:10])
    T_e = length(noise_s) + 1
    u_out, z_out = _alloc_uz(u0_s, C_s, T_e)
    nc = _alloc_noise_cache(B_s, T_e)

    test_forward(
        di_impulse_solve!, Const,
        (copy(A_s), Duplicated), (copy(B_s), Duplicated),
        (copy(C_s), Duplicated), (copy(u0_s), Duplicated),
        ([copy(n) for n in noise_s], Duplicated),
        (u_out, Duplicated), (z_out, Duplicated), (nc, Duplicated);
        fdm = _fdm_di,
    )
end

# --- Reverse: z_sum and u_sum (prob as Duplicated) ---

@testset "EnzymeTestUtils - DirectIteration z_sum reverse (prob Duplicated)" begin
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]; H_s = [0.1 0.0; 0.0 0.1]
    u0_s = zeros(2); noise_s = [[0.1, -0.1], [0.2, 0.05]]
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    prob = make_di_prob(A_s, B_s, C_s, u0_s, noise_s, y_s, H_s)
    ws = init(prob, DirectIteration())

    test_reverse(
        di_z_sum_prob, Active,
        (prob, Duplicated),
        (deepcopy(ws.output), Duplicated), (deepcopy(ws.cache), Duplicated);
        fdm = _fdm_di,
    )
end

@testset "EnzymeTestUtils - DirectIteration u_sum reverse (prob Duplicated)" begin
    A_s = [0.8 0.1; -0.1 0.7]; B_s = [0.1 0.0; 0.0 0.1]
    C_s = [1.0 0.0; 0.0 1.0]; H_s = [0.1 0.0; 0.0 0.1]
    u0_s = zeros(2); noise_s = [[0.1, -0.1], [0.2, 0.05]]
    y_s = [[0.5, 0.3], [0.2, 0.1]]
    prob = make_di_prob(A_s, B_s, C_s, u0_s, noise_s, y_s, H_s)
    ws = init(prob, DirectIteration())

    test_reverse(
        di_u_sum_prob, Active,
        (prob, Duplicated),
        (deepcopy(ws.output), Duplicated), (deepcopy(ws.cache), Duplicated);
        fdm = _fdm_di,
    )
end

GC.enable(true)
