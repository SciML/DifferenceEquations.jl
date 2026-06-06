using DifferenceEquations, LinearAlgebra, Test, Random, DelimitedFiles, DiffEqBase
using DifferenceEquations: init, solve!

# =============================================================================
# Small random test data (N=2, K=1, M=2, T=5)
# =============================================================================

Random.seed!(99)
const N_q = 2; const K_q = 1; const M_q = 2; const T_q = 5

const A_0_sm = 0.01 * randn(N_q)
const A_1_sm_raw = randn(N_q, N_q)
const A_1_sm = 0.5 * A_1_sm_raw / maximum(abs.(eigvals(A_1_sm_raw)))
const A_2_sm = 0.01 * randn(N_q, N_q, N_q)
const B_sm = 0.1 * randn(N_q, K_q)
const C_0_sm = 0.01 * randn(M_q)
const C_1_sm = randn(M_q, N_q)
const C_2_sm = 0.01 * randn(M_q, N_q, N_q)
const D_sm = abs2.([0.1, 0.1])
const u0_sm = zeros(N_q)

Random.seed!(200)
const noise_sm = [randn(K_q) for _ in 1:T_q]

# Pre-simulate observations for logpdf tests
Random.seed!(300)
const sim_unpruned = solve(
    QuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm, noise = noise_sm
    )
)
const obs_sm = [sim_unpruned.z[t + 1] + 0.05 * randn(M_q) for t in 1:T_q]

# =============================================================================
# Unpruned QuadraticStateSpaceProblem tests
# =============================================================================

@testset "Unpruned simulation (no obs) — finite and solve! matches solve" begin
    Random.seed!(1234)
    prob = QuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm
    )
    Random.seed!(1234)
    sol = solve(prob)
    @test all(all(isfinite, u) for u in sol.u)
    @test all(all(isfinite, z) for z in sol.z)
    @test sol.logpdf == 0.0

    # solve! matches solve
    Random.seed!(1234)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol.u
    @test sol_ws.z ≈ sol.z
    @test sol_ws.logpdf ≈ sol.logpdf
end

@testset "Unpruned with observations + obs_noise — logpdf finite" begin
    prob = QuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm,
        noise = noise_sm, observables = obs_sm, observables_noise = Diagonal(D_sm)
    )
    sol = solve(prob)
    @test isfinite(sol.logpdf)
    @test sol.logpdf != 0.0
end

@testset "Unpruned no noise (B=nothing) — deterministic" begin
    u0_det = [0.5, -0.3]
    prob = QuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, nothing, u0_det, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm
    )
    sol1 = solve(prob)
    sol2 = solve(prob)
    @test sol1.u ≈ sol2.u
    @test sol1.z ≈ sol2.z
    @test sol1.W === nothing
    @test sol2.W === nothing
end

@testset "Unpruned C=nothing — no observation process" begin
    Random.seed!(1234)
    prob = QuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q)
    )
    sol = solve(prob)
    @test sol.z === nothing
    @test all(all(isfinite, u) for u in sol.u)
    @test sol.logpdf == 0.0
end

# =============================================================================
# Pruned PrunedQuadraticStateSpaceProblem tests
# =============================================================================

# Pre-simulate observations for pruned logpdf tests
Random.seed!(400)
const sim_pruned = solve(
    PrunedQuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm, noise = noise_sm
    )
)
const obs_pruned_sm = [sim_pruned.z[t + 1] + 0.05 * randn(M_q) for t in 1:T_q]

@testset "Pruned simulation (no obs) — finite and solve! matches solve" begin
    Random.seed!(1234)
    prob = PrunedQuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm
    )
    Random.seed!(1234)
    sol = solve(prob)
    @test all(all(isfinite, u) for u in sol.u)
    @test all(all(isfinite, z) for z in sol.z)
    @test sol.logpdf == 0.0

    # solve! matches solve
    Random.seed!(1234)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol.u
    @test sol_ws.z ≈ sol.z
    @test sol_ws.logpdf ≈ sol.logpdf
end

@testset "Pruned with observations + obs_noise — logpdf finite" begin
    prob = PrunedQuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm,
        noise = noise_sm, observables = obs_pruned_sm, observables_noise = Diagonal(D_sm)
    )
    sol = solve(prob)
    @test isfinite(sol.logpdf)
    @test sol.logpdf != 0.0
end

@testset "Pruned no noise (B=nothing) — deterministic" begin
    u0_det = [0.5, -0.3]
    prob = PrunedQuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, nothing, u0_det, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm
    )
    sol1 = solve(prob)
    sol2 = solve(prob)
    @test sol1.u ≈ sol2.u
    @test sol1.z ≈ sol2.z
    @test sol1.W === nothing
end

@testset "Pruned C=nothing — no observation process" begin
    Random.seed!(1234)
    prob = PrunedQuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q)
    )
    sol = solve(prob)
    @test sol.z === nothing
    @test all(all(isfinite, u) for u in sol.u)
    @test sol.logpdf == 0.0
end

# =============================================================================
# Regression: PrunedQuadraticStateSpaceProblem matches old closure-based value
# =============================================================================

# RBC quadratic data (from test/direct_iteration.jl)
A_0_rbc = [-7.824904812740593e-5, 0.0]
A_1_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
A_2_rbc = cat(
    [-0.00019761505863889124 0.03375055315837927; 0.0 0.0],
    [0.03375055315837913 3.128758481817603; 0.0 0.0]; dims = 3
)
B_2_rbc = reshape([0.0; -0.01], 2, 1)
C_0_rbc = [7.824904812740593e-5, 0.0]
C_1_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
C_2_rbc = cat(
    [-0.00018554166974717046 0.0025652363153049716; 0.0 0.0],
    [0.002565236315304951 0.3132705036896446; 0.0 0.0]; dims = 3
)
D_2_rbc = abs2.([0.1, 0.1])
u0_2_rbc = zeros(2)

observables_2_rbc_matrix = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ','
)' |> collect
observables_2_rbc = [observables_2_rbc_matrix[:, t] for t in 1:size(observables_2_rbc_matrix, 2)]
noise_2_rbc_matrix = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ','
)' |> collect
noise_2_rbc = [noise_2_rbc_matrix[:, t] for t in 1:size(noise_2_rbc_matrix, 2)]
T_rbc = 5
observables_2_rbc_short = observables_2_rbc[1:T_rbc]
noise_2_rbc_short = noise_2_rbc[1:T_rbc]

@testset "Pruned RBC regression — matches closure-based quadratic_joint_likelihood" begin
    prob = PrunedQuadraticStateSpaceProblem(
        A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, u0_2_rbc,
        (0, length(observables_2_rbc_short));
        C_0 = C_0_rbc, C_1 = C_1_rbc, C_2 = C_2_rbc,
        observables_noise = Diagonal(D_2_rbc), noise = noise_2_rbc_short,
        observables = observables_2_rbc_short
    )
    sol = solve(prob)
    @test sol.logpdf ≈ -690.81094364573
end

@testset "Pruned RBC — solve! matches solve" begin
    prob = PrunedQuadraticStateSpaceProblem(
        A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, u0_2_rbc,
        (0, length(observables_2_rbc_short));
        C_0 = C_0_rbc, C_1 = C_1_rbc, C_2 = C_2_rbc,
        observables_noise = Diagonal(D_2_rbc), noise = noise_2_rbc_short,
        observables = observables_2_rbc_short
    )
    sol_direct = solve(prob)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.logpdf ≈ sol_direct.logpdf
end

# =============================================================================
# Workspace (init/solve!) additional tests
# =============================================================================

@testset "Unpruned solve!() repeated — idempotent" begin
    prob = QuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm,
        noise = noise_sm, observables = obs_sm, observables_noise = Diagonal(D_sm)
    )
    ws = init(prob, DirectIteration())
    sol1 = solve!(ws)
    sol2 = solve!(ws)
    @test sol1.u ≈ sol2.u
    @test sol1.z ≈ sol2.z
    @test sol1.logpdf ≈ sol2.logpdf
end

@testset "Pruned solve!() repeated — idempotent" begin
    prob = PrunedQuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, B_sm, u0_sm, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm,
        noise = noise_sm, observables = obs_pruned_sm, observables_noise = Diagonal(D_sm)
    )
    ws = init(prob, DirectIteration())
    sol1 = solve!(ws)
    sol2 = solve!(ws)
    @test sol1.u ≈ sol2.u
    @test sol1.z ≈ sol2.z
    @test sol1.logpdf ≈ sol2.logpdf
end

@testset "Unpruned solve!() — no obs, B=nothing" begin
    u0_det = [0.5, -0.3]
    prob = QuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, nothing, u0_det, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm
    )
    sol_direct = solve(prob)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.W === nothing
end

@testset "Pruned solve!() — no obs, B=nothing" begin
    u0_det = [0.5, -0.3]
    prob = PrunedQuadraticStateSpaceProblem(
        A_0_sm, A_1_sm, A_2_sm, nothing, u0_det, (0, T_q);
        C_0 = C_0_sm, C_1 = C_1_sm, C_2 = C_2_sm
    )
    sol_direct = solve(prob)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.W === nothing
end
