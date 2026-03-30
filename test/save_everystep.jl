# Tests for save_everystep=false: endpoints-only solve with correct logpdf.
# Verifies that sol.u[1]=initial, sol.u[2]=final, logpdf matches full solve.

using DifferenceEquations, Distributions, LinearAlgebra, Test, Random, ForwardDiff
using DifferenceEquations: init, solve!
using StaticArrays

# =============================================================================
# Shared test data
# =============================================================================

const A_se = [0.8 0.1; -0.1 0.7]
const B_se = [0.1 0.0; 0.0 0.1]
const C_se = [1.0 0.0; 0.0 1.0]
const u0_se = zeros(2)
const T_se = 10

Random.seed!(42)
const noise_se = [randn(2) for _ in 1:T_se]
const y_se = [randn(2) for _ in 1:T_se]

# =============================================================================
# Primal simulation: DirectIteration
# =============================================================================

@testset "save_everystep=false — DI simulation with C and noise" begin
    prob = LinearStateSpaceProblem(A_se, B_se, u0_se, (0, T_se); C = C_se, noise = noise_se)
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test length(sol_ep.z) == 2
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
    @test sol_ep.z[1] ≈ sol_full.z[1]
    @test sol_ep.z[2] ≈ sol_full.z[end]
    @test sol_ep.logpdf == 0.0
end

@testset "save_everystep=false — DI simulation C=nothing" begin
    prob = LinearStateSpaceProblem(A_se, B_se, u0_se, (0, T_se); noise = noise_se)
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test sol_ep.z === nothing
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
end

@testset "save_everystep=false — DI simulation B=nothing" begin
    prob = LinearStateSpaceProblem(A_se, nothing, [1.0, 0.5], (0, T_se); C = C_se)
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
    @test sol_ep.z[1] ≈ sol_full.z[1]
    @test sol_ep.z[2] ≈ sol_full.z[end]
end

@testset "save_everystep=false — DI with obs noise simulation" begin
    prob = LinearStateSpaceProblem(
        A_se, B_se, u0_se, (0, T_se);
        C = C_se, noise = noise_se, observables_noise = Diagonal([0.01, 0.01])
    )
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test length(sol_ep.z) == 2
    # u endpoints are deterministic (noise is fixed), but z has random obs noise
    # so we can only check u matches, not z (different random draws for 2 vs T+1 elements)
    sol_no_obs_noise = solve(
        LinearStateSpaceProblem(A_se, B_se, u0_se, (0, T_se); C = C_se, noise = noise_se);
        save_everystep = false
    )
    @test sol_ep.u[1] ≈ sol_no_obs_noise.u[1]
    @test sol_ep.u[2] ≈ sol_no_obs_noise.u[2]
end

# =============================================================================
# Primal simulation: Generic StateSpaceProblem
# =============================================================================

@testset "save_everystep=false — Generic DI simulation" begin
    f!! = (x_next, x, w, p, t) -> begin
        mul!(x_next, p.A, x)
        mul!(x_next, p.B, w, 1.0, 1.0)
        return x_next
    end
    g!! = (y, x, p, t) -> begin
        mul!(y, p.C, x)
        return y
    end
    p = (; A = A_se, B = B_se, C = C_se)

    prob = StateSpaceProblem(
        f!!, g!!, u0_se, (0, T_se), p;
        n_shocks = 2, n_obs = 2, noise = noise_se
    )
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
    @test sol_ep.z[1] ≈ sol_full.z[1]
    @test sol_ep.z[2] ≈ sol_full.z[end]
end

# =============================================================================
# Likelihood: DirectIteration
# =============================================================================

@testset "save_everystep=false — DI likelihood" begin
    prob = LinearStateSpaceProblem(
        A_se, B_se, u0_se, (0, T_se); C = C_se,
        observables_noise = Diagonal([0.01, 0.01]),
        observables = y_se, noise = noise_se
    )
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test sol_ep.logpdf ≈ sol_full.logpdf
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
end

# =============================================================================
# Likelihood: ConditionalLikelihood
# =============================================================================

@testset "save_everystep=false — CL AR(1)" begin
    rho = 0.8; sigma_e = 0.5; T_cl = 20
    Random.seed!(111)
    y_cl = [randn(1) for _ in 1:T_cl]

    prob = LinearStateSpaceProblem(
        fill(rho, 1, 1), nothing, [0.0], (0, T_cl);
        observables = y_cl, observables_noise = Diagonal([sigma_e^2])
    )
    sol_full = solve(prob, ConditionalLikelihood())
    sol_ep = solve(prob, ConditionalLikelihood(); save_everystep = false)

    @test sol_ep.logpdf ≈ sol_full.logpdf
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
end

@testset "save_everystep=false — CL VAR(1)" begin
    T_cl = 15
    Random.seed!(222)
    y_cl = [randn(2) for _ in 1:T_cl]

    prob = LinearStateSpaceProblem(
        A_se, nothing, u0_se, (0, T_cl);
        observables = y_cl, observables_noise = Diagonal([0.25, 0.25])
    )
    sol_full = solve(prob, ConditionalLikelihood())
    sol_ep = solve(prob, ConditionalLikelihood(); save_everystep = false)

    @test sol_ep.logpdf ≈ sol_full.logpdf
end

@testset "save_everystep=false — CL generic nonlinear" begin
    rho = 0.8; alpha = 0.05; sigma_e = 0.3; T_cl = 15
    Random.seed!(333)
    y_cl = [randn(1) for _ in 1:T_cl]

    f!! = (x_next, x, w, p, t) -> begin
        val = p.rho * x[1] + p.alpha * x[1]^2
        if ismutable(x_next)
            x_next[1] = val
            return x_next
        else
            return typeof(x)(val)
        end
    end

    prob = StateSpaceProblem(
        f!!, nothing, [0.0], (0, T_cl), (; rho, alpha);
        n_shocks = 0, n_obs = 0,
        observables = y_cl, observables_noise = Diagonal([sigma_e^2])
    )
    sol_full = solve(prob, ConditionalLikelihood())
    sol_ep = solve(prob, ConditionalLikelihood(); save_everystep = false)

    @test sol_ep.logpdf ≈ sol_full.logpdf
end

# =============================================================================
# Likelihood: KalmanFilter
# =============================================================================

@testset "save_everystep=false — KF" begin
    Random.seed!(444)
    y_kf = [randn(2) for _ in 1:T_se]

    prob = LinearStateSpaceProblem(
        A_se, B_se, u0_se, (0, T_se); C = C_se,
        observables_noise = Diagonal([0.01, 0.01]), observables = y_kf,
        u0_prior_mean = zeros(2), u0_prior_var = Matrix(1.0 * I(2))
    )
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test length(sol_ep.P) == 2
    @test length(sol_ep.z) == 2
    @test sol_ep.logpdf ≈ sol_full.logpdf
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
    @test sol_ep.P[1] ≈ sol_full.P[1]
    @test sol_ep.P[2] ≈ sol_full.P[end]
    @test sol_ep.z[1] ≈ sol_full.z[1]
    @test sol_ep.z[2] ≈ sol_full.z[end]
end

# =============================================================================
# Quadratic models
# =============================================================================

@testset "save_everystep=false — Unpruned quadratic simulation" begin
    A_0 = [0.0, 0.0]
    A_1 = A_se
    A_2 = zeros(2, 2, 2)
    A_2[1, 1, 1] = 0.01

    prob = QuadraticStateSpaceProblem(
        A_0, A_1, A_2, B_se, u0_se, (0, T_se);
        C_0 = [0.0, 0.0], C_1 = C_se, C_2 = zeros(2, 2, 2), noise = noise_se
    )
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
    @test sol_ep.z[1] ≈ sol_full.z[1]
    @test sol_ep.z[2] ≈ sol_full.z[end]
end

@testset "save_everystep=false — Pruned quadratic simulation" begin
    A_0 = [0.0, 0.0]
    A_1 = A_se
    A_2 = zeros(2, 2, 2)
    A_2[1, 1, 1] = 0.01

    prob = PrunedQuadraticStateSpaceProblem(
        A_0, A_1, A_2, B_se, u0_se, (0, T_se);
        C_0 = [0.0, 0.0], C_1 = C_se, C_2 = zeros(2, 2, 2), noise = noise_se
    )
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
    @test sol_ep.z[1] ≈ sol_full.z[1]
    @test sol_ep.z[2] ≈ sol_full.z[end]
end

# =============================================================================
# StaticArrays
# =============================================================================

@testset "save_everystep=false — StaticArrays DI simulation" begin
    A_s = SMatrix{2, 2}(A_se)
    B_s = SMatrix{2, 2}(B_se)
    C_s = SMatrix{2, 2}(C_se)
    u0_s = SVector{2}(u0_se)
    noise_s = [SVector{2}(n) for n in noise_se]

    prob = LinearStateSpaceProblem(A_s, B_s, u0_s, (0, T_se); C = C_s, noise = noise_s)
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test length(sol_ep.u) == 2
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
end

@testset "save_everystep=false — StaticArrays CL" begin
    T_cl = 10
    y_s = [SVector{2}(randn(2)) for _ in 1:T_cl]

    prob = LinearStateSpaceProblem(
        SMatrix{2, 2}(A_se), nothing, SVector{2}(u0_se), (0, T_cl);
        observables = y_s, observables_noise = Diagonal(SVector{2}(0.25, 0.25))
    )
    sol_full = solve(prob, ConditionalLikelihood())
    sol_ep = solve(prob, ConditionalLikelihood(); save_everystep = false)

    @test sol_ep.logpdf ≈ sol_full.logpdf
end

@testset "save_everystep=false — StaticArrays KF" begin
    T_kf = 10
    Random.seed!(555)
    y_s = [SVector{2}(randn(2)) for _ in 1:T_kf]

    prob = LinearStateSpaceProblem(
        SMatrix{2, 2}(A_se), SMatrix{2, 2}(B_se), SVector{2}(u0_se), (0, T_kf);
        C = SMatrix{2, 2}(C_se),
        observables_noise = Diagonal(SVector{2}(0.01, 0.01)), observables = y_s,
        u0_prior_mean = SVector{2}(0.0, 0.0),
        u0_prior_var = SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0)
    )
    sol_full = solve(prob)
    sol_ep = solve(prob; save_everystep = false)

    @test sol_ep.logpdf ≈ sol_full.logpdf
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
end

# =============================================================================
# Workspace reuse
# =============================================================================

@testset "save_everystep=false — workspace init/solve! reuse" begin
    prob = LinearStateSpaceProblem(
        A_se, nothing, u0_se, (0, T_se);
        observables = y_se, observables_noise = Diagonal([0.25, 0.25])
    )
    ws = init(prob, ConditionalLikelihood(); save_everystep = false)
    sol1 = solve!(ws)
    sol2 = solve!(ws)
    @test sol1.logpdf ≈ sol2.logpdf
    @test sol1.u ≈ sol2.u
    @test length(sol1.u) == 2
end

# =============================================================================
# Edge cases
# =============================================================================

@testset "save_everystep=false — edge case T=2 (1 step)" begin
    y1 = [randn(2)]
    prob = LinearStateSpaceProblem(
        A_se, nothing, u0_se, (0, 1);
        observables = y1, observables_noise = Diagonal([0.25, 0.25])
    )
    sol_full = solve(prob, ConditionalLikelihood())
    sol_ep = solve(prob, ConditionalLikelihood(); save_everystep = false)
    @test sol_ep.logpdf ≈ sol_full.logpdf
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
end

@testset "save_everystep=false — edge case T=3 (2 steps)" begin
    y2 = [randn(2) for _ in 1:2]
    prob = LinearStateSpaceProblem(
        A_se, nothing, u0_se, (0, 2);
        observables = y2, observables_noise = Diagonal([0.25, 0.25])
    )
    sol_full = solve(prob, ConditionalLikelihood())
    sol_ep = solve(prob, ConditionalLikelihood(); save_everystep = false)
    @test sol_ep.logpdf ≈ sol_full.logpdf
    @test sol_ep.u[1] ≈ sol_full.u[1]
    @test sol_ep.u[2] ≈ sol_full.u[end]
end

# =============================================================================
# ForwardDiff gradients match
# =============================================================================

include("forwarddiff_test_utils.jl")

@testset "save_everystep=false — ForwardDiff CL gradient matches" begin
    function cl_fd(A_vec, y, se)
        T_el = eltype(A_vec)
        A = reshape(A_vec, 2, 2)
        prob = LinearStateSpaceProblem(
            A, nothing, zeros(T_el, 2), (0, length(y));
            observables = y, observables_noise = Diagonal([T_el(0.25), T_el(0.25)])
        )
        return solve(prob, ConditionalLikelihood(); save_everystep = se).logpdf
    end

    Random.seed!(777)
    y_fd = [randn(2) for _ in 1:10]
    x0 = vec(copy(A_se))

    g_true = ForwardDiff.gradient(a -> cl_fd(a, y_fd, true), x0)
    g_false = ForwardDiff.gradient(a -> cl_fd(a, y_fd, false), x0)
    @test g_true ≈ g_false
end

@testset "save_everystep=false — ForwardDiff KF gradient matches" begin
    function kf_fd(A_vec, B, C, mu0, Sigma0, R, y, se)
        T_el = eltype(A_vec)
        A = reshape(A_vec, 2, 2)
        prob = LinearStateSpaceProblem(
            A, promote_array(T_el, B), zeros(T_el, 2), (0, length(y));
            C = promote_array(T_el, C),
            observables_noise = promote_array(T_el, R), observables = y,
            u0_prior_mean = promote_array(T_el, mu0),
            u0_prior_var = promote_array(T_el, Sigma0)
        )
        return solve(prob, KalmanFilter(); save_everystep = se).logpdf
    end

    Random.seed!(888)
    y_fd = [randn(2) for _ in 1:10]
    mu0 = zeros(2)
    Sigma0 = Matrix(1.0 * I(2))
    R = Diagonal([0.01, 0.01])
    x0 = vec(copy(A_se))

    g_true = ForwardDiff.gradient(
        a -> kf_fd(a, B_se, C_se, mu0, Sigma0, R, y_fd, true), x0
    )
    g_false = ForwardDiff.gradient(
        a -> kf_fd(a, B_se, C_se, mu0, Sigma0, R, y_fd, false), x0
    )
    @test g_true ≈ g_false
end
