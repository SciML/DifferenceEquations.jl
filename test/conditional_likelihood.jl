# ConditionalLikelihood tests — prediction error decomposition for fully-observed models.
# Validates correctness against manual log-likelihood, type stability, workspace API,
# and StaticArrays support.

using DifferenceEquations, Distributions, LinearAlgebra, Test, Random
using DifferenceEquations: init, solve!
using StaticArrays

# =============================================================================
# AR(1) manual log-likelihood helper
# =============================================================================

function manual_ar1_loglik(y, rho, sigma_e; u0 = 0.0)
    T = length(y)
    loglik = 0.0
    x_prev = u0
    for t in 1:T
        mu = rho * x_prev
        loglik += logpdf(Normal(mu, sigma_e), y[t])
        x_prev = y[t]
    end
    return loglik
end

function manual_var_loglik(y, A, R; u0 = zeros(size(A, 1)))
    T = length(y)
    loglik = 0.0
    x_prev = u0
    M = size(R, 1)
    dist = MvNormal(zeros(M), R)
    for t in 1:T
        mu = A * x_prev
        loglik += logpdf(MvNormal(mu, R), y[t])
        x_prev = y[t]
    end
    return loglik
end

# =============================================================================
# AR(1) — C = nothing (identity observation)
# =============================================================================

@testset "ConditionalLikelihood — AR(1), C=nothing" begin
    rho = 0.8
    sigma_e = 0.5
    T = 50

    # Generate AR(1) data
    Random.seed!(123)
    y_scalar = zeros(T)
    x = 0.0
    for t in 1:T
        x = rho * x + sigma_e * randn()
        y_scalar[t] = x
    end
    y = [[yi] for yi in y_scalar]

    prob = LinearStateSpaceProblem(
        fill(rho, 1, 1), nothing, [0.0], (0, T);
        observables = y,
        observables_noise = Diagonal([sigma_e^2]),
    )
    sol = solve(prob, ConditionalLikelihood())

    expected = manual_ar1_loglik(y_scalar, rho, sigma_e)
    @test sol.logpdf ≈ expected atol = 1.0e-12
    @test sol.z === nothing
    @test length(sol.u) == T + 1
    # State should be clamped to observations
    for t in 1:T
        @test sol.u[t + 1] ≈ y[t]
    end
end

# =============================================================================
# AR(1) — C = I (explicit observation matrix, same result)
# =============================================================================

@testset "ConditionalLikelihood — AR(1), C=I" begin
    rho = 0.8
    sigma_e = 0.5
    T = 50

    Random.seed!(123)
    y_scalar = zeros(T)
    x = 0.0
    for t in 1:T
        x = rho * x + sigma_e * randn()
        y_scalar[t] = x
    end
    y = [[yi] for yi in y_scalar]

    prob_no_c = LinearStateSpaceProblem(
        fill(rho, 1, 1), nothing, [0.0], (0, T);
        observables = y,
        observables_noise = Diagonal([sigma_e^2]),
    )
    sol_no_c = solve(prob_no_c, ConditionalLikelihood())

    prob_with_c = LinearStateSpaceProblem(
        fill(rho, 1, 1), nothing, [0.0], (0, T);
        C = fill(1.0, 1, 1),
        observables = y,
        observables_noise = Diagonal([sigma_e^2]),
    )
    sol_with_c = solve(prob_with_c, ConditionalLikelihood())

    @test sol_no_c.logpdf ≈ sol_with_c.logpdf atol = 1.0e-12
    @test !isnothing(sol_with_c.z)
end

# =============================================================================
# VAR(1) — multivariate
# =============================================================================

@testset "ConditionalLikelihood — VAR(1)" begin
    A = [0.8 0.1; -0.1 0.7]
    R = Diagonal([0.25, 0.25])
    T = 30

    Random.seed!(456)
    y = Vector{Vector{Float64}}(undef, T)
    x = zeros(2)
    for t in 1:T
        x = A * x + cholesky(R).L * randn(2)
        y[t] = copy(x)
    end

    prob = LinearStateSpaceProblem(
        A, nothing, zeros(2), (0, T);
        observables = y,
        observables_noise = R,
    )
    sol = solve(prob, ConditionalLikelihood())

    expected = manual_var_loglik(y, A, R)
    @test sol.logpdf ≈ expected atol = 1.0e-10
end

# =============================================================================
# With B and noise — prediction includes noise term
# =============================================================================

@testset "ConditionalLikelihood — with B and explicit noise" begin
    rho = 0.8
    sigma_e = 0.5
    T = 20

    Random.seed!(789)
    noise = [[randn()] for _ in 1:T]
    y_scalar = zeros(T)
    x = 0.0
    B_val = 0.1
    for t in 1:T
        x = rho * x + B_val * noise[t][1]
        y_scalar[t] = x + sigma_e * randn()
    end
    y = [[yi] for yi in y_scalar]

    prob = LinearStateSpaceProblem(
        fill(rho, 1, 1), fill(B_val, 1, 1), [0.0], (0, T);
        observables = y,
        observables_noise = Diagonal([sigma_e^2]),
        noise = noise,
    )
    sol = solve(prob, ConditionalLikelihood())

    # Manual: prediction is rho * y[t-1] + B * w[t]
    loglik = 0.0
    x_prev = 0.0
    for t in 1:T
        mu = rho * x_prev + B_val * noise[t][1]
        loglik += logpdf(Normal(mu, sigma_e), y_scalar[t])
        x_prev = y_scalar[t]
    end
    @test sol.logpdf ≈ loglik atol = 1.0e-12
end

# =============================================================================
# Generic StateSpaceProblem — nonlinear AR(1)
# =============================================================================

@testset "ConditionalLikelihood — generic nonlinear AR(1)" begin
    rho = 0.8
    alpha = 0.05
    sigma_e = 0.3
    T = 30

    Random.seed!(111)
    y_scalar = zeros(T)
    x = 0.0
    for t in 1:T
        x = rho * x + alpha * x^2 + sigma_e * randn()
        y_scalar[t] = x
    end
    y = [[yi] for yi in y_scalar]

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

    p = (; rho, alpha)
    prob = StateSpaceProblem(
        nl_f!!, nothing, [0.0], (0, T), p;
        n_shocks = 0, n_obs = 0,
        observables = y,
        observables_noise = Diagonal([sigma_e^2]),
    )
    sol = solve(prob, ConditionalLikelihood())

    # Manual
    loglik = 0.0
    x_prev = 0.0
    for t in 1:T
        mu = rho * x_prev + alpha * x_prev^2
        loglik += logpdf(Normal(mu, sigma_e), y_scalar[t])
        x_prev = y_scalar[t]
    end
    @test sol.logpdf ≈ loglik atol = 1.0e-12
end

# =============================================================================
# QuadraticStateSpaceProblem
# =============================================================================

@testset "ConditionalLikelihood — QuadraticStateSpaceProblem" begin
    rho = 0.8
    alpha = 0.05
    sigma_e = 0.3
    T = 30

    Random.seed!(111)
    y_scalar = zeros(T)
    x = 0.0
    for t in 1:T
        x = rho * x + alpha * x^2 + sigma_e * randn()
        y_scalar[t] = x
    end
    y = [[yi] for yi in y_scalar]

    A_0 = [0.0]
    A_1 = fill(rho, 1, 1)
    A_2 = fill(alpha, 1, 1, 1)

    prob = QuadraticStateSpaceProblem(
        A_0, A_1, A_2, nothing, [0.0], (0, T);
        C_0 = [0.0], C_1 = fill(1.0, 1, 1), C_2 = zeros(1, 1, 1),
        observables = y,
        observables_noise = Diagonal([sigma_e^2]),
    )
    sol = solve(prob, ConditionalLikelihood())

    # Manual — same as the generic nonlinear test
    loglik = 0.0
    x_prev = 0.0
    for t in 1:T
        mu = rho * x_prev + alpha * x_prev^2
        loglik += logpdf(Normal(mu, sigma_e), y_scalar[t])
        x_prev = y_scalar[t]
    end
    @test sol.logpdf ≈ loglik atol = 1.0e-10
end

# =============================================================================
# Type stability
# =============================================================================

@testset "ConditionalLikelihood — type stability" begin
    T = 5
    Random.seed!(42)
    y = [randn(2) for _ in 1:T]

    prob_linear = LinearStateSpaceProblem(
        [0.8 0.1; -0.1 0.7], nothing, zeros(2), (0, T);
        observables = y,
        observables_noise = Diagonal([0.25, 0.25]),
    )
    @test @inferred(solve(prob_linear, ConditionalLikelihood())) isa Any

    prob_linear_c = LinearStateSpaceProblem(
        [0.8 0.1; -0.1 0.7], nothing, zeros(2), (0, T);
        C = [1.0 0.0; 0.0 1.0],
        observables = y,
        observables_noise = Diagonal([0.25, 0.25]),
    )
    @test @inferred(solve(prob_linear_c, ConditionalLikelihood())) isa Any
end

# =============================================================================
# Workspace init/solve!
# =============================================================================

@testset "ConditionalLikelihood — solve!() matches solve()" begin
    T = 20
    rho = 0.8
    sigma_e = 0.5

    Random.seed!(321)
    y = [randn(1) for _ in 1:T]

    prob = LinearStateSpaceProblem(
        fill(rho, 1, 1), nothing, [0.0], (0, T);
        observables = y,
        observables_noise = Diagonal([sigma_e^2]),
    )
    sol_direct = solve(prob, ConditionalLikelihood())

    ws = init(prob, ConditionalLikelihood())
    sol_ws = solve!(ws)
    @test sol_ws.logpdf ≈ sol_direct.logpdf
    @test sol_ws.u ≈ sol_direct.u
end

@testset "ConditionalLikelihood — solve!() with C matrix" begin
    T = 10
    A = [0.8 0.1; -0.1 0.7]
    C = [1.0 0.0; 0.0 1.0]
    R = Diagonal([0.1, 0.1])

    Random.seed!(654)
    y = [randn(2) for _ in 1:T]

    prob = LinearStateSpaceProblem(
        A, nothing, zeros(2), (0, T);
        C = C, observables = y, observables_noise = R,
    )
    sol_direct = solve(prob, ConditionalLikelihood())

    ws = init(prob, ConditionalLikelihood())
    sol_ws = solve!(ws)
    @test sol_ws.logpdf ≈ sol_direct.logpdf
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
end

@testset "ConditionalLikelihood — solve!() repeated is idempotent" begin
    T = 10
    rho = 0.8
    sigma_e = 0.5

    Random.seed!(987)
    y = [randn(1) for _ in 1:T]

    prob = LinearStateSpaceProblem(
        fill(rho, 1, 1), nothing, [0.0], (0, T);
        observables = y,
        observables_noise = Diagonal([sigma_e^2]),
    )
    ws = init(prob, ConditionalLikelihood())
    sol1 = solve!(ws)
    sol2 = solve!(ws)
    @test sol1.logpdf ≈ sol2.logpdf
    @test sol1.u ≈ sol2.u
end

# =============================================================================
# Error handling
# =============================================================================

@testset "ConditionalLikelihood — error handling" begin
    @testset "missing observables" begin
        prob = LinearStateSpaceProblem(
            fill(0.8, 1, 1), nothing, [0.0], (0, 5);
            observables_noise = Diagonal([0.25]),
        )
        @test_throws ArgumentError solve(prob, ConditionalLikelihood())
    end

    @testset "missing observables_noise" begin
        y = [randn(1) for _ in 1:5]
        prob = LinearStateSpaceProblem(
            fill(0.8, 1, 1), nothing, [0.0], (0, 5);
            observables = y,
        )
        @test_throws ArgumentError solve(prob, ConditionalLikelihood())
    end

    @testset "observables wrong length" begin
        y = [randn(1) for _ in 1:3]
        prob = LinearStateSpaceProblem(
            fill(0.8, 1, 1), nothing, [0.0], (0, 5);
            observables = y,
            observables_noise = Diagonal([0.25]),
        )
        @test_throws ArgumentError solve(prob, ConditionalLikelihood())
    end
end

# =============================================================================
# StaticArrays
# =============================================================================

@testset "ConditionalLikelihood — StaticArrays AR(1)" begin
    rho = 0.8
    sigma_e = 0.5
    T = 20

    Random.seed!(555)
    y_scalar = zeros(T)
    x = 0.0
    for t in 1:T
        x = rho * x + sigma_e * randn()
        y_scalar[t] = x
    end

    y_mut = [[yi] for yi in y_scalar]
    y_static = [SVector{1}(yi) for yi in y_scalar]

    prob_mut = LinearStateSpaceProblem(
        fill(rho, 1, 1), nothing, [0.0], (0, T);
        observables = y_mut,
        observables_noise = Diagonal([sigma_e^2]),
    )
    sol_mut = solve(prob_mut, ConditionalLikelihood())

    prob_static = LinearStateSpaceProblem(
        SMatrix{1, 1}(rho), nothing, SVector{1}(0.0), (0, T);
        observables = y_static,
        observables_noise = Diagonal(SVector{1}(sigma_e^2)),
    )
    sol_static = solve(prob_static, ConditionalLikelihood())

    @test sol_static.logpdf ≈ sol_mut.logpdf atol = 1.0e-12
end

@testset "ConditionalLikelihood — StaticArrays VAR(1)" begin
    A = [0.8 0.1; -0.1 0.7]
    R = Diagonal([0.25, 0.25])
    T = 15

    Random.seed!(666)
    y_mut = [randn(2) for _ in 1:T]
    y_static = [SVector{2}(yi) for yi in y_mut]

    prob_mut = LinearStateSpaceProblem(
        A, nothing, zeros(2), (0, T);
        observables = y_mut,
        observables_noise = R,
    )
    sol_mut = solve(prob_mut, ConditionalLikelihood())

    prob_static = LinearStateSpaceProblem(
        SMatrix{2, 2}(A), nothing, SVector{2}(0.0, 0.0), (0, T);
        observables = y_static,
        observables_noise = Diagonal(SVector{2}(0.25, 0.25)),
    )
    sol_static = solve(prob_static, ConditionalLikelihood())

    @test sol_static.logpdf ≈ sol_mut.logpdf atol = 1.0e-12
end

@testset "ConditionalLikelihood — StaticArrays generic nonlinear" begin
    rho = 0.8
    alpha = 0.05
    sigma_e = 0.3
    T = 15

    Random.seed!(777)
    y_scalar = zeros(T)
    x = 0.0
    for t in 1:T
        x = rho * x + alpha * x^2 + sigma_e * randn()
        y_scalar[t] = x
    end

    y_mut = [[yi] for yi in y_scalar]
    y_static = [SVector{1}(yi) for yi in y_scalar]

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

    p = (; rho, alpha)
    prob_mut = StateSpaceProblem(
        nl_f!!, nothing, [0.0], (0, T), p;
        n_shocks = 0, n_obs = 0,
        observables = y_mut,
        observables_noise = Diagonal([sigma_e^2]),
    )
    sol_mut = solve(prob_mut, ConditionalLikelihood())

    prob_static = StateSpaceProblem(
        nl_f!!, nothing, SVector{1}(0.0), (0, T), p;
        n_shocks = 0, n_obs = 0,
        observables = y_static,
        observables_noise = Diagonal(SVector{1}(sigma_e^2)),
    )
    sol_static = solve(prob_static, ConditionalLikelihood())

    @test sol_static.logpdf ≈ sol_mut.logpdf atol = 1.0e-12
end
