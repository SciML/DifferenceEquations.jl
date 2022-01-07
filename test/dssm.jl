# Tests as a downstream of DSSM, taking the solutions there as inputs
using DifferenceEquations
using DifferentiableStateSpaceModels
using Distributions
using LaTeXStrings
using LinearAlgebra
using SymbolicUtils

# Grab the model
# Keep the model generated files static in the test folder just to avoid DSSM upstream disturbances
include(joinpath(pkgdir(DifferenceEquations), "test/generated_models/rbc_observables.jl"))
m = PerturbationModel(Main.rbc_observables)
p_f = (ρ = 0.2, δ = 0.02, σ = 0.01, Ω_1 = 0.1)
p_d = (α = 0.5, β = 0.95)

# Generate cache, create perutrbation solution
c = SolverCache(m, Val(1), p_d)
sol = generate_perturbation(m, p_d, p_f; cache = c)

# Timespan to simulate across
T = 500

# Set initial state
u0 = zeros(m.n_x)

# Construct problem with no observables
problem = StateSpaceProblem(
    DifferentiableStateSpaceModels.dssm_evolution,
    DifferentiableStateSpaceModels.dssm_volatility,
    DifferentiableStateSpaceModels.dssm_observation,
    u0,
    (0, T),
    sol,
    noise = [randn(sol.n_ϵ) for _ in 1:T]
)

# Solve the model, this generates
# simulated data.
simul = DifferenceEquations.solve(problem, NoiseConditionalFilter())
# Grab simulated data for the next tests, leave the first one -- the initial condition -- out
z = simul.z[2:end]

@testset "General Nonlinear Simulations" begin
    # Now solve using the previous data as observables.
    # Solving this problem also includes a likelihood.
    problem_data = StateSpaceProblem(
        DifferentiableStateSpaceModels.dssm_evolution,
        DifferentiableStateSpaceModels.dssm_volatility,
        DifferentiableStateSpaceModels.dssm_observation,
        u0,
        (0, T),
        sol,
        obs_noise = sol.D,
        observables = z,
        noise = [randn(sol.n_ϵ) for _ in 1:T]
    )
    # Generate likelihood.
    simul_with_likelihood = @inferred solve(problem_data, NoiseConditionalFilter())
end

@testset "Kalman" begin
    ## Kalman filter test
    linear_problem = LinearStateSpaceProblem(
        sol.A,
        sol.B,
        sol.C,
        MvNormal(zeros(eltype(u0), length(u0)), I), # Prior of initial point
        (0, T),
        noise = nothing,
        obs_noise = sol.D,
        observables = z
    )

    # Solve with Kalman filter
    simul_kalman_filter = @inferred solve(linear_problem, KalmanFilter())
end

@testset "Linear" begin
    linear_problem = LinearStateSpaceProblem(
        sol.A,
        sol.B,
        sol.C,
        u0,
        (0, T),
        noise = [randn(sol.n_ϵ) for _ in 1:T],
        obs_noise = sol.D,
        observables = z
    )

    # Simulate linear AR(1) process
    simul_linear = @inferred solve(linear_problem, NoiseConditionalFilter())
end

# Second-order preparations
c = SolverCache(m, Val(2), p_d)
sol = generate_perturbation(m, p_d, p_f, Val(2); cache = c)

@testset "Quadratic" begin
    quadratic_problem = QuadraticStateSpaceProblem(
        sol.A_0,
        sol.A_1,
        sol.A_2,
        sol.B,
        sol.C_0,
        sol.C_1,
        sol.C_2,
        u0,
        (0, T),
        noise = [randn(sol.n_ϵ) for _ in 1:T],
        obs_noise = sol.D,
        observables = z
    )

    # Simulate quadratic pruned process
    simul_quadratic = @inferred solve(quadratic_problem, NoiseConditionalFilter())
end