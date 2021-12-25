using Test
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
    noise = StandardGaussian(1)
)

# Solve the model, this generates
# simulated data.
simul = @inferred DifferenceEquations.solve(problem, NoiseConditionalFilter())

# Extract the observables, latent noise, and latent states.
z, n, u = simul.z, simul.n, simul.u

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
    noise = StandardGaussian(1)
)

# Generate likelihood.
s2 = @inferred DifferenceEquations.solve(problem_data, NoiseConditionalFilter())

## Kalman filter test
linear_problem = LinearStateSpaceProblem(
    sol.A,
    sol.B,
    sol.C,
    MvNormal(u0, diagm(ones(length(u0)))),
    (1,T),
    noise=sol.Q,
    obs_noise=sol.D,
    observables = z
)

# Solve with Kalman filter
simul_k = @inferred DifferenceEquations.solve(linear_problem, KalmanFilter())
