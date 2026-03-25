# Getting Started

This tutorial walks through the core workflow of DifferenceEquations.jl: defining a linear state-space model, simulating it, and computing likelihoods.

## Creating a Linear State Space Model

A [`LinearStateSpaceProblem`](@ref) represents a linear time-invariant state-space model:

```math
u_{n+1} = A\, u_n + B\, w_{n+1}, \qquad z_n = C\, u_n + v_n
```

Define the model primitives, create a problem, and solve:

```@example getting_started
using DifferenceEquations, LinearAlgebra, Random, DiffEqBase
A = [0.95 6.2; 0.0 0.2]
B = [0.0; 0.01;;]
C = [0.09 0.67; 1.00 0.00]
u0 = zeros(2)
T = 10

prob = LinearStateSpaceProblem(A, B, u0, (0, T); C)
sol = solve(prob)
sol.u[end]
```

## Computing Likelihood

To compute log-likelihoods, provide `observables` (a `Vector{Vector}` of length `T`) and `observables_noise` (a diagonal covariance as a `Vector`, or a full covariance matrix).

!!! note "Timing convention"

    Observations correspond to ``z_1, z_2, \ldots, z_T`` -- that is, the states *after* the initial condition. Pass `T` observation vectors for a `tspan` of `(0, T)`.

First, simulate some data to use as observables:

```@example getting_started
Random.seed!(123)
D = [0.1, 0.1]  # diagonal observation noise
prob_sim = LinearStateSpaceProblem(A, B, u0, (0, T); C, observables_noise = D)
sol_sim = solve(prob_sim)

# Extract observations at times 1..T (skip the initial condition at t=0)
observables = sol_sim.z[2:end]
length(observables)  # should be T
```

Compute the **joint** log-likelihood given fixed noise using [`DirectIteration`](@ref):

```@example getting_started
prob_lik = LinearStateSpaceProblem(A, B, u0, (0, length(observables)); C,
    observables = observables,
    observables_noise = D,
    noise = sol_sim.W)
sol_lik = solve(prob_lik)
sol_lik.logpdf  # joint log-likelihood
```

For the **marginal** log-likelihood (integrating out the latent noise), use a [`KalmanFilter`](@ref) by additionally providing a Gaussian prior on `u0`:

```@example getting_started
prob_kf = LinearStateSpaceProblem(A, B, u0, (0, length(observables)); C,
    observables = observables,
    observables_noise = D,
    u0_prior_mean = zeros(2),
    u0_prior_var = Matrix(1.0I, 2, 2))
sol_kf = solve(prob_kf)  # KalmanFilter is auto-selected
sol_kf.logpdf  # marginal log-likelihood
```

## DataFrame Conversion

Convert the state trajectory to a `DataFrame` for analysis. Column names come from `syms` if provided:

```@example getting_started
using DataFrames
prob_df = LinearStateSpaceProblem(A, B, u0, (0, T); C,
    syms = [:capital, :productivity])
sol_df = solve(prob_df)
DataFrame(sol_df)
```

## Next Steps

  - [Linear Simulation](@ref) -- detailed simulation examples, symbolic indexing, fixed noise, and ensemble runs.
  - [Likelihood & Kalman Filter](@ref) -- marginal and joint likelihood, gradient-based estimation.
  - [Quadratic Models](@ref) -- second-order perturbation models.
  - [Generic Callbacks](@ref) -- user-defined nonlinear transition and observation functions.
  - [Workspace API](@ref) -- allocation-free repeated solves for performance-critical loops.
  - [Enzyme AD](@ref) -- differentiating through solvers and filters with Enzyme.jl.
