# Likelihood & Kalman Filter

DifferenceEquations.jl supports two approaches to computing the log-likelihood of
observed data:

- **Marginal likelihood** via the [`KalmanFilter`](@ref): the probability of the
  observed data conditioned on the core model parameters (``A, B, C``, etc.) and the
  initial condition prior, with the latent noise sequence analytically integrated out.
  This is the standard approach for maximum likelihood estimation (MLE) of structural
  parameters.
- **Joint likelihood** via [`DirectIteration`](@ref): the probability of the observed
  data AND a specific noise realization, conditioned on the core parameters and initial
  conditions. Requires fixing the noise sequence. Useful in Bayesian methods where the
  noise is sampled as part of inference (e.g., particle MCMC, HMC on latent variables).

Both approaches are fully differentiable with Enzyme.jl and ForwardDiff.jl.

## Simulating Observations

First, let us simulate a model with observation noise to produce synthetic data.

```@example linear_lik
using DifferenceEquations, LinearAlgebra, Distributions, Random

A = [0.95 6.2; 0.0 0.2]
B = [0.0; 0.01;;]
C = [0.09 0.67; 1.00 0.00]
D = Diagonal([0.1, 0.1])
u0 = zeros(2)
T = 80

Random.seed!(42)
prob_sim = LinearStateSpaceProblem(A, B, MvNormal(zeros(2), I(2)), (0, T);
    C, observables_noise = D)
sol_sim = solve(prob_sim)
sol_sim.z  # simulated observations with noise (Vector{Vector})
```

## Marginal Likelihood with the Kalman Filter

The Kalman filter computes the marginal log-likelihood by integrating out the latent
noise sequence. It requires a Gaussian prior on the initial state (`u0_prior_mean`,
`u0_prior_var`) and Gaussian observation noise (`observables_noise`).

!!! note "Timing convention"

    Observations correspond to ``z_1, z_2, \ldots, z_T`` (predictions starting from
    the second state). When the simulation produces `T+1` observation vectors
    (including ``z_0``), pass `sol.z[2:end]` as the observables. The length of
    `observables` must equal the integer distance of `tspan`.

```@example linear_lik
observables = sol_sim.z[2:end]  # Vector{Vector}, length T

u0_prior_mean = zeros(2)
u0_prior_var = Matrix(1.0 * I(2))

prob_kalman = LinearStateSpaceProblem(A, B, u0, (0, length(observables)); C,
    observables_noise = D, observables,
    u0_prior_mean, u0_prior_var)

# KalmanFilter is auto-selected when priors + observables + noise covariance are given
sol_kalman = solve(prob_kalman)
sol_kalman.logpdf  # marginal log-likelihood
```

The Kalman solution also provides filtered state estimates in `sol.u` and posterior
covariances in `sol.P`:

```@example linear_lik
sol_kalman.u[end]  # filtered mean at the final time step
```

```@example linear_lik
sol_kalman.P[end]  # posterior covariance at the final time step
```

## Joint Likelihood with Fixed Noise

When both `noise` and `observables` are provided, `DirectIteration` iterates the
state transition forward using the given noise and accumulates the joint
log-likelihood of the observations.

```@example linear_lik
noise = sol_sim.W  # realized noise from simulation (Vector{Vector})

prob_joint = LinearStateSpaceProblem(A, B, u0, (0, length(observables)); C,
    observables_noise = D, observables, noise)
sol_joint = solve(prob_joint)
sol_joint.logpdf  # joint log-likelihood conditioned on noise
```

## Composing Structural Models

In practice, the state-space matrices ``A, B, C, D`` are often generated from deeper
structural (or "deep") parameters. The entire pipeline from structural parameters
to log-likelihood is differentiable.

```@example linear_lik
function generate_model(beta)
    A = [beta 6.2; 0.0 0.2]
    B = [0.0; 0.001;;]
    C = [0.09 0.67; 1.00 0.00]
    D = Diagonal([0.01, 0.01])
    return (; A, B, C, D)
end

function kalman_model_likelihood(beta, observables)
    mod = generate_model(beta)
    prob = LinearStateSpaceProblem(mod.A, mod.B, zeros(2), (0, length(observables));
        C = mod.C, observables, observables_noise = mod.D,
        u0_prior_mean = zeros(2), u0_prior_var = Matrix(1.0 * I(2)))
    return solve(prob).logpdf
end

kalman_model_likelihood(0.95, observables)
```

## Next Steps

To differentiate these likelihoods with Enzyme.jl and use them inside an optimization loop, see the [Enzyme AD](@ref) page. It covers the workspace-based `init`/`solve!` pattern required by Enzyme, gradient computation for both `DirectIteration` and `KalmanFilter`, and a full maximum-likelihood example with Optimization.jl.
