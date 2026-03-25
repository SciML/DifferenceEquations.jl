# Linear Simulation

This tutorial walks through simulating a linear time-invariant state-space model of the form

```math
u_{n+1} = A \, u_n + B \, w_{n+1}
```

with observation equation

```math
z_n = C \, u_n + v_n
```

where ``w_{n+1} \sim N(0, I)`` and optionally ``v_n \sim N(0, D)``.

## Simulating a Linear State Space Model

We begin by defining system matrices and creating a [`LinearStateSpaceProblem`](@ref).
Passing `C` enables the observation equation, `observables_noise` adds Gaussian
measurement noise to the simulated observations, and `syms` attaches symbolic names
to the state variables.

```@example linear_sim
using DifferenceEquations, LinearAlgebra, Distributions, Random, Plots, DiffEqBase

A = [0.95 6.2; 0.0 0.2]
B = [0.0; 0.01;;]
C = [0.09 0.67; 1.00 0.00]
D = [0.1, 0.1]
u0 = zeros(2)
T = 10

prob = LinearStateSpaceProblem(A, B, u0, (0, T); C, observables_noise = D, syms = [:a, :b])
sol = solve(prob)
```

## Plotting

The solution object integrates with Plots.jl recipes. When `syms` are provided,
the legend labels correspond to those names.

```@example linear_sim
plot(sol)
```

## Accessing the Solution

The state trajectory is stored in `sol.u` as a `Vector{Vector}`. Standard indexing
works on the solution object directly.

```@example linear_sim
sol.u  # full state trajectory, Vector{Vector}
```

Access a specific time step:

```@example linear_sim
sol[2]  # state at the second time step (same as sol.u[2])
```

Or a specific element of the last state:

```@example linear_sim
sol[end][1]  # first element of the final state
```

Observations and noise are also available:

```@example linear_sim
sol.z  # observed trajectory, Vector{Vector}
```

```@example linear_sim
sol.W  # realized noise sequence, Vector{Vector}
```

## Symbolic Indexing

When `syms` are provided, you can extract the full time series for a state variable
by name:

```@example linear_sim
sol[:a]  # time series for state variable :a
```

If `obs_syms` are also provided, observation variables can be accessed similarly:

```@example linear_sim
prob_obs = LinearStateSpaceProblem(A, B, u0, (0, T); C, observables_noise = D,
    syms = [:a, :b], obs_syms = (:output, :consumption))
sol_obs = solve(prob_obs)
sol_obs[:output]  # time series for observation :output
```

## Fixed Noise

We can extract the noise from a previous simulation and use it to reproduce a
trajectory (possibly with different initial conditions). This is essential for
joint likelihood computations.

```@example linear_sim
noise = sol.W  # extract realized noise (Vector{Vector})
u0_new = [0.1, 0.0]
prob_fixed = LinearStateSpaceProblem(A, B, u0_new, (0, T); C, observables_noise = D,
    syms = [:a, :b], noise)
sol_fixed = solve(prob_fixed)
plot(sol_fixed)
```

## Impulse Response Functions

An impulse response function (IRF) applies a one-time unit shock at the first period
and traces the system's response. We construct this by passing a fixed noise sequence
where only the first entry is nonzero.

```@example linear_sim
function irf(A, B, C, T = 20)
    noise = [[i == 1 ? 1.0 : 0.0] for i in 1:T]
    problem = LinearStateSpaceProblem(A, B, zeros(2), (0, T); C, noise, syms = [:a, :b])
    return solve(problem)
end
plot(irf(A, B, C))
```

## Deterministic Dynamics (`B = nothing`)

When the model has no process noise, pass `B = nothing`. The solver will skip
noise generation entirely. No `sol.W` is produced.

```@example linear_sim
prob_det = LinearStateSpaceProblem(A, nothing, [1.0, 0.5], (0, T); C, syms = [:a, :b])
sol_det = solve(prob_det)
sol_det.W === nothing  # no noise generated
```

```@example linear_sim
plot(sol_det)
```

## No Observation Equation (`C = nothing`)

When you only need the state trajectory and don't require observations,
omit `C` (or pass `C = nothing`). No `sol.z` is produced.

```@example linear_sim
prob_no_obs = LinearStateSpaceProblem(A, B, u0, (0, T); syms = [:a, :b])
sol_no_obs = solve(prob_no_obs)
sol_no_obs.z === nothing  # no observations
```

```@example linear_sim
plot(sol_no_obs)
```

## Random Initial Conditions

Passing a `Distribution` for `u0` draws a random initial state at each solve.

```@example linear_sim
u0_dist = MvNormal([1.0 0.1; 0.1 1.0])  # zero-mean Gaussian
prob_rand = LinearStateSpaceProblem(A, nothing, u0_dist, (0, T); C)
sol_rand = solve(prob_rand)
plot(sol_rand)
```

## Ensemble Simulations

The SciML `EnsembleProblem` interface runs many independent simulations in parallel.
Each trajectory draws a fresh initial condition (when `u0` is a distribution) and/or
fresh noise. `EnsembleSummary` computes quantile bands across the ensemble.

```@example linear_sim
trajectories = 50
prob_ens = LinearStateSpaceProblem(A, B, u0_dist, (0, T); C)
ensemble_sol = solve(EnsembleProblem(prob_ens), DirectIteration(), EnsembleThreads();
    trajectories)
summ = EnsembleSummary(ensemble_sol)
plot(summ)
```
