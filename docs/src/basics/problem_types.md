# Problem Types

DifferenceEquations.jl provides a hierarchy of problem types for defining discrete-time state-space models. All concrete problem types inherit from `AbstractStateSpaceProblem` and share a common interface for specifying dynamics, observations, and noise.

## Abstract Type

```@docs
AbstractStateSpaceProblem
```

## LinearStateSpaceProblem

```@docs
LinearStateSpaceProblem
```

## QuadraticStateSpaceProblem

```@docs
QuadraticStateSpaceProblem
```

## PrunedQuadraticStateSpaceProblem

```@docs
PrunedQuadraticStateSpaceProblem
```

## StateSpaceProblem

```@docs
StateSpaceProblem
```

## Common Keyword Arguments

All problem constructors accept the following keyword arguments:

| Keyword | Description | Default |
|---------|-------------|---------|
| `C` | Observation matrix (linear) or `C_0,C_1,C_2` (quadratic) | `nothing` |
| `observables_noise` | Observation noise covariance (vector = diagonal, matrix = full) | `nothing` |
| `observables` | Observed data as `Vector{Vector{T}}` | `nothing` |
| `noise` | Fixed noise as `Vector{Vector{T}}` | `nothing` (drawn randomly) |
| `syms` | State variable names for symbolic indexing | `nothing` |
| `obs_syms` | Observation variable names for symbolic indexing | `nothing` |
| `u0_prior_mean` | Prior mean for Kalman filtering (linear only) | `nothing` |
| `u0_prior_var` | Prior covariance for Kalman filtering (linear only) | `nothing` |

The `observables_noise` keyword has a dual role:
- **During simulation** (when `observables` is not provided): observation noise with this covariance is added to the simulated observations `sol.z`.
- **During likelihood computation** (when `observables` is provided): it defines the observation noise covariance used in the log-likelihood calculation.

## Remaking Problems

Use `remake` to create a modified copy of a problem, changing specific fields while keeping everything else. This is useful for parameter sweeps and optimization loops.

```@example remake_example
using DifferenceEquations, LinearAlgebra
A = [0.95 6.2; 0.0 0.2]
B = [0.0; 0.01;;]
prob = LinearStateSpaceProblem(A, B, zeros(2), (0, 5))
prob2 = remake(prob; u0 = [0.1, 0.2])
sol2 = solve(prob2)
sol2.u[1]  # new initial condition
```
