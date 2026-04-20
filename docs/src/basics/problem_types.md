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

The following keywords are shared by all problem constructors:

| Keyword | Description | Default |
|---------|-------------|---------|
| `observables_noise` | Observation noise covariance matrix (`AbstractMatrix`, e.g. `Diagonal(d)` or `Symmetric(H * H')`) | `nothing` |
| `observables` | Observed data as `Vector{Vector{T}}` | `nothing` |
| `noise` | Fixed noise as `Vector{Vector{T}}` | `nothing` (drawn randomly) |
| `syms` | State variable names as a `Tuple` of `Symbol`s, e.g. `(:x, :y)` | `nothing` |
| `obs_syms` | Observation variable names as a `Tuple` of `Symbol`s | `nothing` |

### Linear-only keywords

These are accepted only by [`LinearStateSpaceProblem`](@ref):

| Keyword | Description | Default |
|---------|-------------|---------|
| `C` | Observation matrix | `nothing` |
| `u0_prior_mean` | Prior mean for Kalman filtering | `nothing` |
| `u0_prior_var` | Prior covariance for Kalman filtering | `nothing` |

### Quadratic-only keywords

[`QuadraticStateSpaceProblem`](@ref) and [`PrunedQuadraticStateSpaceProblem`](@ref) accept `C_0`, `C_1`, `C_2` instead of `C`.

### Generic-only keywords

[`StateSpaceProblem`](@ref) requires the additional positional/keyword arguments `n_shocks` and `n_obs` to specify dimensions.

### Dual role of `observables_noise`

The `observables_noise` keyword has a dual role:
- **During simulation** (when `observables` is not provided): observation noise with this covariance is added to the simulated observations `sol.z`.
- **During likelihood computation** (when `observables` is provided): it defines the observation noise covariance used in the log-likelihood calculation.

!!! note

    `observables_noise` must be an `AbstractMatrix`. For diagonal noise, use `Diagonal([σ₁², σ₂², …])` where the entries are **variances** (not standard deviations). For a general covariance, use a full `Matrix` or `Symmetric(H * H')`.

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
