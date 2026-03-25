# Solvers

Solving a state-space problem is as simple as calling `solve(prob)`, which automatically selects an appropriate algorithm. You can also pass an algorithm explicitly via `solve(prob, alg)`.

```@docs
DirectIteration
```

```@docs
KalmanFilter
```

## Default Algorithm Selection

When no algorithm is specified, `solve(prob)` selects the algorithm based on the problem type and its fields:

- **`DirectIteration`** is the default for all problem types. It simulates the state-space model forward in time, generating states and observations directly. If `observables` are provided, it computes the joint log-likelihood of the observed data given the noise sequence.

- **`KalmanFilter`** is auto-selected for `LinearStateSpaceProblem` when all of the following conditions hold:
  - `u0_prior_var` is an `AbstractMatrix` (prior covariance is specified)
  - `noise` is `nothing` (noise is not fixed)
  - `observables` is an `AbstractVector` (observed data is provided)
  - `observables_noise` is provided (observation noise covariance is specified)

  The Kalman filter computes the filtered state estimates and the marginal log-likelihood of the observations, integrating over the unknown noise sequence.
