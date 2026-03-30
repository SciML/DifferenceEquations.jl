# Solvers

Solving a state-space problem is as simple as calling `solve(prob)`, which automatically selects an appropriate algorithm. You can also pass an algorithm explicitly via `solve(prob, alg)`.

```@docs
DirectIteration
```

```@docs
KalmanFilter
```

```@docs
ConditionalLikelihood
```

## Default Algorithm Selection

When no algorithm is specified, `solve(prob)` selects the algorithm based on the problem type and its fields:

- **`DirectIteration`** is the default for all problem types. It simulates the state-space model forward in time, generating states and observations directly. If `observables` are provided, it computes the joint log-likelihood of the observed data given the noise sequence.

- **`KalmanFilter`** is auto-selected for `LinearStateSpaceProblem` when all of the following conditions hold:
  - `u0_prior_var` is an `AbstractMatrix` (prior covariance is specified)
  - `noise` is `nothing` (noise is not fixed)
  - `observables` is an `AbstractVector` (observed data is provided)
  - `observables_noise` is an `AbstractMatrix` (observation noise covariance is specified)
  - `A`, `B`, and `C` are all `AbstractMatrix` (not `nothing`)

  The Kalman filter computes the filtered state estimates and the marginal log-likelihood of the observations, integrating over the unknown noise sequence.

- **`ConditionalLikelihood`** is never auto-selected. You must pass it explicitly via `solve(prob, ConditionalLikelihood())`. Use it for fully-observed state-space models (AR, VAR, nonlinear) where the state is directly observed and you want the prediction error decomposition log-likelihood. Works with all problem types.

!!! warning

    If any of the KalmanFilter conditions are not met, `DirectIteration` is silently selected instead. For example, forgetting to pass `C` or `u0_prior_var` will produce a `DirectIteration` solve with `logpdf = 0.0` rather than the expected Kalman filter result.

## `save_everystep` Keyword

All algorithms support `save_everystep=false`, which stores only the initial and final states instead of the full trajectory:

```julia
sol = solve(prob; save_everystep=false)              # 2-element sol.u
sol = solve(prob, ConditionalLikelihood(); save_everystep=false)
sol = solve(prob, KalmanFilter(); save_everystep=false)
```

When `save_everystep=false`:
- `sol.u` contains `[u_initial, u_final]` (2 entries instead of T+1)
- `sol.z` contains `[z_initial, z_final]` (if observations are present)
- `sol.P` contains `[P_initial, P_final]` (KalmanFilter only)
- `sol.logpdf` is **identical** — computed on the fly, not from stored trajectory

This is useful when you only need the final state or the log-likelihood (e.g., in optimization loops). It dramatically reduces memory allocation, which benefits ForwardDiff gradient computation:

| Scenario | Typical speedup | Allocation reduction |
|----------|----------------|---------------------|
| ForwardDiff + StaticArrays (KF, N=5) | **7x** | 4,288 → 175 |
| ForwardDiff + StaticArrays (CL, N=5) | **3.4x** | 805 → 190 |
| ForwardDiff + mutable (KF, N=30) | **1.5x** | 342k → 8k |

The workspace API also supports it:

```julia
ws = init(prob, alg; save_everystep=false)
sol = solve!(ws)   # reads save_everystep from workspace
```
