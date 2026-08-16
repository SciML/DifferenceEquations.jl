# Developer Interfaces

This page documents the extension contracts for DifferenceEquations.jl. These are
developer interfaces for adding problem or algorithm implementations; they are not
ordinary user APIs. User code should construct one of the documented problem types
and call [`solve`](@ref), [`init`](@ref), or [`solve!`](@ref).

## Abstract interfaces

```@docs
DifferenceEquations.AbstractDifferenceEquationAlgorithm
DifferenceEquations.default_alg
```

## Problem contract

A subtype of [`AbstractStateSpaceProblem`](@ref) must expose the fields consumed by
the generic SciML and solver interfaces:

- `u0`: Initial state, or an initial-state distribution supported by the problem.
- `tspan`: A two-element time span whose endpoint difference is an integer. The
  solver stores one state for each integer step.
- `p`: Parameters passed through `SciMLBase.remake` and into model callbacks.
- `noise`: A fixed process-noise sequence, or `nothing` when noise is generated.
- `observables`: Observed data, or `nothing` when no likelihood is requested.
- `observables_noise`: Observation covariance, or `nothing` when observation noise is
  disabled.
- `f`: An `SciMLBase.ODEFunction` bridge with a `SymbolCache` when symbolic indexing
  is supported.
- `obs_syms`: Observation names, or `nothing` when observations are unnamed.

The type must also support `SciMLBase.remake` for every field users are expected to
vary, and it must have a `default_alg(prob)` method or require callers to provide an
algorithm explicitly. The constructor must reject a non-integer time-span length.

## Problem dispatch contract

The generic solver calls the following package-internal methods for each problem and
algorithm combination. A new problem implementation must provide these methods for
every algorithm it supports:

| Method | Contract |
| --- | --- |
| `_noise_matrix(prob)` | Return the process-noise matrix, or `nothing` for a deterministic problem. Its second dimension is the number of shocks. |
| `_init_model_state!!(prob, cache[, ::Val{false}])` | Initialize model-specific cache state before the first transition. Return `nothing`. |
| `_transition!!(x_next, x, w, prob, cache, t[, ::Val{false}])` | Compute the next state and return `x_next`. `t` is the one-based internal trajectory index. |
| `_observation!!(y, x, prob, cache, t[, ::Val{false}])` | Compute the observation and return `y`. |
| `alloc_sol(prob, alg, T[, ::Val{false}])` | Return a named tuple containing preallocated `u`, plus `z` and `P` when the selected problem has observations or posterior covariances. |
| `alloc_cache(prob, alg, T[, ::Val{false}])` | Return all scratch buffers required by the corresponding solver loop. |

The `Val(false)` methods are the endpoint-only implementation used by
`save_everystep = false`; they must use two solution slots and one scratch slot where
the algorithm stores time-dependent state. The regular methods allocate `T` slots,
where `T = tspan[2] - tspan[1] + 1`.

Transition and observation hooks follow the package's bang-bang convention: mutate
the first argument and return it for mutable arrays, or return a new value for
immutable arrays such as `SVector`. Every allocation made by the hooks must have the
same element type and shape as the corresponding problem state or observation.

## Algorithm contract

A subtype of [`DifferenceEquations.AbstractDifferenceEquationAlgorithm`](@ref) must
have compatible `alloc_sol`, `alloc_cache`, and `_solve!` methods. The built-in
algorithms use the following semantics:

- [`DirectIteration`](@ref) advances the state from `t = 0` through the integer
  time span and optionally computes observations and a joint likelihood.
- [`KalmanFilter`](@ref) is for linear Gaussian problems with a prior, observations,
  and an observation covariance; it returns filtered means, posterior covariances,
  and the marginal likelihood.
- [`ConditionalLikelihood`](@ref) evaluates the prediction-error likelihood for
  observed trajectories and requires both `observables` and `observables_noise`.

An algorithm may use the complete-trajectory or endpoint-only allocation contract,
but it must return a [`StateSpaceSolution`](@ref) through the SciML solution builder.
It must not require callers to access the scratch cache or private dispatch hooks.

## Generic callback contract

[`StateSpaceProblem`](@ref) is the user-facing generic model implementation and is
the reference implementation for the callback contract:

```julia
transition(x_next, x, w, p, t) -> x_next
observation(y, x, p, t) -> y
```

`transition` receives the reusable destination, current state, process shock,
parameters, and a zero-based time index. `observation` receives the reusable
destination, current state, parameters, and the same zero-based time convention.
The first transition and observation use `t = 0`; a transition at internal index `k`
receives `t = k - 2`, while an observation at index `k` receives `t = k - 1`.

For mutable arrays, callbacks must write the destination and return it. For immutable
states, callbacks must return a new value. Set `observation = nothing` and
`n_obs = 0` when the model has no observation equation. If `n_shocks > 0`, the fixed
noise sequence must contain exactly one shock for each transition.

The Core suite tests this contract through a consumer function typed only as
`AbstractStateSpaceProblem` and the generic `init`, `solve!`, and
`StateSpaceSolution` interfaces. It checks both complete trajectories and
`save_everystep = false` endpoint storage, including callback time and noise values.
