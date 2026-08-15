# Developer Interfaces

This page records the contracts used when adding a new state-space problem or
algorithm implementation. These are developer interfaces: ordinary users should use
the documented concrete problem types and algorithms instead of depending on the
internal dispatch hooks.

## Problem contract

Every `AbstractStateSpaceProblem` subtype must provide:

- `u0`, `tspan`, and `p` fields compatible with the SciML problem interface.
- An integer distance between `tspan[1]` and `tspan[2]`.
- `SciMLBase.remake` behavior for fields that users are expected to vary.
- A corresponding `default_alg` method or an explicitly supplied algorithm.

The generic solver allocates a `StateSpaceWorkspace` and calls the model hooks below.
The hooks must preserve the buffer and callback conventions used by the concrete
problem types.

## Algorithm contract

An `AbstractDifferenceEquationAlgorithm` subtype must provide the allocation and
solve behavior required by `init`, `solve!`, and `solve`. The built-in algorithms
are:

- `DirectIteration`: calls a model's transition and observation callbacks at every
  integer time step.
- `KalmanFilter`: adds Gaussian filtering and posterior-covariance propagation for
  linear problems with a Gaussian prior and observation noise.
- `ConditionalLikelihood`: evaluates prediction errors for observed trajectories.

## Generic callback contract

`StateSpaceProblem` is the supported user-facing generic model interface. Its
callbacks are called as follows:

```julia
transition(x_next, x, w, p, t) -> x_next
observation(y, x, p, t) -> y
```

The callback may mutate and return `x_next`/`y` for mutable arrays, or return a new
immutable value such as an `SVector`. The time argument is zero-based: the first
transition receives `t = 0`, and the first observation receives `t = 0`. Set
`observation = nothing` and `n_obs = 0` when no observation equation exists.

The generic callback contract is tested with `StateSpaceProblem` in the Core suite;
the test consumer uses only `AbstractStateSpaceProblem`, `init`, `solve!`, and
`StateSpaceSolution`.
