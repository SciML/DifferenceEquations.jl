# Generic Callbacks

The [`StateSpaceProblem`](@ref) type provides a fully generic interface for
discrete-time state-space models. Instead of specifying matrices, you supply
callback functions for the state transition and observation equations. This is
useful for nonlinear models, time-varying dynamics, or any structure that does not
fit the linear or quadratic templates.

## Callback Signatures

The two callbacks follow the "bang-bang" convention used throughout SciML: for
mutable arrays, mutate the output buffer in place and return it; for immutable
arrays (e.g., `SVector`), ignore the buffer and return a new value.

**Transition function:** `f!!(x_next, x, w, p, t) -> x_next`
- `x_next`: pre-allocated output buffer (mutate in place for mutable arrays)
- `x`: current state
- `w`: noise shock at this step (or `nothing` if `n_shocks = 0`)
- `p`: parameters passed to the problem
- `t`: integer time index (0-based)

**Observation function:** `g!!(y, x, p, t) -> y`
- `y`: pre-allocated output buffer
- `x`: current state
- `p`: parameters
- `t`: integer time index (0-based)

Pass `nothing` for the observation function if no observations are needed.

## Example: Linear Model via Callbacks

We can reproduce the behavior of [`LinearStateSpaceProblem`](@ref) using generic
callbacks. This verifies the interface and demonstrates the pattern.

```@example generic
using DifferenceEquations, LinearAlgebra, DiffEqBase, Random

A = [0.95 6.2; 0.0 0.2]
B = [0.0; 0.01;;]
C = [0.09 0.67; 1.00 0.00]

linear_f!! = (x_next, x, w, p, t) -> begin
    mul!(x_next, p.A, x)
    mul!(x_next, p.B, w, 1.0, 1.0)
    return x_next
end
linear_g!! = (y, x, p, t) -> begin
    mul!(y, p.C, x)
    return y
end
p = (; A, B, C)
u0 = zeros(2)
T = 10

prob = StateSpaceProblem(linear_f!!, linear_g!!, u0, (0, T), p;
    n_shocks = 1, n_obs = 2, syms = (:a, :b))
sol = solve(prob)
```

The solution has the same structure as the linear case:

```@example generic
sol.u  # state trajectory, Vector{Vector}
```

```@example generic
sol.z  # observations, Vector{Vector}
```

We can verify this matches the matrix-based formulation:

```@example generic
Random.seed!(123)
sol_generic = solve(StateSpaceProblem(linear_f!!, linear_g!!, u0, (0, T), p;
    n_shocks = 1, n_obs = 2))

Random.seed!(123)
sol_linear = solve(LinearStateSpaceProblem(A, B, u0, (0, T); C))

sol_generic.u ≈ sol_linear.u
```

## Example: Nonlinear Growth Model

`StateSpaceProblem` handles arbitrary nonlinear dynamics. Here is a discrete-time logistic growth model with process noise, demonstrating that the generic callback interface works for any transition function:

```@example generic
# Nonlinear transition: logistic growth with stochastic shocks
logistic_f!! = (x_next, x, w, p, t) -> begin
    x_next[1] = p.r * x[1] * (1.0 - x[1] / p.K) + p.sigma * w[1]
    return x_next
end

# Observation: noisy measurement of population
logistic_g!! = (y, x, p, t) -> begin
    y[1] = x[1]
    return y
end

p_logistic = (; r = 1.5, K = 100.0, sigma = 2.0)
u0_logistic = [50.0]

prob_logistic = StateSpaceProblem(logistic_f!!, logistic_g!!, u0_logistic, (0, 50), p_logistic;
    n_shocks = 1, n_obs = 1, syms = (:population,), obs_syms = (:measured_pop,))
sol_logistic = solve(prob_logistic)
```

## Parametric Models and `remake`

The `p` argument holds all model parameters. When exploring different parameter
values, use `remake` to create a new problem without reallocating everything.

```@example generic
new_u0 = [0.1, 0.2]
new_p = (; A = A * 0.99, B, C)

prob2 = remake(prob; u0 = new_u0, p = new_p)
sol2 = solve(prob2)
sol2.u[1]  # new initial condition
```

The `remake` function preserves all keyword arguments (noise, observables, syms, etc.)
from the original problem.

## Symbolic Indexing

`StateSpaceProblem` supports the same symbolic indexing as the linear problem types.
Pass `syms` for state variable names and `obs_syms` for observation names.

```@example generic
D = [0.1, 0.1]
noise = sol.W  # reuse noise from earlier

prob_sym = StateSpaceProblem(linear_f!!, linear_g!!, u0, (0, T), p;
    n_shocks = 1, n_obs = 2,
    syms = (:capital, :productivity),
    obs_syms = (:output, :consumption),
    observables_noise = D, noise)
sol_sym = solve(prob_sym)

sol_sym[:capital]  # state time series by name
```

```@example generic
sol_sym[:output]  # observation time series by name
```
