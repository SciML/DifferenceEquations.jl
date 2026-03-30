# Conditional Likelihood

The [`ConditionalLikelihood`](@ref) algorithm computes the prediction error
decomposition log-likelihood for fully-observed state-space models. At each
time step, it predicts the next observation from the *observed* current state
(not the model-predicted state), and accumulates the Gaussian log-likelihood
of the innovation (prediction error).

This is the standard approach for maximum likelihood estimation of AR(1),
VAR(1), nonlinear DSGE, and other models where the state is directly observed.

## When to Use Each Algorithm

| Algorithm | Use Case |
|-----------|----------|
| [`DirectIteration`](@ref) | Simulation, or joint likelihood given a fixed noise sequence |
| [`KalmanFilter`](@ref) | Marginal likelihood for linear models with latent (unobserved) noise |
| [`ConditionalLikelihood`](@ref) | MLE for fully-observed models (AR, VAR, nonlinear) |

## Mathematical Formulation

Given a state-space model with transition ``x_{t+1} = f(x_t, w_t)`` and
observation ``z_t = g(x_t)``, the conditional log-likelihood is:

```math
\log L = \sum_{t=1}^{T} \left[ -\frac{1}{2} \left( M \log(2\pi) + \log|R| + \nu_t^\top R^{-1} \nu_t \right) \right]
```

where ``\nu_t = y_t - g(f(y_{t-1}, w_t))`` is the innovation (prediction error),
``R`` is the observation noise covariance, and ``M`` is the observation dimension.

The key difference from `DirectIteration` is that at each step the state is
**clamped to the observation**: the prediction uses ``f(y_{t-1}, \ldots)``
rather than ``f(f(\ldots, u_0), \ldots)``.

## AR(1) Example

```@example cond_lik
using DifferenceEquations, LinearAlgebra, Random

rho_true = 0.8
sigma_e = 0.5
T = 200

# Generate AR(1) data: y_t = rho * y_{t-1} + e_t
Random.seed!(42)
y_scalar = zeros(T)
x = 0.0
for t in 1:T
    x = rho_true * x + sigma_e * randn()
    y_scalar[t] = x
end
y = [[yi] for yi in y_scalar]  # Vector{Vector{Float64}}

# Compute conditional log-likelihood
prob = LinearStateSpaceProblem(
    fill(rho_true, 1, 1), nothing, [0.0], (0, T);
    observables = y,
    observables_noise = Diagonal([sigma_e^2]),
)
sol = solve(prob, ConditionalLikelihood())
sol.logpdf
```

## VAR(1) Example

The same approach works for multivariate models:

```@example cond_lik
A = [0.8 0.1; -0.1 0.7]
R = Diagonal([0.25, 0.25])
T_var = 100

Random.seed!(123)
y_var = Vector{Vector{Float64}}(undef, T_var)
x_var = zeros(2)
for t in 1:T_var
    x_var = A * x_var + cholesky(R).L * randn(2)
    y_var[t] = copy(x_var)
end

prob_var = LinearStateSpaceProblem(
    A, nothing, zeros(2), (0, T_var);
    observables = y_var,
    observables_noise = R,
)
sol_var = solve(prob_var, ConditionalLikelihood())
sol_var.logpdf
```

## Nonlinear Example with StateSpaceProblem

`ConditionalLikelihood` works with all problem types, including user-defined
nonlinear callbacks via [`StateSpaceProblem`](@ref).

Here we estimate a nonlinear AR(1): ``x_{t+1} = \rho x_t + \alpha x_t^2 + e_t``.

```@example cond_lik
rho_nl = 0.8
alpha_nl = 0.05
sigma_nl = 0.3
T_nl = 100

Random.seed!(99)
y_nl_scalar = zeros(T_nl)
x_nl = 0.0
for t in 1:T_nl
    x_nl = rho_nl * x_nl + alpha_nl * x_nl^2 + sigma_nl * randn()
    y_nl_scalar[t] = x_nl
end
y_nl = [[yi] for yi in y_nl_scalar]

# Define nonlinear transition (supports both mutable and immutable arrays)
function nl_transition!!(x_next, x, w, p, t)
    (; rho, alpha) = p
    val = rho * x[1] + alpha * x[1]^2
    if ismutable(x_next)
        x_next[1] = val
        return x_next
    else
        return typeof(x)(val)
    end
end

p_nl = (; rho = rho_nl, alpha = alpha_nl)
prob_nl = StateSpaceProblem(
    nl_transition!!, nothing, [0.0], (0, T_nl), p_nl;
    n_shocks = 0, n_obs = 0,
    observables = y_nl,
    observables_noise = Diagonal([sigma_nl^2]),
)
sol_nl = solve(prob_nl, ConditionalLikelihood())
sol_nl.logpdf
```

## Maximum Likelihood Estimation

`ConditionalLikelihood` is fully differentiable with ForwardDiff.jl, making it
straightforward to use with gradient-based optimization for MLE.

Use `save_everystep=false` when you only need the log-likelihood (not the
full trajectory). This reduces allocations and speeds up ForwardDiff
gradient computation — up to 7x faster with StaticArrays.

```@example cond_lik
using ForwardDiff

# Negative log-likelihood as a function of rho
function neg_loglik(rho_vec)
    T_el = eltype(rho_vec)
    A_opt = fill(rho_vec[1], 1, 1)
    prob_opt = LinearStateSpaceProblem(
        A_opt, nothing, [zero(T_el)], (0, length(y));
        observables = y,
        observables_noise = Diagonal([T_el(sigma_e^2)]),
    )
    return -solve(prob_opt, ConditionalLikelihood(); save_everystep=false).logpdf
end

# Gradient at the true value
grad = ForwardDiff.gradient(neg_loglik, [rho_true])
```

The gradient is near zero at the true parameter value, confirming the MLE
is correctly identified. For full optimization, use Optimization.jl with
`AutoForwardDiff()` and an optimizer like `LBFGS()`.

## Workspace API

For repeated solves (e.g., inside an optimizer), use the `init`/`solve!`
pattern to avoid repeated memory allocation:

```@example cond_lik
ws = init(prob, ConditionalLikelihood())
sol_ws = solve!(ws)
sol_ws.logpdf
```

With `save_everystep=false`, the workspace allocates only 2-element
buffers:

```@example cond_lik
ws_ep = init(prob, ConditionalLikelihood(); save_everystep=false)
sol_ep = solve!(ws_ep)
length(sol_ep.u)  # 2: [u_initial, u_final]
```
