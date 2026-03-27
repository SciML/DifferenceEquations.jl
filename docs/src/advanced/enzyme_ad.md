# Enzyme AD

DifferenceEquations.jl is fully differentiable with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) in both reverse and forward mode. All examples below use the workspace-based `init`/`solve!` pattern with [`StateSpaceWorkspace`](@ref), which gives Enzyme the pre-allocated buffers it needs.

## The Core Pattern

Every Enzyme example in this package follows the same recipe:

1. **Flat-argument wrapper function.** Construct the `LinearStateSpaceProblem` *inside* the function from plain matrix/vector arguments. This keeps the Enzyme call site simple and avoids closing over mutable state.

2. **Pre-allocate with `init`.** Call `init(prob, alg)` once to obtain a workspace whose `.output` (solution) and `.cache` fields are correctly sized buffers. Then pass those buffers into the wrapper via `StateSpaceWorkspace(prob, alg, sol, cache)` followed by `solve!(ws).logpdf`.

3. **All arguments `Duplicated`.** Because every argument flows into the *same* `LinearStateSpaceProblem` struct, Enzyme treats the whole struct as active. If even one field is `Const` while others are `Duplicated`, Enzyme may silently produce wrong gradients. The safe rule: **mark every argument `Duplicated`**.

4. **Zero-initialized shadows for `sol`/`cache`.** Shadow copies for the solution and cache buffers must be created with `Enzyme.make_zero(deepcopy(...))`. A plain `deepcopy` copies the primal values into the shadow, which can produce `NaN` gradients. `make_zero` recursively zeroes every numeric field while preserving the nested structure.

## Differentiating Joint Likelihood

The joint likelihood conditions on a fixed noise sequence and accumulates the observation log-likelihood along the trajectory via [`DirectIteration`](@ref).

```@example enzyme
using DifferenceEquations, LinearAlgebra, Enzyme, Random

N, K, M = 2, 1, 2
A = [0.8 0.1; -0.1 0.7]
B = [0.1; 0.0;;]
C = [1.0 0.0; 0.0 1.0]
D = Diagonal([0.01, 0.01])  # diagonal covariance; use Symmetric(H * H') for non-diagonal
u0 = zeros(N)

Random.seed!(42)
noise = [randn(K) for _ in 1:5]
sim = solve(LinearStateSpaceProblem(A, B, u0, (0, 5); C, noise))
obs = [sim.z[t + 1] + 0.1 * randn(M) for t in 1:5]

# Likelihood function: all matrix args as separate parameters
function di_loglik(A, B, C, u0, noise, obs, R, sol, cache)::Float64
    prob = LinearStateSpaceProblem(A, B, u0, (0, length(obs));
        C, observables_noise = R, observables = obs, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol, cache)
    return solve!(ws).logpdf
end

# Pre-allocate buffers
prob0 = LinearStateSpaceProblem(A, B, u0, (0, length(obs));
    C, observables_noise = D, observables = obs, noise)
ws0 = init(prob0, DirectIteration())

# Compute gradient wrt A
dA = zero(A)
autodiff(Reverse, di_loglik,
    Duplicated(copy(A), dA),
    Duplicated(copy(B), zero(B)),
    Duplicated(copy(C), zero(C)),
    Duplicated(copy(u0), zero(u0)),
    Duplicated(deepcopy(noise), [zeros(K) for _ in noise]),
    Duplicated(deepcopy(obs), [zeros(M) for _ in obs]),
    Duplicated(copy(D), zero(D)),
    Duplicated(deepcopy(ws0.output), Enzyme.make_zero(deepcopy(ws0.output))),
    Duplicated(deepcopy(ws0.cache), Enzyme.make_zero(deepcopy(ws0.cache))))
dA  # gradient of logpdf with respect to A
```

## Differentiating the Kalman Filter

The [`KalmanFilter`](@ref) computes the marginal log-likelihood by integrating out the latent noise analytically. The same all-`Duplicated` pattern applies.

```@example enzyme
# Kalman filter likelihood
function kf_loglik(A, B, C, mu0, Sigma0, R, obs, sol, cache)::Float64
    prob = LinearStateSpaceProblem(A, B, zeros(eltype(A), size(A,1)), (0, length(obs));
        C, u0_prior_mean = mu0, u0_prior_var = Sigma0,
        observables_noise = R, observables = obs)
    ws = StateSpaceWorkspace(prob, KalmanFilter(), sol, cache)
    return solve!(ws).logpdf
end

mu0 = zeros(N)
Sigma0 = Matrix(1.0 * I(N))
prob_kf = LinearStateSpaceProblem(A, B, zeros(N), (0, length(obs));
    C, u0_prior_mean = mu0, u0_prior_var = Sigma0,
    observables_noise = D, observables = obs)
ws_kf = init(prob_kf, KalmanFilter())

dA_kf = zero(A)
autodiff(Reverse, kf_loglik,
    Duplicated(copy(A), dA_kf),
    Duplicated(copy(B), zero(B)),
    Duplicated(copy(C), zero(C)),
    Duplicated(copy(mu0), zero(mu0)),
    Duplicated(copy(Sigma0), zero(Sigma0)),
    Duplicated(copy(D), zero(D)),
    Duplicated(deepcopy(obs), [zeros(M) for _ in obs]),
    Duplicated(deepcopy(ws_kf.output), Enzyme.make_zero(deepcopy(ws_kf.output))),
    Duplicated(deepcopy(ws_kf.cache), Enzyme.make_zero(deepcopy(ws_kf.cache))))
dA_kf  # gradient of Kalman logpdf with respect to A
```

## Integration with Optimization.jl

The differentiable Kalman likelihood composes naturally with [Optimization.jl](https://github.com/SciML/Optimization.jl) for maximum-likelihood estimation. Because the all-`Duplicated` requirement cannot be expressed through `AutoEnzyme()`, we supply an explicit `grad` function that calls `Enzyme.autodiff` directly.

```@example enzyme
using Optimization, OptimizationOptimJL

# Simulate data from a known model
Random.seed!(42)
T_opt = 200
B_opt = [0.0; 0.001;;]
C_opt = [0.09 0.67; 1.00 0.00]
D_opt = Diagonal([0.01, 0.01])
prob_data = LinearStateSpaceProblem([0.95 6.2; 0.0 0.2], B_opt, zeros(2), (0, T_opt);
    C = C_opt, observables_noise = D_opt)
sol_data = solve(prob_data)
obs_data = sol_data.z[2:end]

# Pre-allocate Kalman workspace
mu0_opt = zeros(2)
Sigma0_opt = Matrix(1e-2 * I(2))
prob_base = LinearStateSpaceProblem([0.95 6.2; 0.0 0.2], B_opt, zeros(2),
    (0, length(obs_data)); C = C_opt, observables = obs_data,
    observables_noise = D_opt, u0_prior_mean = mu0_opt, u0_prior_var = Sigma0_opt)
ws_opt = init(prob_base, KalmanFilter())

# Objective and gradient using the flat-argument pattern
function neg_loglik(beta, p)
    A = [beta[1] 6.2; 0.0 0.2]
    return -kf_loglik(A, p.B, p.C, p.mu0, p.Sigma0, p.D, p.obs,
        deepcopy(p.sol), deepcopy(p.cache))
end

function neg_loglik_grad!(g, beta, p)
    A = [beta[1] 6.2; 0.0 0.2]
    dA = zero(A)
    autodiff(Reverse, kf_loglik,
        Duplicated(A, dA),
        Duplicated(copy(p.B), zero(p.B)),
        Duplicated(copy(p.C), zero(p.C)),
        Duplicated(copy(p.mu0), zero(p.mu0)),
        Duplicated(copy(p.Sigma0), zero(p.Sigma0)),
        Duplicated(copy(p.D), zero(p.D)),
        Duplicated(deepcopy(p.obs), [zeros(2) for _ in p.obs]),
        Duplicated(deepcopy(p.sol), Enzyme.make_zero(deepcopy(p.sol))),
        Duplicated(deepcopy(p.cache), Enzyme.make_zero(deepcopy(p.cache))))
    g[1] = -dA[1, 1]
end

params = (; B = B_opt, C = C_opt, D = D_opt, obs = obs_data,
    mu0 = mu0_opt, Sigma0 = Sigma0_opt, sol = ws_opt.output, cache = ws_opt.cache)

optf = OptimizationFunction(neg_loglik; grad = neg_loglik_grad!)
optprob = OptimizationProblem(optf, [0.90], params)
optsol = solve(optprob, LBFGS())
optsol.u  # estimated beta (true value: 0.95)
```

## Quadratic and Generic Models

The same all-`Duplicated` pattern works for [`QuadraticStateSpaceProblem`](@ref), [`PrunedQuadraticStateSpaceProblem`](@ref), and [`StateSpaceProblem`](@ref). Replace the constructor and add the extra arguments (`A_0`, `A_1`, `A_2`, `C_0`, `C_1`, `C_2` for quadratic; callback functions for generic) as separate `Duplicated` parameters. See the [Quadratic Models](@ref) tutorial for an Enzyme example with quadratic problems.

## Important Notes

- All arguments to the likelihood function that flow into the problem struct must be `Duplicated`, not `Const`. This is because Enzyme tracks activity at the struct level.
- Shadow copies for `sol` and `cache` buffers must be zero-initialized using `Enzyme.make_zero(deepcopy(...))`. Using plain `deepcopy` produces `NaN` gradients.
- The `Optimization.jl` integration requires an explicit `grad` function because `AutoEnzyme()` cannot directly handle the all-Duplicated requirement. The gradient function calls `Enzyme.autodiff` manually.
- Avoid calling `GC.gc()` inside functions differentiated by Enzyme -- this can cause segfaults when combined with `BenchmarkTools`.
- See the [Workspace API](@ref) page for details on `init`, `solve!`, and `StateSpaceWorkspace`.
- For small models (N ≤ 5), [ForwardDiff AD](@ref) offers a simpler alternative with comparable performance and no `Duplicated` bookkeeping.
