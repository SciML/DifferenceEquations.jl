# ForwardDiff AD

DifferenceEquations.jl works with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) out of the box for computing gradients and Jacobians. ForwardDiff requires no shadow arrays, no activity annotations, and no workspace pre-allocation -- just wrap your function in `ForwardDiff.gradient`.

!!! tip "When to use ForwardDiff vs Enzyme"

    | Scenario | Recommendation |
    |----------|---------------|
    | Small models (N ≤ 5 states), few parameters | **ForwardDiff** -- same speed as Enzyme reverse, zero setup cost |
    | Large models (N ≥ 10 states) | **Enzyme reverse** -- scales as O(1) backward passes vs O(N²) forward passes |
    | Many parameters (e.g., noise perturbation over T periods) | **Enzyme reverse** -- ForwardDiff cost scales with parameter count |
    | Quick prototyping | **ForwardDiff** -- simpler API, no `Duplicated` bookkeeping |
    | Production estimation loops | **Enzyme reverse** -- lower memory, pre-allocated workspace |

    For `DirectIteration` problems where you differentiate with respect to the noise sequence, the effective parameter dimension is `K × T` (shocks × periods), not just `N²`. Even for small state dimensions, long horizons make Enzyme reverse the better choice.

## The Core Pattern

ForwardDiff propagates dual numbers through the computation. The key requirement is that all arrays must have a consistent element type (either `Float64` or `Dual{...}`). The pattern is:

1. **Write a scalar function of a vector.** ForwardDiff.gradient takes `f: ℝⁿ → ℝ`.
2. **Promote all arrays inside the function.** When `ForwardDiff.gradient` calls your function with a `Vector{Dual{...}}`, convert all other matrices to the same `Dual` element type so that caches are allocated correctly.
3. **Use the public `solve()` API.** Unlike Enzyme, ForwardDiff creates fresh caches each call (with the correct `Dual` element type), so the simple `solve(prob, alg)` path works directly.

```julia
_promote(::Type{T}, x::AbstractArray{T}) where {T} = x
_promote(::Type{T}, x::AbstractArray) where {T} = T.(x)
```

## Differentiating Joint Likelihood

```@example forwarddiff
using DifferenceEquations, LinearAlgebra, ForwardDiff, Random

N, K, M = 2, 1, 2
A = [0.8 0.1; -0.1 0.7]
B = [0.1; 0.0;;]
C = [1.0 0.0; 0.0 1.0]
H = [0.1 0.0; 0.0 0.1]
u0 = zeros(N)

Random.seed!(42)
noise = [randn(K) for _ in 1:5]
sim = solve(LinearStateSpaceProblem(A, B, u0, (0, 5); C, noise))
obs = [sim.z[t + 1] + 0.1 * randn(M) for t in 1:5]

# Type-promotion helper
_promote(::Type{T}, x::AbstractArray{T}) where {T} = x
_promote(::Type{T}, x::AbstractArray) where {T} = T.(x)

# Gradient of joint loglik w.r.t. vec(A)
function di_loglik(A_vec, B, C, u0, noise, obs, H)
    T_el = eltype(A_vec)
    A = reshape(A_vec, 2, 2)
    H_d = _promote(T_el, H)
    R = H_d * H_d'
    prob = LinearStateSpaceProblem(
        A, _promote(T_el, B), _promote(T_el, u0), (0, length(obs));
        C = _promote(T_el, C), observables_noise = R,
        observables = obs, noise = noise)
    sol = solve(prob, DirectIteration())
    return sol.logpdf
end

grad_A = ForwardDiff.gradient(
    a -> di_loglik(a, B, C, u0, noise, obs, H), vec(copy(A)))
```

## Differentiating the Kalman Filter

The [`KalmanFilter`](@ref) marginal log-likelihood works the same way.

```@example forwarddiff
# General Kalman loglik that promotes all inputs consistently
function kf_loglik(A, B, C, mu0, Sigma0, R, obs)
    T_el = promote_type(eltype(A), eltype(B), eltype(C),
        eltype(mu0), eltype(Sigma0), eltype(R))
    N_st = size(A, 1)
    prob = LinearStateSpaceProblem(
        _promote(T_el, A), _promote(T_el, B),
        zeros(T_el, N_st), (0, length(obs));
        C = _promote(T_el, C),
        u0_prior_mean = _promote(T_el, mu0),
        u0_prior_var = _promote(T_el, Sigma0),
        observables_noise = _promote(T_el, R),
        observables = obs)
    sol = solve(prob, KalmanFilter())
    return sol.logpdf
end

mu0 = zeros(N)
Sigma0 = Matrix(1.0 * I(N))
R = [0.01 0.0; 0.0 0.01]

grad_kf = ForwardDiff.gradient(
    a -> kf_loglik(reshape(a, N, N), B, C, mu0, Sigma0, R, obs), vec(copy(A)))
```

## Differentiating with Respect to Multiple Parameters

Because `kf_loglik` promotes all inputs via `promote_type`, you can differentiate with respect to any parameter.

```@example forwarddiff
# Gradient w.r.t. observation matrix C
grad_C = ForwardDiff.gradient(
    c_vec -> kf_loglik(A, B, reshape(c_vec, M, N), mu0, Sigma0, R, obs),
    vec(copy(C)))
```

```@example forwarddiff
# Gradient w.r.t. prior mean
grad_mu0 = ForwardDiff.gradient(
    m -> kf_loglik(A, B, C, m, Sigma0, R, obs), copy(mu0))
```

## Integration with Optimization.jl

ForwardDiff integrates with [Optimization.jl](https://github.com/SciML/Optimization.jl) via `AutoForwardDiff()`, which is simpler than the Enzyme path (no manual `Duplicated` bookkeeping).

```@example forwarddiff
using Optimization, OptimizationOptimJL

# Simulate data from a known model
Random.seed!(42)
T_opt = 200
B_opt = [0.0; 0.001;;]
C_opt = [0.09 0.67; 1.00 0.00]
R_opt = [0.01 0.0; 0.0 0.01]
prob_data = LinearStateSpaceProblem([0.95 6.2; 0.0 0.2], B_opt, zeros(2), (0, T_opt);
    C = C_opt, observables_noise = R_opt)
sol_data = solve(prob_data)
obs_data = sol_data.z[2:end]

# Objective: negative Kalman loglik as function of β = [A[1,1]]
mu0_opt = zeros(2)
Sigma0_opt = Matrix(1e-2 * I(2))

function neg_loglik(beta, p)
    A = [beta[1] 6.2; 0.0 0.2]
    return -kf_loglik(A, p.B, p.C, p.mu0, p.Sigma0, p.R, p.obs)
end

params = (; B = B_opt, C = C_opt, R = R_opt, obs = obs_data,
    mu0 = mu0_opt, Sigma0 = Sigma0_opt)

optf = OptimizationFunction(neg_loglik, AutoForwardDiff())
optprob = OptimizationProblem(optf, [0.90], params)
optsol = solve(optprob, LBFGS())
optsol.u  # estimated β (true value: 0.95)
```

## StaticArrays

ForwardDiff also works with `SVector`/`SMatrix` inputs. Construct static arrays from the dual-typed input vector inside the function.

```@example forwarddiff
using StaticArrays

function kf_loglik_static(A_vec, B, C, mu0, Sigma0, R, obs,
        ::Val{N_}, ::Val{M_}, ::Val{K_}) where {N_, M_, K_}
    T_el = eltype(A_vec)
    A = SMatrix{N_, N_}(reshape(A_vec, N_, N_))
    prob = LinearStateSpaceProblem(
        A, SMatrix{N_, K_}(T_el.(B)),
        SVector{N_}(zeros(T_el, N_)), (0, length(obs));
        C = SMatrix{M_, N_}(T_el.(C)),
        u0_prior_mean = SVector{N_}(T_el.(mu0)),
        u0_prior_var = SMatrix{N_, N_}(T_el.(Sigma0)),
        observables_noise = SMatrix{M_, M_}(T_el.(R)),
        observables = obs)
    sol = solve(prob, KalmanFilter())
    return sol.logpdf
end

obs_s = [SVector{M}(o) for o in obs]
grad_static = ForwardDiff.gradient(
    a -> kf_loglik_static(a, SMatrix{N,K}(B), SMatrix{M,N}(C),
        SVector{N}(mu0), SMatrix{N,N}(Sigma0), SMatrix{M,M}(R),
        obs_s, Val(N), Val(M), Val(K)),
    collect(vec(Matrix(A))))
```

!!! note

    ForwardDiff with StaticArrays does not improve AD performance for this package. The overhead of constructing `SMatrix{N,N,Dual{...}}` temporaries outweighs the benefit. StaticArrays are most useful for the primal solve (no AD) of small models.

## Quadratic and Generic Models

ForwardDiff works with all problem types: [`QuadraticStateSpaceProblem`](@ref), [`PrunedQuadraticStateSpaceProblem`](@ref), and [`StateSpaceProblem`](@ref). The same pattern applies — promote all arrays to the `Dual` element type inside the gradient function and call `solve(prob, DirectIteration())`.

## Important Notes

- **Type promotion is required.** All arrays flowing into the problem must have the same element type. Use `promote_type` across all inputs (as in `kf_loglik` above) or the `_promote` helper to convert `Float64` arrays to the `Dual` type.
- **Fresh allocation each call.** ForwardDiff creates new caches with `Dual` element types via `solve()`. This is unavoidable (unlike Enzyme, which reuses `Float64` caches with separate shadow arrays).
- **Chunk size.** `ForwardDiff.gradient` defaults to a chunk size of ~10, processing 10 partial derivatives per forward pass. For parameter count > 10, it runs multiple passes. This makes ForwardDiff cost scale linearly with the number of parameters being differentiated.
- **Observations stay `Float64`.** The `observables` (data) are not differentiated and can remain `Vector{Vector{Float64}}`. The solver's internal buffers are allocated with the `Dual` element type, so when `Float64` observations are copied in, the dual partials are zero — which is correct since observations are data, not parameters being differentiated.
- **DirectIteration noise sensitivity.** When differentiating `DirectIteration` w.r.t. the noise sequence, the parameter dimension is `K × T` (shocks × periods). Even for small state-space models, long time series make ForwardDiff expensive and Enzyme reverse the better choice.
