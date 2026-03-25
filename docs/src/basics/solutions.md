# Solutions

```@docs
StateSpaceSolution
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `u` | `Vector{Vector{T}}` | State trajectory |
| `t` | Range | Time values |
| `z` | `Vector{Vector{T}}` or `nothing` | Observations |
| `W` | `Vector{Vector{T}}` or `nothing` | Noise sequence (DirectIteration only) |
| `P` | `Vector{Matrix{T}}` or `nothing` | Posterior covariances (KalmanFilter only) |
| `logpdf` | `Float64` | Log-likelihood (0.0 if no observables) |
| `retcode` | `Symbol` | `:Success` or `:Default` |
| `prob` | Problem | Original problem |
| `alg` | Algorithm | Algorithm used |

## Symbolic Indexing

If `syms` or `obs_syms` were provided when constructing the problem, the solution supports symbolic indexing:

```julia
prob = LinearStateSpaceProblem(A, B, u0, (0, 10); C, syms=[:x, :y], obs_syms=[:obs1, :obs2])
sol = solve(prob)

# Access state variables by name
sol[:x]    # vector of :x values across all time steps
sol[:obs1] # vector of :obs1 observations across all time steps
```

## Standard Indexing

Solutions support standard Julia indexing to access states at specific time steps:

```julia
sol = solve(prob)

sol[1]     # state at t=0 (initial condition)
sol[end]   # state at the final time step
sol.u[3]   # state at the third time index
sol.z[2]   # observation at the second time index (if C was provided)
```

## DataFrame Conversion

The state trajectory can be converted to a DataFrame. Column names come from `syms` if provided. Note that only the state variables (not observations) appear in the DataFrame.

```@example solutions_df
using DifferenceEquations, LinearAlgebra, DataFrames
A = [0.95 6.2; 0.0 0.2]
B = [0.0; 0.01;;]
C = [0.09 0.67; 1.00 0.00]
prob = LinearStateSpaceProblem(A, B, zeros(2), (0, 5); C,
    syms = [:capital, :productivity], obs_syms = (:output, :investment))
sol = solve(prob)
DataFrame(sol)
```
