# StaticArrays

For small state-space models (typically 2--5 states), using [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) can significantly improve performance by eliminating heap allocations and enabling compiler optimizations such as loop unrolling.

## Example

```@example static
using DifferenceEquations, StaticArrays, LinearAlgebra
A = @SMatrix [0.95 6.2; 0.0 0.2]
B = @SMatrix [0.0; 0.01;;]
C = @SMatrix [0.09 0.67; 1.00 0.00]
u0 = @SVector zeros(2)
prob = LinearStateSpaceProblem(A, B, u0, (0, 10); C)
sol = solve(prob)
sol.u[end]
```

## When to Use

StaticArrays are most beneficial when:

- **State dimensions are small**: The performance advantage is greatest for matrices up to roughly 10x10. Beyond that, the compile-time overhead and code size can outweigh the benefits.
- **Sizes are known at compile time**: StaticArrays encode their dimensions as type parameters, so the sizes must be fixed constants rather than runtime values.
- **You need stack allocation**: StaticArrays are stored on the stack rather than the heap, eliminating GC pressure entirely for small models.

For larger models or models where dimensions vary at runtime, use standard `Array` types instead.

## Bang-Bang Operators

The package internally uses "bang-bang" operators (e.g., `mul!!`, `copyto!!`, `assign!!`) that handle both mutable and immutable arrays transparently. When you pass `SMatrix` and `SVector` types, these operators return new immutable values rather than mutating in place. When you pass standard `Matrix` and `Vector` types, they mutate in place and return the result. This means you do not need to change any solver code to switch between static and dynamic arrays -- simply change the array types in your problem definition.

See [Internals](@ref) for the full list of bang-bang operators and their behavior.

## Supported Problem Types

StaticArrays work with all problem types:

- [`LinearStateSpaceProblem`](@ref) with `DirectIteration` — simulation and log-likelihood
- [`LinearStateSpaceProblem`](@ref) with `KalmanFilter` — filtering, smoothing, and log-likelihood
- [`QuadraticStateSpaceProblem`](@ref) and [`PrunedQuadraticStateSpaceProblem`](@ref) — second-order perturbation models
- [`StateSpaceProblem`](@ref) — generic callbacks using bang-bang operators

## Kalman Filter Example

```@example static
using DifferenceEquations, StaticArrays, LinearAlgebra, Random
Random.seed!(42)
A = SMatrix{2,2}(0.8*I(2))
B = SMatrix{2,1}([0.1; 0.05])
C = SMatrix{2,2}(1.0*I(2))
R = SMatrix{2,2}(0.01*I(2))
mu0 = @SVector zeros(2)
Sig0 = SMatrix{2,2}(1.0*I(2))
y = [SVector{2}(randn(2)) for _ in 1:10]
prob = LinearStateSpaceProblem(A, B, mu0, (0, 10); C,
    u0_prior_mean=mu0, u0_prior_var=Sig0,
    observables_noise=R, observables=y)
sol = solve(prob, KalmanFilter())
sol.logpdf
```

## AD Performance Note

Enzyme reverse-mode AD benefits significantly from StaticArrays at small dimensions (N ≤ 5), with 5--7x speedups over mutable arrays for both the primal and AD passes.

ForwardDiff with StaticArrays does not improve AD performance for this package. The overhead of constructing `SMatrix{N,N,Dual{...}}` temporaries outweighs the benefit. See [ForwardDiff AD](@ref) for details.
