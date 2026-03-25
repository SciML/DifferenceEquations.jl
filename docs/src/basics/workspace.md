# Workspace API

The workspace API provides a pre-allocated, reusable solving pattern via `init` and `solve!`. This avoids repeated memory allocation when solving the same type of problem many times, and is required for compatibility with Enzyme.jl reverse-mode AD.

```@docs
StateSpaceWorkspace
```

## Creating and Using a Workspace

```@docs
DifferenceEquations.init
DifferenceEquations.solve!
```

## Basic Usage

```@example workspace
using DifferenceEquations, LinearAlgebra, Random
A = [0.95 6.2; 0.0 0.2]
B = [0.0; 0.01;;]
C = [0.09 0.67; 1.00 0.00]
u0 = zeros(2)
prob = LinearStateSpaceProblem(A, B, u0, (0, 5); C)
ws = init(prob, DirectIteration())
sol = solve!(ws)
sol.u[end]
```

## Cache Reuse

Calling `solve!(ws)` again on the same workspace reuses all previously allocated buffers. The internal state is automatically zeroed before each solve, so there is no need to manually reset anything between calls. This makes the workspace pattern ideal for tight loops:

```julia
ws = init(prob, DirectIteration())
for i in 1:1000
    sol = solve!(ws)
    # process sol...
end
```

## When to Use

The workspace API is useful in the following scenarios:

- **Enzyme AD**: Enzyme requires pre-allocated buffers passed as `Duplicated` arguments. The workspace pattern via `init`/`solve!` is the recommended way to use Enzyme with DifferenceEquations.jl. See [Enzyme AD](@ref) for details.
- **Repeated solves in optimization loops**: When solving the same problem structure many times (e.g., during parameter estimation), the workspace avoids allocating new arrays on every iteration.
- **Performance-critical code**: Eliminating allocations reduces GC pressure and improves performance, especially for small to medium-sized problems.
