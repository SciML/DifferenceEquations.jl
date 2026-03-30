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

Calling `solve!(ws)` again on the same workspace reuses all previously allocated buffers. The solver fully overwrites all output arrays on each call, so no manual reset is needed between calls. This makes the workspace pattern ideal for tight loops:

```julia
ws = init(prob, DirectIteration())
for i in 1:1000
    sol = solve!(ws)
    # process sol...
end
```

You can also change the problem between calls for parameter sweeps using `remake`:

```julia
ws = init(prob, DirectIteration())
for a11 in [0.9, 0.95, 1.0]
    ws.prob = remake(ws.prob; A = [a11 6.2; 0.0 0.2])
    sol = solve!(ws)
    # process sol.logpdf...
end
```

## Endpoints-Only Mode (`save_everystep=false`)

Pass `save_everystep=false` to `init` to allocate minimal 2-element buffers. The solver stores only the initial and final states, while still correctly accumulating `logpdf`:

```@example workspace
ws_ep = init(prob, DirectIteration(); save_everystep=false)
sol_ep = solve!(ws_ep)
length(sol_ep.u)  # 2: [u_initial, u_final]
```

This is especially useful for ForwardDiff gradient computation, where reducing the number of dual-number allocations from O(T) to O(1) gives significant speedups (up to 7x with StaticArrays):

```julia
# ForwardDiff benefits from save_everystep=false
function neg_loglik(params)
    prob = make_problem(params)
    return -solve(prob, ConditionalLikelihood(); save_everystep=false).logpdf
end
ForwardDiff.gradient(neg_loglik, params0)
```

## When to Use

The workspace API is useful in the following scenarios:

- **Enzyme AD**: Enzyme requires pre-allocated buffers passed as `Duplicated` arguments. The workspace pattern via `init`/`solve!` is the recommended way to use Enzyme with DifferenceEquations.jl. See [Enzyme AD](@ref) for details.
- **Repeated solves in optimization loops**: When solving the same problem structure many times (e.g., during parameter estimation), the workspace avoids allocating new arrays on every iteration.
- **ForwardDiff with `save_everystep=false`**: Combining the workspace API with endpoints-only mode minimizes dual-number allocations, giving the best ForwardDiff performance.
- **Performance-critical code**: Eliminating allocations reduces GC pressure and improves performance, especially for small to medium-sized problems.
