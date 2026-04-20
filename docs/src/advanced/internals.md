# Internals

This page documents the internal architecture of DifferenceEquations.jl. It is intended for developers who want to understand the package internals or extend the package with new problem types or algorithms.

## Architecture

The solving pipeline follows these stages:

1. **Problem construction**: The user creates a problem (e.g., `LinearStateSpaceProblem`) that encodes the model dynamics, parameters, and data.
2. **Algorithm dispatch**: `solve(prob)` or `solve(prob, alg)` selects the algorithm. If no algorithm is provided, the default is chosen based on the problem type and its fields.
3. **Workspace allocation**: `init(prob, alg)` allocates the solution output via `alloc_sol` and scratch workspace via `alloc_cache`, then wraps them in a `StateSpaceWorkspace`.
4. **Solve**: `solve!(ws)` runs the algorithm, which fully overwrites all solution and cache arrays during the time loop.
5. **Solution**: A `StateSpaceSolution` is returned containing the state trajectory, observations, noise, log-likelihood, and other results.

## Bang-Bang Operators

DifferenceEquations.jl uses a "bang-bang" (`!!`) convention for internal operators. These functions behave differently depending on whether their arguments are mutable or immutable:

- **Mutable arrays** (`Vector`, `Matrix`): The operator mutates the destination in place and returns it.
- **Immutable arrays** (`SVector`, `SMatrix`): The operator creates and returns a new value, since mutation is not possible.

This dual behavior allows the same algorithm code to work with both standard arrays and StaticArrays without any branching or specialization at the call site.

The main bang-bang operators are:

| Operator | Description |
|----------|-------------|
| `mul!!(C, A, B)` | Matrix multiply `A * B`, storing in `C` |
| `copyto!!(dest, src)` | Copy contents of `src` into `dest` |
| `assign!!(dest, i, val)` | Assign `val` to position `i` in `dest` |
| `cholesky!!(F, A)` | Compute the Cholesky factorization of `A` |
| `ldiv!!(Y, F, B)` | Solve `F \ B`, storing in `Y` |
| `transpose!!(dest, src)` | Transpose `src` into `dest` |

## Cache System

Each combination of problem type and algorithm defines two allocation functions:

- **`alloc_sol(prob, alg, T)`**: Allocates the output structure that will hold the solution (state trajectory, observations, noise, covariances, etc.). Returns a named tuple or struct of pre-allocated arrays.
- **`alloc_cache(prob, alg, T)`**: Allocates scratch workspace needed during the solve (temporary vectors, matrices for intermediate computations, etc.). Returns a named tuple or struct of pre-allocated buffers.

The solver loop fully overwrites all solution and cache arrays on each call, so no explicit zeroing step is needed between calls. For Enzyme AD, shadow copies should be zero-initialized via `Enzyme.make_zero(deepcopy(...))` at creation time (see [Enzyme AD](@ref)).

## Adding a New Problem Type

To add a new problem type to DifferenceEquations.jl, you need to implement the following methods:

| Method | Signature | Description |
|--------|-----------|-------------|
| `_noise_matrix(prob)` | `prob → Matrix` | Return the noise input matrix (e.g., `B` for linear models) |
| `_init_model_state!!(prob, cache)` | `prob, cache → cache` | Initialize any model-specific cache state before the time loop |
| `_transition!!(x_next, x, w, prob, cache, t)` | `x_next, x, w, prob, cache, t → x_next` | Compute the next state given current state `x` and noise `w` at time `t` |
| `_observation!!(y, x, prob, cache, t)` | `y, x, prob, cache, t → y` | Compute the observation given state `x` at time `t` |
| `alloc_sol(prob, alg, T)` | `prob, alg, Int → NamedTuple` | Allocate the solution output arrays for `T` time steps |
| `alloc_cache(prob, alg, T)` | `prob, alg, Int → NamedTuple` | Allocate scratch workspace for `T` time steps |

All transition and observation methods should follow the bang-bang convention: mutate the first argument if it is mutable, otherwise return a new value. This ensures compatibility with both standard arrays and StaticArrays.
