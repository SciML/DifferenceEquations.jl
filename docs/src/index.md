# DifferenceEquations.jl

DifferenceEquations.jl solves initial value problems for deterministic and stochastic difference equations, with differentiable solvers and filters. Automatic differentiation is powered by [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) (reverse and forward mode). The package is part of the [SciML](https://sciml.ai/) ecosystem.

## Features

  - **Linear, quadratic, and generic state-space models** -- [`LinearStateSpaceProblem`](@ref), [`QuadraticStateSpaceProblem`](@ref), [`PrunedQuadraticStateSpaceProblem`](@ref), and [`StateSpaceProblem`](@ref) with user-defined callbacks.
  - **Kalman filter** for computing the marginal log-likelihood of linear Gaussian models via [`KalmanFilter`](@ref).
  - **Differentiable via Enzyme.jl** -- reverse-mode and forward-mode AD through both simulation (`DirectIteration`) and filtering (`KalmanFilter`).
  - **StaticArrays support** for small models where heap allocations dominate runtime.
  - **Workspace API** -- [`StateSpaceWorkspace`](@ref) with `init` / `solve!` for allocation-free repeated solves (useful inside AD and tight loops).
  - **SciML ecosystem integration** -- `EnsembleProblem` for Monte Carlo, plot recipes, `DataFrame` conversion, symbolic indexing, and `remake`.

## Installation

To install DifferenceEquations.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("DifferenceEquations")
```

## Quick Example

```@example index
using DifferenceEquations, LinearAlgebra
A = [0.95 0.1; 0.0 0.2]
B = [0.0; 0.01;;]
u0 = zeros(2)
T = 10
prob = LinearStateSpaceProblem(A, B, u0, (0, T))
sol = solve(prob)
sol.u[end]  # final state
```

## Mathematical Background

The general class of discrete-time state-space models supported by this package takes an initial condition ``u_0`` and an evolution equation

```math
u_{n+1} = f(u_n, p, t_n) + g(u_n, p, t_n)\, w_{n+1}
```

for transition function ``f``, noise coefficient ``g``, and IID noise shocks ``w_{n+1}``. The parameter vector ``p`` is potentially differentiable.

An optional observation equation relates the latent state to measured data:

```math
z_n = h(u_n, p, t_n) + v_n
```

where ``v_n`` is observation noise and ``z_n`` may have a different dimension from ``u_n``.

### Specializations

  - **Linear**: ``f(u) = A\,u``, ``g(u) = B``, ``h(u) = C\,u``. Solved by [`DirectIteration`](@ref) or [`KalmanFilter`](@ref). See [`LinearStateSpaceProblem`](@ref).
  - **Quadratic**: Adds second-order terms ``u^\top A_2\, u`` to both transition and observation. Useful for pruned perturbation solutions of DSGE models. See [`QuadraticStateSpaceProblem`](@ref) and [`PrunedQuadraticStateSpaceProblem`](@ref).
  - **Generic**: User-supplied `transition` and `observation` callbacks. See [`StateSpaceProblem`](@ref).

When the system is linear, the shocks are Gaussian, and a Gaussian prior is provided, the [`KalmanFilter`](@ref) computes the exact marginal log-likelihood. For all other cases, [`DirectIteration`](@ref) iterates the state forward and (optionally) accumulates a joint log-likelihood.

!!! note

    Boundary value problems and difference-algebraic equations are not in scope. See [DifferentiableStateSpaceModels.jl](https://github.com/HighDimensionalEconLab/DifferentiableStateSpaceModels.jl) for perturbation solutions and DSGEs.

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:

      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
