# DifferenceEquations.jl

This package simulates for **initial value problems** for deterministic and stochastic difference equations, with or without a separate observation equation.  In addition, the package provides likelihoods for some standard filters for estimating state-space models.

Relative to existing solvers, this package is intended to provide **differentiable solvers and filters**.  For example, you can simulate a linear gaussian state space model and find the gradient of the solution with respect to the model primitives.  Similarly, the likelihood for the Kalman Filter can itself be differentiated with respect to the underlying model primitives.  This makes the package especially amenable to estimation and calibration where the entire solution blocks become auto-differentiable.

!!! note

    Boundary value problems and difference-algebraic equations are not in scope. See see [DifferentiableStateSpaceModels.jl](https://github.com/HighDimensionalEconLab/DifferentiableStateSpaceModels.jl) for experimental support for perturbation solutions and DSGEs.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
DifferenceEquations.jl in the standard way:

```julia
] DifferenceEquations
```

For additional functionality, you may want to add `Plots, DiffEqBase`.  If you want to explore differentiable filters, you can install `Zygote`



## Mathematical Specification of a Discrete Problem
For comparison, see the specification of the deterministic [Discrete Problem](https://diffeq.sciml.ai/latest/types/discrete_types/#Mathematical-Specification-of-a-Discrete-Problem) (albeit with a small difference in timing conventions) and the [SDE Problem](https://diffeq.sciml.ai/latest/types/sde_types/).  Other introductions can be found by [checking out DiffEqTutorials.jl](https://github.com/JuliaDiffEq/DiffEqTutorials.jl).


The general class of problems intended to be supported in this package is to take an initial condition, $u_0$, and an evolution equation

```math
u_{n+1} = f(u_n,p,t_n) + g(u_n,p,t_n) w_{n+1}
```

for some functions $f$ and $g$, and where $w_{n+1}$ are IID random shocks to the evolution equation.  The $p$ is a vector of potentially differentiable parameters.

In addition, there is an optional observation equation

```math
z_n = h(u_n, p, t_n) +  v_n
```

where $v_n$ is noisy observation error and the size of $z_n$ may be different from $u_n$.

A few notes on the structure:

1. Frequently, the $g$ provides the covariance structure so a reasonable default is $w_{n+1} \sim N(0,I)$, and $z_n \sim N(0, D)$ is a common observation error for some covariance matrix $D$.
2. If $f,g,h$ are all linear, and the shocks are both gaussian, then this is a linear gaussian state-space model.  Kalman filters can be used to calculate likelihoods and simulates can be executed with very little overhead.
3. ``t_n`` is the current time at which the map is applied where ``t_n = t_0 + n*dt`` (with `dt=1` being the default).
4. If $f, g, h$ are not functions of time, then it is a time-invariant state-space model.

## Current Status
At this point, the package does not cover all of the variations on these features. In particular,
1. It only supports linear and quadratic $f, g, h$ functions.  General $f,g$ will be next.
2. It only supports time-invariant functions
3. There is limited support for non-Gaussian $w_n$ and $v_n$ processes.
4. It does not support linear or quadratic functions parameterized by the $p$ vector for differentiation
5. There are some hardcoded types which prevent it from working with fully generic arrays
6. It does not support in-place vs. out-of-place, support static arrays, or matrix-free linear operators.
7. While many functions in the SciML framework are working, support is incomplete.
8. There is not complete coverage of gradients for the solution for all parameter inputs/etc.

To help contribute on filling in these features, see the [issues](https://github.com/SciML/DifferenceEquations.jl/issues).