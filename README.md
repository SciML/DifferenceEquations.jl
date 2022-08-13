# DifferenceEquations.jl
Solving difference equations with DifferenceEquations.jl and the SciML ecosystem.

[![Github Action CI](https://github.com/SciML/DifferenceEquations.jl/workflows/CI/badge.svg)](https://github.com/SciML/DifferenceEquations.jl/actions)
[![Coverage Status](https://coveralls.io/repos/github/SciML/DifferenceEquations.jl/badge.svg?branch=main)](https://coveralls.io/github/SciML/DifferenceEquations.jl?branch=main)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SciML.github.io/DifferenceEquations.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SciML.github.io/DifferenceEquations.jl/dev)


This package simulates for **initial value problems** for deterministic and stochastic difference equations, with or without a separate observation equation.  In addition, the package provides likelihoods for some standard filters for estimating state-space models.

Relative to existing solvers, this package is intended to provide **differentiable solvers and filters**.  For example, you can simulate a linear gaussian state space model and find the gradient of the solution with respect to the model primitives.  Similarly, the likelihood for the Kalman Filter can itself be differentiated with respect to the underlying model primitives.  This makes the package especially amenable to estimation and calibration where the entire solution blocks become auto-differentiable.

**NOTE**: While the features of this package have unit tests and documentation, the package is in a pre-release state.  if you require additional features or flexibility, you will need to contribute them.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://DifferenceEquations.sciml.ai/stable/). Use the
[in-development documentation](https://DifferenceEquations.sciml.ai/dev/) for the version of
the documentation, which contains the unreleased features.