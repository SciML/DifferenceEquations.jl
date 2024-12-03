# DifferenceEquations.jl

Solving difference equations with DifferenceEquations.jl and the SciML ecosystem.

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DifferenceEquations/stable/)

[![codecov](https://codecov.io/gh/SciML/DifferenceEquations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/DifferenceEquations.jl)
[![Build Status](https://github.com/SciML/DifferenceEquations.jl/workflows/CI/badge.svg)](https://github.com/SciML/LinearSolvers.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/74699764ce224514c9632e2750e08f77c6d174c5ba7cd38297.svg?branch=main)](https://buildkite.com/julialang/linearsolve-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

This package simulates for **initial value problems** for deterministic and stochastic difference equations, with or without a separate observation equation.  In addition, the package provides likelihoods for some standard filters for estimating state-space models.

Relative to existing solvers, this package is intended to provide **differentiable solvers and filters**.  For example, you can simulate a linear gaussian state space model and find the gradient of the solution with respect to the model primitives.  Similarly, the likelihood for the Kalman Filter can itself be differentiated with respect to the underlying model primitives.  This makes the package especially amenable to estimation and calibration, where the entire solution blocks become auto-differentiable.

**NOTE**: While the features of this package have unit tests and documentation, the package is in a pre-release state. If you require additional features or flexibility, you will need to contribute them.


## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/DifferenceEquations/stable/). Use the
[in-development documentation](https://docs.sciml.ai/DifferenceEquations/dev/) for the version of
the documentation, which contains the unreleased features.
