using Test
using Distributions
using Random
using LinearAlgebra
using DifferenceEquations
using DifferentiableStateSpaceModels
using DifferentiableStateSpaceModels.Examples

@testset "DifferenceEquations.jl" begin
    @testset "Linear model" begin
        include("dssm.jl")
    end

    @testset "Nonlinear model" begin
        # Grab the model
        m = @include_example_module(Examples.rbc_observables)
        p_f = (ρ=0.2, δ=0.02, σ=0.01, Ω_1=0.1)
        p_d = (α=0.5, β=0.95)

        # Generate cache, create perutrbation solution
        c = SolverCache(m, Val(1), p_d)
        sol = generate_perturbation(m, p_d, p_f; cache = c)

        # Timespan to simulate across
        T = 500

        # Set initial state
        u0 = zeros(m.n_x)

        # Construct problem with no observables
        problem = StateSpaceProblem(
            DifferentiableStateSpaceModels.dssm_evolution,
            DifferentiableStateSpaceModels.dssm_volatility,
            DifferentiableStateSpaceModels.dssm_observation,
            u0,
            (1,T),
            sol
        )

        # Solve the model, this generates
        # simulated data.
        simul = DifferenceEquations.solve(problem, ConditionalGaussian())

        # Extract the observables, latent noise, and latent states.
        z, n, u = simul.z, simul.n, simul.u

        # Now solve using the previous data as observables.
        # Solving this problem also includes a likelihood.
        problem_data = StateSpaceProblem(
            DifferentiableStateSpaceModels.dssm_evolution,
            DifferentiableStateSpaceModels.dssm_volatility,
            DifferentiableStateSpaceModels.dssm_observation,
            u0,
            (1,T),
            sol,
            observables = z
        )

        # Generate likelihood.
        s2 = DifferenceEquations.solve(problem_data, ConditionalGaussian())
    end
end