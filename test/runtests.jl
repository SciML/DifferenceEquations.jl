using Test
using Distributions
using Random
using LinearAlgebra
using DifferenceEquations
using DifferentiableStateSpaceModels
using DifferentiableStateSpaceModels.Examples

@testset "DifferenceEquations.jl" begin
    @testset "Linear model" begin
        A = [0.8 0.0; 0.1 0.7]
        B = Diagonal([0.1, 0.5])
        C = [0.5 0.5] # one observable
        R = [0.01]
        
        # Simulate data
        T = 10
        u0 =[0.0, 0.1]
        tspan = (1, T)
        
        prob1 = LinearStateSpaceProblem(A, B, C, u0, tspan, R=R)
        sol1 = solve(prob1, ConditionalGaussian())
        
        # prob2 = LinearStateSpaceProblem(A, B, C, u0, tspan, R=R, observables=sol1.z)
        # sol2 = solve(prob2, ConditionalGaussian())
        
        # prob3 = LinearStateSpaceProblem(A, B, C, u0, tspan, R=R, observables=sol2.z, noise=DefinedNoise(sol2.n))
        # sol3 = solve(prob3, ConditionalGaussian())
        
        # @test sol2.n == sol3.n
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