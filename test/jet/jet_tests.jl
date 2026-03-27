using JET
using DifferenceEquations
using LinearAlgebra
using Distributions
using Test

@testset "JET static analysis" begin
    # Test LinearStateSpaceProblem with DirectIteration
    @testset "LinearStateSpaceProblem DirectIteration" begin
        A = [0.9 0.1; 0.0 0.95]
        B = [0.1 0.0; 0.0 0.1]
        C = [1.0 0.0]
        u0 = [1.0, 0.5]
        tspan = (0, 10)
        noise = randn(2, 10)

        prob = LinearStateSpaceProblem(A, B, u0, tspan; noise)
        rep = JET.report_call(solve, (typeof(prob), typeof(DirectIteration())))
        @test length(JET.get_reports(rep)) == 0
    end

    # Test LinearStateSpaceProblem with KalmanFilter
    @testset "LinearStateSpaceProblem KalmanFilter" begin
        A = [0.9 0.1; 0.0 0.95]
        B = [0.1 0.0; 0.0 0.1]
        C = [1.0 0.0]
        u0 = [1.0, 0.5]
        tspan = (0, 10)
        observables = randn(1, 10)
        observables_noise = Diagonal([0.1])
        u0_prior_mean = [0.0, 0.0]
        u0_prior_var = [1.0 0.0; 0.0 1.0]

        prob = LinearStateSpaceProblem(
            A, B, u0, tspan;
            C,
            u0_prior_mean,
            u0_prior_var,
            observables_noise,
            observables
        )
        rep = JET.report_call(solve, (typeof(prob), typeof(KalmanFilter())))
        @test length(JET.get_reports(rep)) == 0
    end

    # Test LinearStateSpaceProblem with observation equation but no noise
    @testset "LinearStateSpaceProblem with C, no noise" begin
        A = [0.9 0.1; 0.0 0.95]
        B = nothing
        C = [1.0 0.0]
        u0 = [1.0, 0.5]
        tspan = (0, 10)

        prob = LinearStateSpaceProblem(A, B, u0, tspan; C)
        rep = JET.report_call(solve, (typeof(prob), typeof(DirectIteration())))
        @test length(JET.get_reports(rep)) == 0
    end
end
