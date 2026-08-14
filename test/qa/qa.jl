using SciMLTesting, DifferenceEquations, Test
using JET
using LinearAlgebra

run_qa(
    DifferenceEquations;
    JET = nothing,            # JET is run below as bespoke report_call cases (issue #187), not package-wide
    # Public APIs intentionally re-exported from CommonSolve and SciMLBase.
    reexports_allow = (:init, :remake, :solve, :solve!),
)

# JET cases tied to issue #187 are bespoke `report_call`s on specific solve paths:
# `report_package` (the run_qa JET path) does not surface them, so they stay here.
@testset "JET static analysis" begin
    @testset "LinearStateSpaceProblem DirectIteration" begin
        A = [0.9 0.1; 0.0 0.95]
        B = [0.1 0.0; 0.0 0.1]
        C = [1.0 0.0]
        u0 = [1.0, 0.5]
        tspan = (0, 10)
        noise = randn(2, 10)

        prob = LinearStateSpaceProblem(A, B, u0, tspan; noise)
        rep = JET.report_call(solve, (typeof(prob), typeof(DirectIteration())))
        @test_broken length(JET.get_reports(rep)) == 0  # JET: no matching method get_concrete_noise(::LinearStateSpaceProblem, ::Int64) — see https://github.com/SciML/DifferenceEquations.jl/issues/187
    end

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
        @test_broken length(JET.get_reports(rep)) == 0  # JET: no matching method get_observable(::Matrix{Float64}, ::Int64) — see https://github.com/SciML/DifferenceEquations.jl/issues/187
    end

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
