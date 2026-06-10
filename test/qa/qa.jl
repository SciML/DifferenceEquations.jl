using DifferenceEquations
using Aqua
using ExplicitImports
using JET
using LinearAlgebra
using Distributions
using Test

@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(DifferenceEquations)
    Aqua.test_ambiguities(DifferenceEquations, recursive = false)
    Aqua.test_deps_compat(DifferenceEquations)
    Aqua.test_piracies(DifferenceEquations)
    Aqua.test_project_extras(DifferenceEquations)
    Aqua.test_stale_deps(DifferenceEquations)
    Aqua.test_unbound_args(DifferenceEquations)
    Aqua.test_undefined_exports(DifferenceEquations)
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(DifferenceEquations) === nothing
    @test check_no_stale_explicit_imports(DifferenceEquations) === nothing
end

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
