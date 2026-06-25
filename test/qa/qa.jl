using SciMLTesting, DifferenceEquations, Test
using JET
using LinearAlgebra

run_qa(
    DifferenceEquations;
    JET = nothing,            # JET is run below as bespoke report_call cases (issue #187), not package-wide
    explicit_imports = true,
    ei_kwargs = (;
        # Names re-exported by a dependency rather than imported from their owner.
        all_explicit_imports_via_owners = (;
            ignore = (
                :KeywordArgSilent,   # SciMLBase (imported from DiffEqBase)
                :get_concrete_p,     # SciMLBase (imported from DiffEqBase)
                :get_concrete_u0,    # SciMLBase (imported from DiffEqBase)
                :isconcreteu0,       # SciMLBase (imported from DiffEqBase)
                :promote_u0,         # SciMLBase (imported from DiffEqBase)
                :ismutable,          # Base (imported from StaticArrays)
            ),
        ),
        all_qualified_accesses_via_owners = (;
            ignore = (
                :__solve,            # SciMLBase (accessed via DiffEqBase)
            ),
        ),
        # Non-public names of dependency packages (go public as those base libs release).
        all_qualified_accesses_are_public = (;
            ignore = (
                :var"@propagate_inbounds",  # Base
                :Default,                    # SciMLBase.ReturnCode
                :Success,                    # SciMLBase.ReturnCode
                :T,                          # SciMLBase.ReturnCode
                :__solve,                    # DiffEqBase
                :build_solution,             # SciMLBase
                :check_prob_alg_pairing,     # DiffEqBase
                :get_concrete_problem,       # DiffEqBase
                :getindepsym,                # SciMLBase
                :getindepsym_defaultt,       # SciMLBase
                :init,                       # CommonSolve
                :solve,                      # CommonSolve
                :solve!,                     # CommonSolve
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :var"@add_kwonly",       # SciMLBase
                :AbstractDEAlgorithm,    # SciMLBase
                :AbstractDEProblem,      # SciMLBase
                :AbstractRODESolution,   # SciMLBase
                :ConstantInterpolation,  # SciMLBase
                :KeywordArgSilent,       # DiffEqBase
                :NullParameters,         # SciMLBase
                :build_solution,         # SciMLBase
                :get_concrete_p,         # DiffEqBase
                :get_concrete_u0,        # DiffEqBase
                :init,                   # CommonSolve
                :isconcreteu0,           # DiffEqBase
                :ismutable,              # StaticArrays
                :promote_tspan,          # SciMLBase
                :promote_u0,             # DiffEqBase
                :solve,                  # CommonSolve
                :solve!,                 # CommonSolve
            ),
        ),
    ),
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
