using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

function joint_likelihood_1(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                      noise, observables, kwargs...)
    return solve(problem, DirectIteration()).logpdf
end

# CRTU has problems with generating random MvNormal, so just testing diagonals
function kalman_likelihood(A, B, C, u0, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                      u0_prior = MvNormal(u0, diagm(ones(length(u0)))),
                                      noise = nothing, observables, kwargs...)
    return solve(problem).logpdf
end

# Matrices from RBC
A_rbc = [0.9568351489231076 6.209371005755285;
         3.0153731819288737e-18 0.20000000000000007]
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = MvNormal(Diagonal(abs2.([0.1, 0.1])))
u0_rbc = zeros(2)

observables_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"),
                          ',')' |> collect
noise_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ',')' |>
            collect
# Data and Noise
T = 5
observables_rbc = observables_rbc[:, 1:T]
noise_rbc = noise_rbc[:, 1:T]

@testset "basic inference" begin
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, noise = noise_rbc,
                                   observables = observables_rbc)
    @inferred LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2));
                                      C = C_rbc, observables_noise = D_rbc, noise = noise_rbc,
                                      observables = observables_rbc)

    sol = solve(prob)
    @inferred solve(prob)

    DiffEqBase.get_concrete_problem(prob, false)
    @inferred DiffEqBase.get_concrete_problem(prob, false)

    joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc)
    @inferred joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc)
end

@testset "basic kalman inference" begin
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, observables = observables_rbc,
                                   u0_prior = MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))))
    @inferred LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2));
                                      C = C_rbc, observables_noise = D_rbc,
                                      observables = observables_rbc,
                                      u0_prior = MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))))

    sol = solve(prob)
    @inferred solve(prob)

    prob_concrete = DiffEqBase.get_concrete_problem(prob, false)
    @inferred DiffEqBase.get_concrete_problem(prob, false)

    kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc)
    @inferred kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc)
end

gradient((args...) -> joint_likelihood_1(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
         u0_rbc, noise_rbc)

@testset "linear rbc joint likelihood" begin
    @test joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) ≈
          -690.9407412360038
    @inferred joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) # 
    gradient((args...) -> joint_likelihood_1(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
             u0_rbc, noise_rbc)

    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> joint_likelihood_1(args..., observables_rbc, D_rbc), A_rbc, B_rbc,
               C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)
end

gradient((args...) -> kalman_likelihood(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
         u0_rbc)

@testset "linear rbc kalman likelihood" begin
    @test kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc) ≈
          -607.3698273765538
    @inferred kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc) # would this catch inference problems in the solve?
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> kalman_likelihood(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
               u0_rbc; rrule_f = rrule_via_ad, check_inferred = false)
end

# Load FVGQ data for checks
A_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A.csv"), ',')
B_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
C_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C.csv"), ',')
D_FVGQ = MvNormal(Diagonal(abs2.(ones(6) * 1e-3)))

observables_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations),
                                    "test/data/FVGQ20_observables.csv"), ',')' |> collect

noise_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"), ',')' |>
             collect
u0_FVGQ = zeros(size(A_FVGQ, 1))

@testset "linear FVGQ joint likelihood" begin
    @test joint_likelihood_1(A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ, observables_FVGQ,
                             D_FVGQ) ≈ -1.4648817357717388e9
    @inferred joint_likelihood_1(A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ, observables_FVGQ,
                                 D_FVGQ)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> joint_likelihood_1(args..., observables_FVGQ, D_FVGQ), A_FVGQ, B_FVGQ,
               C_FVGQ, u0_FVGQ, noise_FVGQ; rrule_f = rrule_via_ad, check_inferred = false)
end

@testset "linear FVGQ Kalman" begin
    # Note: set rtol to be higher than the default case because of huge gradient numbers
    @test kalman_likelihood(A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, observables_FVGQ, D_FVGQ) ≈
          -108.52706300389917
    gradient((args...) -> kalman_likelihood(args..., observables_FVGQ, D_FVGQ), A_FVGQ, B_FVGQ,
             C_FVGQ, u0_FVGQ)

    # TODO: this is not turned on because the numbers explode.  Need better unit test data to be interior
    # test_rrule(Zygote.ZygoteRuleConfig(), (args...) -> kalman_likelihood(args..., observables, D),
    #            A, B, C, u0; rrule_f = rrule_via_ad, check_inferred = false, rtol = 1e-5)
end

using Plots
@testset "plotting" begin
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, noise = noise_rbc,
                                   observables = observables_rbc, syms = (:a, :b))
    sol = solve(prob)
    plot(sol)
end

# random initial conditions
@testset "ensemble" begin
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))),
                                   (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, noise = noise_rbc,
                                   observables = observables_rbc, syms = (:a, :b))
    sol2 = solve(EnsembleProblem(prob), DirectIteration(), EnsembleThreads(); trajectories = 10)
    plot(sol2)
    summ = EnsembleSummary(sol2)
    plot(summ)
end
