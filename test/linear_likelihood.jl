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

A_kalman = [0.0495388  0.0109918  0.0960529   0.0767147  0.0404643;
            0.020344   0.0627784  0.00865501  0.0394004  0.0601155;
            0.0260677  0.039467   0.0344606   0.033846   0.00224089;
            0.0917289  0.081082   0.0341586   0.0591207  0.0411927;
            0.0837549  0.0515705  0.0429467   0.0209615  0.014668]
B_kalman = [0.589064  0.97337   2.32677;
            0.864922  0.695811  0.618615;
            2.07924   1.11661   0.721113;
            0.995325  1.8416    2.30442;
            1.76884   1.56082   0.749023]
C_kalman = [0.0979797  0.114992   0.0964536  0.110065   0.0946794;
            0.110095   0.0856981  0.0841296  0.0981172  0.0811817;
            0.109134   0.103406   0.112622   0.0925896  0.112384;
            0.0848231  0.0821602  0.099332   0.113586   0.115105]
D_kalman = MvNormal(Diagonal(abs2.(ones(4) * 0.1)))
u0_kalman = zeros(5)

observables_kalman = readdlm(joinpath(pkgdir(DifferenceEquations),
                                    "test/data/Kalman_observables.csv"), ',')' |> collect

noise_kalman = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/Kalman_noise.csv"), ',')' |>
             collect

@testset "linear non-square Kalman" begin
    @test kalman_likelihood(A_kalman, B_kalman, C_kalman, u0_kalman, observables_kalman, D_kalman) ≈
            329.7550738722514
    gradient((args...) -> kalman_likelihood(args..., observables_kalman, D_kalman), A_kalman, B_kalman,
             C_kalman, u0_kalman)

    # TODO: this is not turned on because the numbers explode.  Need better unit test data to be interior
    # test_rrule(Zygote.ZygoteRuleConfig(), (args...) -> kalman_likelihood(args..., observables, D),
    #            A, B, C, u0; rrule_f = rrule_via_ad, check_inferred = false, rtol = 1e-5)
end 

@testset "basic kalman failure" begin
    A = [1e20 0.0; 1e20 0.0]
    u0_prior = MvNormal(u0_rbc, diagm(1e10 * ones(length(u0_rbc))))
    prob = LinearStateSpaceProblem(A, B_rbc, u0_rbc, (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, observables = observables_rbc,
                                   u0_prior)
    sol = solve(prob)
    @test sol.logpdf ≈ -Inf
    @test sol.retcode != :Success
end

@testset "basic kalman failure gradient" begin
    A = [1e20 0.0; 1e20 0.0]
    u0_prior = MvNormal(u0_rbc, diagm(1e10 * ones(length(u0_rbc))))
    function fail_kalman(B_rbc)
        prob = LinearStateSpaceProblem(A, B_rbc, u0_rbc, (0, size(observables_rbc, 2)); C = C_rbc,
                                       observables_noise = D_rbc, observables = observables_rbc,
                                       u0_prior)
        return solve(prob).logpdf
    end
    @test gradient(fail_kalman, B_rbc)[1] ≈ [0.0; 0.0;;] # but hopefully gradients are ignored!
end