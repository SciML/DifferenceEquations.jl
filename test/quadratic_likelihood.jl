using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

# joint case
function joint_likelihood_2(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, observables, D; kwargs...)
    problem = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, size(observables, 2)); C_0, C_1,
                                         C_2, observables_noise = D, noise, observables, kwargs...)
    return solve(problem).logpdf
end

# Matrices from RBC
A_0_rbc = [-7.824904812740593e-5, 0.0]
A_1_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
A_2_rbc = cat([-0.00019761505863889124 0.03375055315837927; 0.0 0.0],
              [0.03375055315837913 3.128758481817603; 0.0 0.0]; dims = 3)
B_2_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_0_rbc = [7.824904812740593e-5, 0.0]
C_1_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
C_2_rbc = cat([-0.00018554166974717046 0.0025652363153049716; 0.0 0.0],
              [0.002565236315304951 0.3132705036896446; 0.0 0.0]; dims = 3)
D_2_rbc = abs2.([0.1, 0.1])
u0_2_rbc = zeros(2)

observables_2_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"),
                            ',')' |> collect
noise_2_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ',')' |>
              collect

# Data and Noise
T = 5
observables_2_rbc = observables_2_rbc[:, 1:T]
noise_2_rbc = noise_2_rbc[:, 1:T]

@testset "basic inference" begin
    prob = QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, u0_2_rbc,
                                      (0, size(observables_2_rbc, 2)); C_0 = C_0_rbc, C_1 = C_1_rbc,
                                      C_2 = C_2_rbc, observables_noise = D_2_rbc,
                                      noise = noise_2_rbc, observables = observables_2_rbc)
    @inferred QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, u0_2_rbc,
                                         (0, size(observables_2_rbc, 2)); C_0 = C_0_rbc,
                                         C_1 = C_1_rbc, C_2 = C_2_rbc, observables_noise = D_2_rbc,
                                         noise = noise_2_rbc, observables = observables_2_rbc)

    DiffEqBase.get_concrete_problem(prob, false)
    @inferred DiffEqBase.get_concrete_problem(prob, false)

    sol = solve(prob)
    @inferred solve(prob)
end

@testset "quadratic rbc joint likelihood" begin
    @test joint_likelihood_2(A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc,
                             u0_2_rbc, noise_2_rbc, observables_2_rbc, D_2_rbc) ≈ -690.81094364573
    @inferred joint_likelihood_2(A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc,
                                 u0_2_rbc, noise_2_rbc, observables_2_rbc, D_2_rbc) # would this catch inference problems in the solve?
    gradient((args...) -> joint_likelihood_2(args..., observables_2_rbc, D_2_rbc), A_0_rbc, A_1_rbc,
             A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_2_rbc, noise_2_rbc)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> joint_likelihood_2(args..., observables_2_rbc, D_2_rbc), A_0_rbc,
               A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_2_rbc, noise_2_rbc;
               rrule_f = rrule_via_ad, check_inferred = false)
end

# Load FVGQ data for checks
A_0_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_0.csv"), ',')
A_0_FVGQ = vec(A_0_raw)
A_1_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_1.csv"), ',')
A_2_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_2.csv"), ',')
A_2_FVGQ = reshape(A_2_raw, length(A_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
B_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
C_0_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_0.csv"), ',')
C_0_FVGQ = vec(C_0_raw)
C_1_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_1.csv"), ',')
C_2_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_2.csv"), ',')
C_2_FVGQ = reshape(C_2_raw, length(C_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
# D_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "FVGQ_D.csv"); header = false)))
D_2_FVGQ = ones(6) * 1e-3
u0_2_FVGQ = zeros(size(A_1_FVGQ, 1))

observables_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations),
                                      "test/data/FVGQ20_observables.csv"), ',')' |> collect

noise_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"), ',')' |>
               collect

@testset "quadratic FVGQ joint likelihood" begin
    @test joint_likelihood_2(A_0_FVGQ, A_1_FVGQ, A_2_FVGQ, B_2_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ,
                             u0_2_FVGQ, noise_2_FVGQ, observables_2_FVGQ, D_2_FVGQ) ≈
          -1.4728927648336522e7
    @inferred joint_likelihood_2(A_0_FVGQ, A_1_FVGQ, A_2_FVGQ, B_2_FVGQ, C_0_FVGQ, C_1_FVGQ,
                                 C_2_FVGQ,
                                 u0_2_FVGQ, noise_2_FVGQ, observables_2_FVGQ, D_2_FVGQ)
    gradient((args...) -> joint_likelihood_2(args..., observables_2_FVGQ, D_2_FVGQ), A_0_FVGQ,
             A_1_FVGQ,
             A_2_FVGQ, B_2_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ, u0_2_FVGQ, noise_2_FVGQ)

    test_rrule(Zygote.ZygoteRuleConfig(),
               (A_0_FVGQ, C_1_FVGQ, u0_2_FVGQ) -> joint_likelihood_2(A_0_FVGQ,
                                                                     A_1_FVGQ,
                                                                     A_2_FVGQ,
                                                                     B_2_FVGQ,
                                                                     C_0_FVGQ,
                                                                     C_1_FVGQ,
                                                                     C_2_FVGQ,
                                                                     u0_2_FVGQ,
                                                                     noise_2_FVGQ,
                                                                     observables_2_FVGQ,
                                                                     D_2_FVGQ),
               A_0_FVGQ, C_1_FVGQ, u0_2_FVGQ;
               rrule_f = rrule_via_ad, check_inferred = false)

    # A little slow to run all of them all every time.  Important occasionally, though, since tests the gradient wrt the noise
    # test_rrule(Zygote.ZygoteRuleConfig(),
    #            (args...) -> joint_likelihood_2(args..., observables_FVGQ, D_FVGQ), A_0_FVGQ,
    #            A_1_FVGQ,
    #            A_2_FVGQ, B_2_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ, u0_2_FVGQ, noise_2_FVGQ;
    #            rrule_f = rrule_via_ad, check_inferred = false)
end
