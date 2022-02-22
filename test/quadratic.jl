using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using FiniteDiff: finite_difference_gradient

# joint case
function joint_likelihood_2(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, observables, D; kwargs...)
    problem = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, (0, size(noise, 2));
                                         obs_noise = MvNormal(Diagonal(abs2.(D))), noise,
                                         observables, kwargs...)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end

# Matrices from RBC
A_0_rbc = [-7.824904812740593e-5, 0.0]
A_1_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
A_2_rbc = cat([-0.00019761505863889124 0.03375055315837927; 0.0 0.0],
              [0.03375055315837913 3.128758481817603; 0.0 0.0]; dims = 3)
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_0_rbc = [7.824904812740593e-5, 0.0]
C_1_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
C_2_rbc = cat([-0.00018554166974717046 0.0025652363153049716; 0.0 0.0],
              [0.002565236315304951 0.3132705036896446; 0.0 0.0]; dims = 3)
D_rbc = [0.1, 0.1]
u0_rbc = zeros(2)

path = joinpath(pkgdir(DifferenceEquations), "test", "data")
file_prefix = "RBC"
observables_rbc = readdlm(joinpath(path, "$(file_prefix)_observables.csv"), ',')' |> collect
noise_rbc = readdlm(joinpath(path, "$(file_prefix)_noise.csv"), ',')' |> collect

# Data and Noise
T = 5
observables_rbc = observables_rbc[:, 1:T]
noise_rbc = noise_rbc[:, 1:T]

@testset "quadratic rbc joint likelihood" begin
    @test joint_likelihood_2(A_0_rbc, A_1_rbc, A_2_rbc, B_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_rbc,
                             noise_rbc, observables_rbc, D_rbc) ≈ -690.81094364573
    @inferred joint_likelihood_2(A_0_rbc, A_1_rbc, A_2_rbc, B_rbc, C_0_rbc, C_1_rbc, C_2_rbc,
                                 u0_rbc, noise_rbc, observables_rbc, D_rbc) # would this catch inference problems in the solve?
    gradient((args...) -> joint_likelihood_2(args..., observables_rbc, D_rbc), A_0_rbc, A_1_rbc,
             A_2_rbc, B_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_rbc, noise_rbc)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> joint_likelihood_2(args..., observables_rbc, D_rbc), A_0_rbc, A_1_rbc,
               A_2_rbc, B_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad,
               check_inferred = false)
end

# Load FVGQ data for checks
path = joinpath(pkgdir(DifferenceEquations), "test", "data")
file_prefix = "FVGQ20"
A_0_raw = readdlm(joinpath(path, "$(file_prefix)_A_0.csv"), ',')
A_0_FVGQ = vec(A_0_raw)
A_1_FVGQ = readdlm(joinpath(path, "$(file_prefix)_A_1.csv"), ',')
A_2_raw = readdlm(joinpath(path, "$(file_prefix)_A_2.csv"), ',')
A_2_FVGQ = reshape(A_2_raw, length(A_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
B_FVGQ = readdlm(joinpath(path, "$(file_prefix)_B.csv"), ',')
C_0_raw = readdlm(joinpath(path, "$(file_prefix)_C_0.csv"), ',')
C_0_FVGQ = vec(C_0_raw)
C_1_FVGQ = readdlm(joinpath(path, "$(file_prefix)_C_1.csv"), ',')
C_2_raw = readdlm(joinpath(path, "$(file_prefix)_C_2.csv"), ',')
C_2_FVGQ = reshape(C_2_raw, length(C_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
# D_raw = readdlm(joinpath(path, "$(file_prefix)_D.csv"); header = false)))
D_FVGQ = ones(6) * 1e-3
observables_FVGQ = readdlm(joinpath(path, "$(file_prefix)_observables.csv"), ',')' |> collect
noise_FVGQ = readdlm(joinpath(path, "$(file_prefix)_noise.csv"), ',')' |> collect
u0_FVGQ = zeros(size(A_1_FVGQ, 1))

@testset "quadratic FVGQ joint likelihood" begin
    @test joint_likelihood_2(A_0_FVGQ, A_1_FVGQ, A_2_FVGQ, B_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ,
                             u0_FVGQ, noise_FVGQ, observables_FVGQ, D_FVGQ) ≈ -1.473244794713955e10
    @inferred joint_likelihood_2(A_0_FVGQ, A_1_FVGQ, A_2_FVGQ, B_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ,
                                 u0_FVGQ, noise_FVGQ, observables_FVGQ, D_FVGQ)
    gradient((args...) -> joint_likelihood_2(args..., observables_FVGQ, D_FVGQ), A_0_FVGQ, A_1_FVGQ,
             A_2_FVGQ, B_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ, u0_FVGQ, noise_FVGQ)
    #test_rrule(Zygote.ZygoteRuleConfig(), (args...) -> joint_likelihood_2(args..., observables, D),
    #    A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise; rrule_f = rrule_via_ad,
    #    check_inferred = false)
end