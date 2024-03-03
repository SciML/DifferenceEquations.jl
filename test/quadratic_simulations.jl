using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

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

observables_2_rbc = readdlm(
    joinpath(pkgdir(DifferenceEquations),
        "test/data/RBC_observables.csv"),
    ',')' |> collect

# Data and Noise

@testset "basic inference, simulated noise" begin
    prob = QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, u0_2_rbc,
        (0, size(observables_2_rbc, 2)); C_0 = C_0_rbc,
        C_1 = C_1_rbc,
        C_2 = C_2_rbc, observables_noise = D_2_rbc,
        observables = observables_2_rbc)
    @inferred QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, u0_2_rbc,
        (0, size(observables_2_rbc, 2)); C_0 = C_0_rbc,
        C_1 = C_1_rbc, C_2 = C_2_rbc,
        observables_noise = D_2_rbc,
        observables = observables_2_rbc)

    sol = solve(prob)
    @inferred solve(prob)
end

@testset "basic inference, simulated noise, no observations and no observation noise" begin
    T = 20
    prob = QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, u0_2_rbc, (0, T);
        C_0 = C_0_rbc, C_1 = C_1_rbc, C_2 = C_2_rbc)

    sol = solve(prob)
    @inferred solve(prob)

    # todo: add in regression tests
end

@testset "basic inference, no simulated noise, no observations with observation noise" begin
    T = 20
    B_no_noise = zeros(2, 2)
    u0 = [1.0, 0.5]
    B_no_noise = reshape([0.0; 0.0], 2, 1)

    prob_no_noise = QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc, B_no_noise, u0,
        (0, T);
        C_0 = C_0_rbc, C_1 = C_1_rbc, C_2 = C_2_rbc)

    sol_no_noise = solve(prob_no_noise)

    prob_obs_noise = QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc, B_no_noise, u0,
        (0, T);
        C_0 = C_0_rbc, C_1 = C_1_rbc, C_2 = C_2_rbc,
        observables_noise = D_2_rbc)
    @inferred QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc, B_no_noise, u0, (0, T);
        C_0 = C_0_rbc, C_1 = C_1_rbc, C_2 = C_2_rbc,
        observables_noise = D_2_rbc)
    sol_obs_noise = solve(prob_obs_noise)
    @inferred solve(prob_obs_noise)

    # check that if the variance of the noise is tiny it is identical
    sol_tiny_obs_noise = solve(QuadraticStateSpaceProblem(A_0_rbc, A_1_rbc, A_2_rbc,
        B_no_noise, u0,
        (0, T);
        C_0 = C_0_rbc, C_1 = C_1_rbc,
        C_2 = C_2_rbc,
        observables_noise = [
            1e-16,
            1e-16
        ]))
    @test maximum(maximum.(sol_tiny_obs_noise.z - sol_no_noise.z)) < 1e-7  # still some noise 
    @test maximum(maximum.(sol_tiny_obs_noise.z - sol_no_noise.z)) > 0.0  # but not zero
end
