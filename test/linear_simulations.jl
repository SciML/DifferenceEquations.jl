
using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

# Matrices from RBC
A_rbc = [0.9568351489231076 6.209371005755285;
         3.0153731819288737e-18 0.20000000000000007]
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = abs2.([0.1, 0.1])
u0_rbc = zeros(2)

observables_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"),
                          ',')' |> collect
# Data and Noise
@testset "basic inference, simulated noise" begin
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, observables = observables_rbc,
                                   syms = [:a, :b])
    @inferred LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2));
                                      C = C_rbc, observables_noise = D_rbc,
                                      observables = observables_rbc, syms = [:a, :b])

    sol = solve(prob)
    @inferred solve(prob)

    # todo: add in regression tests
end

@testset "basic inference, simulated noise, no observations, no observation noise" begin
    T = 20
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, T); C = C_rbc, syms = [:a, :b])
    @inferred LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, T); C = C_rbc, syms = [:a, :b])

    sol = solve(prob)
    @inferred solve(prob)

    # todo: add in regression tests
end

@testset "basic inference, no simulated noise, no observations with observation noise" begin
    T = 20
    B_no_noise = zeros(2, 2)
    u0 = [1.0, 0.5]
    prob_no_noise = LinearStateSpaceProblem(A_rbc, B_no_noise, u0, (0, T); C = C_rbc,
                                            syms = [:a, :b])

    sol_no_noise = solve(prob_no_noise)

    prob_obs_noise = LinearStateSpaceProblem(A_rbc, B_no_noise, u0, (0, T); C = C_rbc,
                                             syms = [:a, :b], observables_noise = D_rbc)
    @inferred LinearStateSpaceProblem(A_rbc, B_no_noise, u0, (0, T); C = C_rbc,
                                      syms = [:a, :b], observables_noise = D_rbc)
    sol_obs_noise = solve(prob_obs_noise)
    @inferred solve(prob_obs_noise)

    # check that if the variance of the noise is tiny it is identical
    sol_tiny_obs_noise = solve(LinearStateSpaceProblem(A_rbc, B_no_noise, u0, (0, T); C = C_rbc,
                                                       syms = [:a, :b],
                                                       observables_noise = [1e-16, 1e-16]))
    @test maximum(maximum.(sol_tiny_obs_noise.z - sol_no_noise.z)) < 1e-7  # still some noise 
    @test maximum(maximum.(sol_tiny_obs_noise.z - sol_no_noise.z)) > 0.0  # but not zero
end
