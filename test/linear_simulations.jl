
using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote, Random
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

@testset "simulation with observations and noise, no observation noise" begin
    Random.seed!(1234)
    sol = solve(LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, 5); C = C_rbc))
    @test sol.u ≈
          [[0.0, 0.0], [0.0, 0.003597289068234817], [0.02233690243961772, -0.010152627110638895],
           [-0.04166869504075366, 0.0021653707472607075],
           [-0.026424481689999797, -0.006756025225207251],
           [-0.06723454002062011, -0.00555367682297924]]
    @test sol.z ≈
          [[0.0, 0.0], [0.0024270440446074832, 0.0], [-0.004710049663169753, 0.02233690243961772],
           [-0.002530764810543453, -0.04166869504075366],
           [-0.007089573167553201, -0.026424481689999797],
           [-0.010187822270025022, -0.06723454002062011]]
    @test sol.W ≈
          [-0.3597289068234817 1.0872084924285859 -0.4195896169388487 0.7189099374659392 0.4202471777937789]
    @test sol.logpdf === nothing
end

@testset "simulation with observations and noise, no observation noise" begin
    Random.seed!(1234)
    sol = solve(LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, 5); C = C_rbc,
                                        observables_noise = D_rbc))
    @test sol.u ≈
          [[0.0, 0.0], [0.0, 0.003597289068234817], [0.02233690243961772, -0.010152627110638895],
           [-0.04166869504075366, 0.0021653707472607075],
           [-0.026424481689999797, -0.006756025225207251],
           [-0.06723454002062011, -0.00555367682297924]]
    @test sol.z ≈
          [[-0.06856709022761191, 0.20547630560640365],
           [0.034916316989299055, -0.030490125519643224],
           [0.0414594477647271, -0.06215886919798015], [0.08614040809827415, -0.040311314885592704],
           [0.0034755874208198837, -0.08053882074804589],
           [-0.07921183287013331, -0.16087605412196193]]
    @test sol.W ≈
          [-0.3597289068234817 1.0872084924285859 -0.4195896169388487 0.7189099374659392 0.4202471777937789]
    @test sol.logpdf === nothing
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

@testset "basic inference, no noise, no observations and no with observation noise" begin
    T = 5
    B_no_noise = zeros(2, 2)
    u0 = [1.0, 0.5]
    sol_no_noise = solve(LinearStateSpaceProblem(A_rbc, B_no_noise, u0, (0, T); C = C_rbc,
                                                 syms = [:a, :b]))

    #Now literally pass in no noise in B with a nothing
    prob = LinearStateSpaceProblem(A_rbc, nothing, u0, (0, T); C = C_rbc,
                                   syms = [:a, :b])
    @inferred LinearStateSpaceProblem(A_rbc, nothing, u0, (0, T); C = C_rbc,
                                      syms = [:a, :b])

    sol_nothing_noise = solve(prob)
    @inferred solve(prob)

    @test sol_no_noise.z ≈ sol_nothing_noise.z
    @test sol_no_noise.u ≈ sol_nothing_noise.u
    @test sol_nothing_noise.W === nothing
end
