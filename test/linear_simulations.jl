
using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

# Matrices from RBC
A_rbc = [0.9568351489231076 6.209371005755285;
         3.0153731819288737e-18 0.20000000000000007]
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = MvNormal(Diagonal(abs2.([0.1, 0.1])))
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

@testset "basic inference, simulated noise, no observations" begin
    T = 20
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, T); C = C_rbc, syms = [:a, :b])
    @inferred LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, T); C = C_rbc, syms = [:a, :b])

    sol = solve(prob)
    @inferred solve(prob)

    # todo: add in regression tests
end
