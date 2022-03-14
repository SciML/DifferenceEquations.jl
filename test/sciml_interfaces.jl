using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient
using Plots, DataFrames

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

@testset "Plotting" begin
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, noise = noise_rbc,
                                   observables = observables_rbc, syms = (:a, :b))
    sol = solve(prob)
    plot(sol)
end

@testset "Ensemble simulation and plotting" begin
    # random initial conditions via the u0
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))),
                                   (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, noise = noise_rbc,
                                   observables = observables_rbc, syms = (:a, :b))
    sol2 = solve(EnsembleProblem(prob), DirectIteration(), EnsembleThreads(); trajectories = 10)
    plot(sol2)
    summ = EnsembleSummary(sol2)
    plot(summ)
end

@testset "Dataframes" begin
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))),
                                   (0, size(observables_rbc, 2)); C = C_rbc,
                                   observables_noise = D_rbc, noise = noise_rbc,
                                   observables = observables_rbc, syms = (:a, :b))
    sol = solve(prob)
    df = DataFrame(sol)
    @test propertynames(df) == [:timestamp, :a, :b]
    @test size(df) == (6, 3)
end