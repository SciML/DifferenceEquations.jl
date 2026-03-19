using DifferenceEquations, Distributions, LinearAlgebra, Test
using DelimitedFiles, DiffEqBase, Plots, DataFrames

# RBC model matrices
A_rbc = [
    0.9568351489231076 6.209371005755285;
    3.0153731819288737e-18 0.20000000000000007
]
B_rbc = reshape([0.0; -0.01], 2, 1)
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = abs2.([0.1, 0.1])
u0_rbc = zeros(2)

observables_rbc = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ','
)' |> collect
noise_rbc = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ','
)' |> collect
T = 5
observables_rbc = observables_rbc[:, 1:T]
noise_rbc = noise_rbc[:, 1:T]

# Callbacks
linear_f!! = (x_next, x, w, p, t) -> begin
    mul!(x_next, p.A, x)
    mul!(x_next, p.B, w, 1.0, 1.0)
    return x_next
end
linear_g!! = (y, x, p, t) -> begin
    mul!(y, p.C, x)
    return y
end
p_rbc = (; A = A_rbc, B = B_rbc, C = C_rbc)

@testset "remake with u0 and p" begin
    prob = GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, noise = noise_rbc, observables = observables_rbc
    )

    # remake with new u0
    new_u0 = [0.1, 0.2]
    prob2 = remake(prob; u0 = new_u0)
    @test prob2.u0 == new_u0
    @test prob2.p === p_rbc
    sol2 = solve(prob2)
    @test length(sol2.u) == T + 1

    # remake with new p
    new_p = (; A = A_rbc * 0.99, B = B_rbc, C = C_rbc)
    prob3 = remake(prob; p = new_p)
    @test prob3.p === new_p
    @test prob3.u0 == u0_rbc
    sol3 = solve(prob3)
    @test length(sol3.u) == T + 1
end

@testset "Plotting given noise" begin
    prob = GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc, syms = (:a, :b)
    )
    sol = solve(prob)
    plot(sol)
end

@testset "Ensemble simulation and plotting given noise" begin
    prob = GenericStateSpaceProblem(
        linear_f!!, linear_g!!,
        MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))),
        (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc, syms = (:a, :b)
    )
    sol2 = solve(
        EnsembleProblem(prob), DirectIteration(), EnsembleThreads();
        trajectories = 10
    )
    plot(sol2)
    summ = EnsembleSummary(sol2)
    plot(summ)
end

@testset "Dataframes" begin
    prob = GenericStateSpaceProblem(
        linear_f!!, linear_g!!,
        MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))),
        (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc, syms = (:a, :b)
    )
    sol = solve(prob)
    df = DataFrame(sol)
    @test propertynames(df) == [:timestamp, :a, :b]
    @test size(df) == (T + 1, 3)
end

@testset "Plotting simulating noise" begin
    prob = GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, observables = observables_rbc,
        syms = (:a, :b)
    )
    sol = solve(prob)
    plot(sol)
end

@testset "Ensemble simulation and plotting, simulating noise" begin
    prob = GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, observables = observables_rbc,
        syms = (:a, :b)
    )
    sol2 = solve(
        EnsembleProblem(prob), DirectIteration(), EnsembleThreads();
        trajectories = 10
    )
    plot(sol2)
    summ = EnsembleSummary(sol2)
    plot(summ)
end
