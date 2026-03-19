using DifferenceEquations, Distributions, LinearAlgebra, Test
using DelimitedFiles, DiffEqBase, Random
using DifferenceEquations: init, solve!

A_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
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

@testset "init/solve! matches solve for DirectIteration" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2));
        C = C_rbc, observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc
    )
    sol_direct = solve(prob)

    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)

    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.logpdf ≈ sol_direct.logpdf
end

@testset "init/solve! matches solve for KalmanFilter" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2));
        C = C_rbc, observables_noise = D_rbc, observables = observables_rbc,
        u0_prior_mean = u0_rbc,
        u0_prior_var = diagm(ones(length(u0_rbc)))
    )
    sol_direct = solve(prob)

    ws = init(prob, KalmanFilter())
    sol_ws = solve!(ws)

    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.logpdf ≈ sol_direct.logpdf
    @test sol_ws.P ≈ sol_direct.P
end

@testset "repeated solve! gives consistent results" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2));
        C = C_rbc, observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc
    )

    ws = init(prob, DirectIteration())
    sol1 = solve!(ws)
    sol2 = solve!(ws)

    @test sol1.u ≈ sol2.u
    @test sol1.z ≈ sol2.z
    @test sol1.logpdf ≈ sol2.logpdf
end

@testset "repeated solve! for KalmanFilter" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, size(observables_rbc, 2));
        C = C_rbc, observables_noise = D_rbc, observables = observables_rbc,
        u0_prior_mean = u0_rbc,
        u0_prior_var = diagm(ones(length(u0_rbc)))
    )

    ws = init(prob, KalmanFilter())
    sol1 = solve!(ws)
    sol2 = solve!(ws)

    @test sol1.logpdf ≈ sol2.logpdf
    @test sol1.u ≈ sol2.u
    @test sol1.P ≈ sol2.P
end

