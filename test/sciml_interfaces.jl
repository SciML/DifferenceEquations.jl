using DifferenceEquations, Distributions, LinearAlgebra, Test
using DelimitedFiles, DiffEqBase, Plots, DataFrames

# --- RBC model data (shared by both problem types) ---

A_rbc = [
    0.9568351489231076 6.209371005755285;
    3.0153731819288737e-18 0.20000000000000007
]
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = abs2.([0.1, 0.1])
u0_rbc = zeros(2)

observables_rbc_matrix = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ','
)' |> collect
noise_rbc_matrix = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ','
)' |> collect
T = 5
observables_rbc = [observables_rbc_matrix[:, t] for t in 1:T]
noise_rbc = [noise_rbc_matrix[:, t] for t in 1:T]

# --- LinearStateSpaceProblem SciML interfaces ---

@testset "Plotting given noise (Linear)" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc));
        C = C_rbc,
        observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc, syms = (:a, :b)
    )
    sol = solve(prob)
    plot(sol)
end

@testset "Ensemble simulation and plotting given noise (Linear)" begin
    # random initial conditions via the u0
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc,
        MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))),
        (0, length(observables_rbc)); C = C_rbc,
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

@testset "Dataframes (Linear)" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc,
        MvNormal(u0_rbc, diagm(ones(length(u0_rbc)))),
        (0, length(observables_rbc)); C = C_rbc,
        observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc, syms = (:a, :b)
    )
    sol = solve(prob)
    df = DataFrame(sol)
    @test propertynames(df) == [:timestamp, :a, :b]
    @test size(df) == (6, 3)
end

@testset "Symbolic indexing — state and obs (Linear)" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc));
        C = C_rbc,
        observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc,
        syms = (:capital, :productivity),
        obs_syms = (:output, :consumption)
    )
    sol = solve(prob)

    # State indexing
    @test sol[:capital] ≈ [sol.u[t][1] for t in eachindex(sol.u)]
    @test sol[:productivity] ≈ [sol.u[t][2] for t in eachindex(sol.u)]

    # Observation indexing
    @test sol[:output] ≈ [sol.z[t][1] for t in eachindex(sol.z)]
    @test sol[:consumption] ≈ [sol.z[t][2] for t in eachindex(sol.z)]

    # Unknown symbol errors
    @test_throws Exception sol[:nonexistent]

    # Direct u access works
    @test length(sol.u) == length(observables_rbc) + 1

    # DataFrame still works
    df = DataFrame(sol)
    @test :capital in propertynames(df)
end

@testset "No syms — backward compat (Linear)" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc))
    )
    sol = solve(prob)
    @test length(sol.u) == length(observables_rbc) + 1
end

@testset "Plotting simulating noise (Linear)" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc));
        C = C_rbc,
        observables_noise = D_rbc, observables = observables_rbc,
        syms = (:a, :b)
    )
    sol = solve(prob)
    plot(sol)
end

@testset "Ensemble simulation and plotting, simulating noise (Linear)" begin
    # fixed initial condition, random noise
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc));
        C = C_rbc,
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

# --- StateSpaceProblem callbacks + data ---

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

# --- StateSpaceProblem SciML interfaces ---

@testset "remake with u0 and p (Generic)" begin
    prob = StateSpaceProblem(
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

@testset "Plotting given noise (Generic)" begin
    prob = StateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, noise = noise_rbc,
        observables = observables_rbc, syms = (:a, :b)
    )
    sol = solve(prob)
    plot(sol)
end

@testset "Ensemble simulation and plotting given noise (Generic)" begin
    prob = StateSpaceProblem(
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

@testset "Dataframes (Generic)" begin
    prob = StateSpaceProblem(
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

@testset "Symbolic indexing — state and obs (Generic)" begin
    prob = StateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        syms = (:capital, :productivity),
        obs_syms = (:output, :consumption),
        observables_noise = D_rbc, noise = noise_rbc, observables = observables_rbc
    )
    sol = solve(prob)

    # State indexing
    @test sol[:capital] ≈ [sol.u[t][1] for t in eachindex(sol.u)]
    @test sol[:productivity] ≈ [sol.u[t][2] for t in eachindex(sol.u)]

    # Observation indexing
    @test sol[:output] ≈ [sol.z[t][1] for t in eachindex(sol.z)]
    @test sol[:consumption] ≈ [sol.z[t][2] for t in eachindex(sol.z)]

    # Unknown symbol errors
    @test_throws Exception sol[:nonexistent]

    # Direct u access works
    @test length(sol.u) == T + 1

    # DataFrame still works
    df = DataFrame(sol)
    @test :capital in propertynames(df)
end

@testset "Symbolic indexing — syms only, no obs_syms (Generic)" begin
    prob = StateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        syms = (:capital, :productivity),
        observables_noise = D_rbc, noise = noise_rbc, observables = observables_rbc
    )
    sol = solve(prob)
    @test sol[:capital] ≈ [sol.u[t][1] for t in eachindex(sol.u)]
    @test_throws ArgumentError sol[:output]  # no obs_syms defined
end

@testset "Symbolic indexing — obs_syms only, no syms (Generic)" begin
    prob = StateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        obs_syms = (:output, :consumption),
        observables_noise = D_rbc, noise = noise_rbc, observables = observables_rbc
    )
    sol = solve(prob)
    @test sol[:output] ≈ [sol.z[t][1] for t in eachindex(sol.z)]
    @test_throws ArgumentError sol[:capital]  # no syms defined
end

@testset "Symbolic indexing — obs_syms but no observations in solution (Generic)" begin
    prob = StateSpaceProblem(
        linear_f!!, nothing, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 0,
        obs_syms = (:output, :consumption),
        noise = noise_rbc
    )
    sol = solve(prob)
    @test sol.z === nothing
    @test_throws Exception sol[:output]  # obs_syms defined but z is nothing
end

@testset "Symbolic indexing survives remake (Generic)" begin
    prob = StateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        syms = (:capital, :productivity),
        obs_syms = (:output, :consumption),
        observables_noise = D_rbc, noise = noise_rbc, observables = observables_rbc
    )
    prob2 = remake(prob; u0 = [0.1, 0.2])
    sol2 = solve(prob2)
    @test sol2[:capital] ≈ [sol2.u[t][1] for t in eachindex(sol2.u)]
    @test sol2[:output] ≈ [sol2.z[t][1] for t in eachindex(sol2.z)]
end

@testset "No syms — backward compat (Generic)" begin
    prob = StateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2
    )
    sol = solve(prob)
    @test length(sol.u) == T + 1
end

@testset "Plotting simulating noise (Generic)" begin
    prob = StateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p_rbc;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, observables = observables_rbc,
        syms = (:a, :b)
    )
    sol = solve(prob)
    plot(sol)
end

@testset "Ensemble simulation and plotting, simulating noise (Generic)" begin
    prob = StateSpaceProblem(
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
