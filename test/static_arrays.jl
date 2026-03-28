using DifferenceEquations, LinearAlgebra, Random, Test
using StaticArrays
using DifferenceEquations: mul!!, muladd!!

# --- LinearStateSpaceProblem ---

@testset "StaticArrays linear DirectIteration" begin
    A = @SMatrix [0.9 0.1; 0.0 0.8]
    B = @SMatrix [0.0; 0.1;;]  # 2×1 SMatrix
    C = @SMatrix [1.0 0.0; 0.0 1.0]
    u0 = @SVector [0.5, 0.3]

    # Create noise as vector of SVector
    noise = [SVector{1, Float64}(randn()) for _ in 1:9]

    prob = LinearStateSpaceProblem(A, B, u0, (0, 9); C, noise)

    # Compare SVector result to Vector result
    A_v = Matrix(A)
    B_v = Matrix(B)
    C_v = Matrix(C)
    u0_v = Vector(u0)
    noise_v = [Vector(n) for n in noise]

    prob_v = LinearStateSpaceProblem(A_v, B_v, u0_v, (0, 9); C = C_v, noise = noise_v)

    sol_s = solve(prob)
    sol_v = solve(prob_v)

    # Results should match
    for t in eachindex(sol_s.u)
        @test Vector(sol_s.u[t]) ≈ sol_v.u[t]
    end
    for t in eachindex(sol_s.z)
        @test Vector(sol_s.z[t]) ≈ sol_v.z[t]
    end
end

@testset "StaticArrays linear no noise" begin
    A = @SMatrix [0.9 0.1; 0.0 0.8]
    C = @SMatrix [1.0 0.0; 0.0 1.0]
    u0 = @SVector [1.0, 0.5]

    prob = LinearStateSpaceProblem(A, nothing, u0, (0, 5); C)

    A_v = Matrix(A)
    C_v = Matrix(C)
    u0_v = Vector(u0)
    prob_v = LinearStateSpaceProblem(A_v, nothing, u0_v, (0, 5); C = C_v)

    sol_s = solve(prob)
    sol_v = solve(prob_v)

    for t in eachindex(sol_s.u)
        @test Vector(sol_s.u[t]) ≈ sol_v.u[t]
    end
end

@testset "StaticArrays no observation" begin
    A = @SMatrix [0.9 0.1; 0.0 0.8]
    B = @SMatrix [0.0; 0.1;;]
    u0 = @SVector [1.0, 0.5]

    noise = [SVector{1, Float64}(randn()) for _ in 1:4]

    prob = LinearStateSpaceProblem(A, B, u0, (0, 4); C = nothing, noise)
    sol = solve(prob)

    @test sol.z === nothing
    @test length(sol.u) == 5

    # Verify against manual computation
    A_v = Matrix(A)
    B_v = Matrix(B)
    u0_v = Vector(u0)
    noise_v = [Vector(n) for n in noise]
    prob_v = LinearStateSpaceProblem(A_v, B_v, u0_v, (0, 4); C = nothing, noise = noise_v)
    sol_v = solve(prob_v)

    for t in eachindex(sol.u)
        @test Vector(sol.u[t]) ≈ sol_v.u[t]
    end
end

# --- Generic !! callbacks ---

@inline function f_lss!!(x_p, x, w, p, t)
    x_p = mul!!(x_p, p.A, x)
    return muladd!!(x_p, p.B, w)
end

@inline function g_lss!!(y, x, p, t)
    return mul!!(y, p.C, x)
end

@testset "Generic !! callbacks — mutable vs static consistency" begin
    A_m = [0.9 0.1; 0.0 0.8]
    B_m = reshape([0.0; 0.1], 2, 1)
    C_m = [1.0 0.0; 0.0 1.0]
    u0_m = [0.5, 0.3]
    noise_vals = [randn(1) for _ in 1:9]

    # Mutable version
    p_m = (; A = A_m, B = B_m, C = C_m)
    prob_m = StateSpaceProblem(
        f_lss!!, g_lss!!, u0_m, (0, 9), p_m;
        n_shocks = 1, n_obs = 2, noise = noise_vals
    )
    sol_m = solve(prob_m)

    # Static version — same callbacks, same data, just wrapped in SMatrix/SVector
    A_s = SMatrix{2, 2}(A_m)
    B_s = SMatrix{2, 1}(B_m)
    C_s = SMatrix{2, 2}(C_m)
    u0_s = SVector{2}(u0_m)
    noise_s = [SVector{1}(n) for n in noise_vals]

    p_s = (; A = A_s, B = B_s, C = C_s)
    prob_s = StateSpaceProblem(
        f_lss!!, g_lss!!, u0_s, (0, 9), p_s;
        n_shocks = 1, n_obs = 2, noise = noise_s
    )
    sol_s = solve(prob_s)

    # Results must match exactly
    for t in eachindex(sol_m.u)
        @test Vector(sol_s.u[t]) ≈ sol_m.u[t]
    end
    for t in eachindex(sol_m.z)
        @test Vector(sol_s.z[t]) ≈ sol_m.z[t]
    end

    # Verify static types are preserved
    @test eltype(sol_s.u) <: SVector{2, Float64}
    @test eltype(sol_s.z) <: SVector{2, Float64}
end

@testset "Generic !! callbacks — static matches LinearStateSpaceProblem" begin
    A = @SMatrix [0.9 0.1; 0.0 0.8]
    B = @SMatrix [0.0; 0.1;;]
    C = @SMatrix [1.0 0.0; 0.0 1.0]
    u0 = @SVector [0.5, 0.3]
    noise = [SVector{1, Float64}(randn()) for _ in 1:9]

    prob_linear = LinearStateSpaceProblem(A, B, u0, (0, 9); C, noise)
    sol_linear = solve(prob_linear)

    p = (; A, B, C)
    prob_generic = StateSpaceProblem(
        f_lss!!, g_lss!!, u0, (0, 9), p;
        n_shocks = 1, n_obs = 2, noise = noise
    )
    sol_generic = solve(prob_generic)

    for t in eachindex(sol_linear.u)
        @test sol_linear.u[t] ≈ sol_generic.u[t]
    end
    for t in eachindex(sol_linear.z)
        @test sol_linear.z[t] ≈ sol_generic.z[t]
    end
end

@testset "Generic !! callbacks — static no noise" begin
    A = @SMatrix [0.9 0.1; 0.0 0.8]
    C = @SMatrix [1.0 0.0; 0.0 1.0]
    u0 = @SVector [1.0, 0.5]

    prob_linear = LinearStateSpaceProblem(A, nothing, u0, (0, 5); C)
    sol_linear = solve(prob_linear)

    # f_lss!! handles w=nothing via muladd!!(x_p, B, nothing) → x_p
    p = (; A, B = nothing, C)
    prob_generic = StateSpaceProblem(
        f_lss!!, g_lss!!, u0, (0, 5), p;
        n_shocks = 0, n_obs = 2
    )
    sol_generic = solve(prob_generic)

    for t in eachindex(sol_linear.u)
        @test sol_linear.u[t] ≈ sol_generic.u[t]
    end
    for t in eachindex(sol_linear.z)
        @test sol_linear.z[t] ≈ sol_generic.z[t]
    end
end

# --- KalmanFilter with StaticArrays ---

@testset "StaticArrays Kalman filter" begin
    Random.seed!(789)
    A_raw = randn(3, 3)
    A_m = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    B_m = 0.1 * randn(3, 2)
    C_m = randn(2, 3)
    R_m = 0.01 * I(2) |> Matrix
    mu0_m = zeros(3)
    Sig0_m = Matrix{Float64}(I, 3, 3)

    # Generate observations
    x0 = randn(3)
    noise = [randn(2) for _ in 1:10]
    sim = solve(LinearStateSpaceProblem(A_m, B_m, x0, (0, 10); C = C_m, noise))
    y_m = [sim.z[t + 1] + 0.1 * randn(2) for t in 1:10]

    prob_m = LinearStateSpaceProblem(
        A_m, B_m, zeros(3), (0, 10); C = C_m,
        u0_prior_mean = mu0_m, u0_prior_var = Sig0_m,
        observables_noise = R_m, observables = y_m
    )
    sol_m = solve(prob_m, KalmanFilter())

    # Static version
    A_s = SMatrix{3, 3}(A_m)
    B_s = SMatrix{3, 2}(B_m)
    C_s = SMatrix{2, 3}(C_m)
    R_s = SMatrix{2, 2}(R_m)
    mu0_s = SVector{3}(mu0_m)
    Sig0_s = SMatrix{3, 3}(Sig0_m)
    y_s = [SVector{2}(y) for y in y_m]

    prob_s = LinearStateSpaceProblem(
        A_s, B_s, SVector{3}(zeros(3)), (0, 10); C = C_s,
        u0_prior_mean = mu0_s, u0_prior_var = Sig0_s,
        observables_noise = R_s, observables = y_s
    )
    sol_s = solve(prob_s, KalmanFilter())

    # logpdf must match
    @test sol_s.logpdf ≈ sol_m.logpdf

    # Filtered states and covariances must match
    for t in eachindex(sol_s.u)
        @test Vector(sol_s.u[t]) ≈ sol_m.u[t]
        @test Matrix(sol_s.P[t]) ≈ sol_m.P[t]
    end
    for t in eachindex(sol_s.z)
        @test Vector(sol_s.z[t]) ≈ sol_m.z[t]
    end

    # Verify static types are preserved
    @test eltype(sol_s.u) <: SVector{3, Float64}
    @test eltype(sol_s.P) <: SMatrix{3, 3, Float64}
    @test eltype(sol_s.z) <: SVector{2, Float64}
end

# --- PrunedQuadraticStateSpaceProblem with StaticArrays ---

@testset "StaticArrays pruned quadratic" begin
    Random.seed!(42)
    A_2 = 0.01 * randn(2, 2, 2)
    C_2 = 0.01 * randn(2, 2, 2)
    noise_vals = [randn(1) for _ in 1:10]

    # Mutable
    A0_m = [0.001, -0.001]
    A1_m = [0.3 0.1; -0.1 0.3]
    B_m = reshape([0.1, 0.0], 2, 1)
    C0_m = [0.001, -0.001]
    C1_m = [1.0 0.0; 0.0 1.0]
    u0_m = zeros(2)

    prob_m = PrunedQuadraticStateSpaceProblem(
        A0_m, A1_m, A_2, B_m, u0_m, (0, 10);
        C_0 = C0_m, C_1 = C1_m, C_2 = C_2, noise = noise_vals
    )
    sol_m = solve(prob_m)

    # Static
    A0_s = @SVector [0.001, -0.001]
    A1_s = @SMatrix [0.3 0.1; -0.1 0.3]
    B_s = @SMatrix [0.1; 0.0;;]
    C0_s = @SVector [0.001, -0.001]
    C1_s = @SMatrix [1.0 0.0; 0.0 1.0]
    u0_s = @SVector zeros(2)
    noise_s = [SVector{1}(n) for n in noise_vals]

    prob_s = PrunedQuadraticStateSpaceProblem(
        A0_s, A1_s, A_2, B_s, u0_s, (0, 10);
        C_0 = C0_s, C_1 = C1_s, C_2 = C_2, noise = noise_s
    )
    sol_s = solve(prob_s)

    for t in eachindex(sol_s.u)
        @test Vector(sol_s.u[t]) ≈ sol_m.u[t]
    end
    for t in eachindex(sol_s.z)
        @test Vector(sol_s.z[t]) ≈ sol_m.z[t]
    end

    @test eltype(sol_s.u) <: SVector{2, Float64}
    @test eltype(sol_s.z) <: SVector{2, Float64}
end

@testset "StaticArrays unpruned quadratic" begin
    Random.seed!(42)
    A_2 = 0.01 * randn(2, 2, 2)
    C_2 = 0.01 * randn(2, 2, 2)
    noise_vals = [randn(1) for _ in 1:10]

    # Mutable
    A0_m = [0.001, -0.001]
    A1_m = [0.3 0.1; -0.1 0.3]
    B_m = reshape([0.1, 0.0], 2, 1)
    C0_m = [0.001, -0.001]
    C1_m = [1.0 0.0; 0.0 1.0]
    u0_m = zeros(2)

    prob_m = QuadraticStateSpaceProblem(
        A0_m, A1_m, A_2, B_m, u0_m, (0, 10);
        C_0 = C0_m, C_1 = C1_m, C_2 = C_2, noise = noise_vals
    )
    sol_m = solve(prob_m)

    # Static
    A0_s = @SVector [0.001, -0.001]
    A1_s = @SMatrix [0.3 0.1; -0.1 0.3]
    B_s = @SMatrix [0.1; 0.0;;]
    C0_s = @SVector [0.001, -0.001]
    C1_s = @SMatrix [1.0 0.0; 0.0 1.0]
    u0_s = @SVector zeros(2)
    noise_s = [SVector{1}(n) for n in noise_vals]

    prob_s = QuadraticStateSpaceProblem(
        A0_s, A1_s, A_2, B_s, u0_s, (0, 10);
        C_0 = C0_s, C_1 = C1_s, C_2 = C_2, noise = noise_s
    )
    sol_s = solve(prob_s)

    for t in eachindex(sol_s.u)
        @test Vector(sol_s.u[t]) ≈ sol_m.u[t]
    end
    for t in eachindex(sol_s.z)
        @test Vector(sol_s.z[t]) ≈ sol_m.z[t]
    end

    @test eltype(sol_s.u) <: SVector{2, Float64}
    @test eltype(sol_s.z) <: SVector{2, Float64}
end

# --- solve!() vs solve() consistency for StaticArrays ---

@testset "StaticArrays solve!() vs solve() consistency" begin
    using DifferenceEquations: init, solve!, StateSpaceWorkspace

    @testset "linear DirectIteration" begin
        A = @SMatrix [0.9 0.1; 0.0 0.8]
        B = @SMatrix [0.0; 0.1;;]
        C = @SMatrix [1.0 0.0; 0.0 1.0]
        u0 = @SVector [0.5, 0.3]
        noise = [SVector{1}(randn()) for _ in 1:9]

        prob = LinearStateSpaceProblem(A, B, u0, (0, 9); C, noise)
        sol_alloc = solve(prob)

        ws = init(prob, DirectIteration())
        sol_inplace = solve!(ws)

        for t in eachindex(sol_alloc.u)
            @test sol_alloc.u[t] ≈ sol_inplace.u[t]
        end
        for t in eachindex(sol_alloc.z)
            @test sol_alloc.z[t] ≈ sol_inplace.z[t]
        end
    end

    @testset "Kalman filter" begin
        Random.seed!(789)
        A_raw = randn(3, 3)
        A = SMatrix{3, 3}(0.5 * A_raw / maximum(abs.(eigvals(A_raw))))
        B = SMatrix{3, 2}(0.1 * randn(3, 2))
        C = SMatrix{2, 3}(randn(2, 3))
        R = SMatrix{2, 2}(0.01 * I(2))
        mu0 = SVector{3}(zeros(3))
        Sig0 = SMatrix{3, 3}(1.0 * I(3))

        noise = [SVector{2}(randn(2)) for _ in 1:10]
        sim = solve(LinearStateSpaceProblem(A, B, mu0, (0, 10); C, noise))
        y = [sim.z[t + 1] + SVector{2}(0.1 * randn(2)) for t in 1:10]

        prob = LinearStateSpaceProblem(
            A, B, SVector{3}(zeros(3)), (0, 10); C,
            u0_prior_mean = mu0, u0_prior_var = Sig0,
            observables_noise = R, observables = y
        )
        sol_alloc = solve(prob, KalmanFilter())

        ws = init(prob, KalmanFilter())
        sol_inplace = solve!(ws)

        @test sol_alloc.logpdf ≈ sol_inplace.logpdf
        for t in eachindex(sol_alloc.u)
            @test sol_alloc.u[t] ≈ sol_inplace.u[t]
            @test sol_alloc.P[t] ≈ sol_inplace.P[t]
        end
    end

    @testset "pruned quadratic" begin
        Random.seed!(42)
        A_2 = 0.01 * randn(2, 2, 2)
        C_2 = 0.01 * randn(2, 2, 2)
        noise = [SVector{1}(randn()) for _ in 1:10]

        prob = PrunedQuadraticStateSpaceProblem(
            @SVector([0.001, -0.001]), @SMatrix([0.3 0.1; -0.1 0.3]),
            A_2, @SMatrix([0.1; 0.0;;]), @SVector(zeros(2)), (0, 10);
            C_0 = @SVector([0.001, -0.001]), C_1 = @SMatrix([1.0 0.0; 0.0 1.0]),
            C_2 = C_2, noise = noise
        )
        sol_alloc = solve(prob)

        ws = init(prob, DirectIteration())
        sol_inplace = solve!(ws)

        for t in eachindex(sol_alloc.u)
            @test sol_alloc.u[t] ≈ sol_inplace.u[t]
        end
        for t in eachindex(sol_alloc.z)
            @test sol_alloc.z[t] ≈ sol_inplace.z[t]
        end
    end
end
