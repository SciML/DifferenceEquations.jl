using DifferenceEquations, LinearAlgebra, Test
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
