using DifferenceEquations, LinearAlgebra, Test
using StaticArrays

@testset "StaticArrays linear DirectIteration" begin
    A = @SMatrix [0.9 0.1; 0.0 0.8]
    B = @SMatrix [0.0; 0.1;;]  # 2×1 SMatrix
    C = @SMatrix [1.0 0.0; 0.0 1.0]
    u0 = @SVector [0.5, 0.3]

    # Create noise as vector of SVector
    noise = [SVector{1, Float64}(randn()) for _ in 1:9]

    prob = LinearStateSpaceProblem(A, B, u0, (0, 9); C = C, noise = noise)

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

    prob = LinearStateSpaceProblem(A, nothing, u0, (0, 5); C = C)

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

    prob = LinearStateSpaceProblem(A, B, u0, (0, 4); C = nothing, noise = noise)
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
