# tests the 
using RecursiveArrayTools, DifferenceEquations, Test
@testset "matrix as AbstractVectorOfArray" begin
    A = rand(2, 10)
    vv = VectorOfArray([A[:, i] for i in 1:size(A, 2)])
    mv = MatrixVectorOfArray(A)
    @test mv[1] == A[:, 1]
    @test mv[1] == vv[1]
    @test size(mv) == (2, 10)
    @test mv[5][2] == A[2, 5]
    @test mv[5][2] == vv[5][2]
    @test mean(mv) == mean(vv)
    @test map(x -> 2 * x, mv) == map(x -> 2 * x, eachcol(mv.u))
    @test mapreduce(x -> 2 * x, +, mv) == mapreduce(x -> 2 * x, +, vv)
    @test [v for v in mv] == [v for v in vv]  # interator
    mv[5] = [1.0, 0.1]
    vv[5] = [1.0, 0.1]
    @test mv == vv
    mv[2, 6] = 5.9
    vv[2, 6] = 5.9
    @test mv == vv

    mv_sim = similar(mv)
    fill!(mv_sim, 0.0)

    @inferred MatrixVectorOfArray(A)
    @inferred mv[1]
    @inferred mv[2, 2]
    @inferred similar(mv)
end