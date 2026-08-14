using DifferenceEquations, Test

@testset "solve applies u0 override" begin
    prob = LinearStateSpaceProblem([2.0;;], nothing, [1.0], (0, 2))

    sol = solve(prob, DirectIteration(); u0 = [3.0])

    @test sol.u == [[3.0], [6.0], [12.0]]
end
