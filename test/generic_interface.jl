using DifferenceEquations, Test

@testset "Generic state-space interface" begin
    transition = (x_next, x, w, p, t) -> copyto!(x_next, x)
    problem = StateSpaceProblem(
        transition, nothing, [1.0], (0, 2);
        n_shocks = 0
    )

    function generic_solve(prob::AbstractStateSpaceProblem)
        workspace = init(prob, DirectIteration(); save_everystep = false)
        return solve!(workspace)
    end

    solution = generic_solve(problem)
    @test solution isa StateSpaceSolution
    @test solution.u == [[1.0], [1.0]]
    @test solution.t == [0, 2]
end
