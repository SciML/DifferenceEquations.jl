using DifferenceEquations, Test

@testset "Generic state-space interface" begin
    transition = function (x_next, x, w, p, t)
        x_next[1] = x[1] + w[1] + t
        return x_next
    end
    observation = (y, x, p, t) -> begin
        y[1] = 2 * x[1] + t
        return y
    end
    problem = StateSpaceProblem(
        transition, observation, [1.0], (0, 2);
        n_shocks = 1,
        n_obs = 1,
        noise = [[2.0], [3.0]],
    )

    function generic_solve(prob::AbstractStateSpaceProblem)
        workspace = init(prob, DirectIteration())
        return solve!(workspace)
    end

    solution = generic_solve(problem)
    @test solution isa StateSpaceSolution
    @test solution.u == [[1.0], [3.0], [7.0]]
    @test solution.z == [[2.0], [7.0], [16.0]]
    @test solution.t == [0, 1, 2]

    endpoint_workspace = init(problem, DirectIteration(); save_everystep = false)
    endpoint_solution = solve!(endpoint_workspace)
    @test endpoint_solution.u == [[1.0], [7.0]]
    @test endpoint_solution.z == [[2.0], [16.0]]
    @test endpoint_solution.t == [0, 2]
end
