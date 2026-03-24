using DifferenceEquations, Distributions, LinearAlgebra, Test, Random
using DelimitedFiles
using DiffEqBase
using DifferenceEquations: init, solve!

# --- RBC Model Data ---

A_rbc = [
    0.9568351489231076 6.209371005755285;
    3.0153731819288737e-18 0.20000000000000007
]
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = abs2.([0.1, 0.1])
u0_rbc = zeros(2)

observables_rbc_matrix = readdlm(
    joinpath(
        pkgdir(DifferenceEquations),
        "test/data/RBC_observables.csv"
    ),
    ','
)' |> collect
observables_rbc = [observables_rbc_matrix[:, t] for t in 1:size(observables_rbc_matrix, 2)]

noise_rbc_matrix = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"),
    ','
)' |> collect

T_rbc = 5
observables_rbc_5 = [observables_rbc_matrix[:, t] for t in 1:T_rbc]
noise_rbc_5 = [noise_rbc_matrix[:, t] for t in 1:T_rbc]

# --- Joint Likelihood Helper ---

function joint_likelihood_1(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(
        A, B, u0, (0, length(observables)); C,
        observables_noise = D,
        noise, observables, kwargs...
    )
    return solve(problem).logpdf
end

# --- Simulation Tests ---

@testset "simulation with noise, observations, and observation noise" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc));
        C = C_rbc,
        observables_noise = D_rbc, observables = observables_rbc,
        syms = [:a, :b]
    )

    sol = solve(prob)
    @inferred solve(prob)
end

@testset "simulation with noise, no observations, no observation noise" begin
    T = 20
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, T); C = C_rbc, syms = [:a, :b])

    sol = solve(prob)
    @inferred solve(prob)
end

@testset "simulation with noise and C, no observation noise" begin
    Random.seed!(1234)
    sol = solve(LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, 5); C = C_rbc))
    @test sol.u ≈
        [
        [0.0, 0.0], [0.0, 0.003597289068234817],
        [0.02233690243961772, -0.010152627110638895],
        [-0.04166869504075366, 0.0021653707472607075],
        [-0.026424481689999797, -0.006756025225207251],
        [-0.06723454002062011, -0.00555367682297924],
    ]
    @test sol.z ≈
        [
        [0.0, 0.0], [0.0024270440446074832, 0.0],
        [-0.004710049663169753, 0.02233690243961772],
        [-0.002530764810543453, -0.04166869504075366],
        [-0.007089573167553201, -0.026424481689999797],
        [-0.010187822270025022, -0.06723454002062011],
    ]
    @test sol.W ≈
        [[-0.3597289068234817], [1.0872084924285859], [-0.4195896169388487], [0.7189099374659392], [0.4202471777937789]]
    @test sol.logpdf == 0.0
end

@testset "simulation with noise, C, and observation noise" begin
    Random.seed!(1234)
    sol = solve(
        LinearStateSpaceProblem(
            A_rbc, B_rbc, u0_rbc, (0, 5); C = C_rbc,
            observables_noise = D_rbc
        )
    )
    @test sol.u ≈
        [
        [0.0, 0.0], [0.0, 0.003597289068234817],
        [0.02233690243961772, -0.010152627110638895],
        [-0.04166869504075366, 0.0021653707472607075],
        [-0.026424481689999797, -0.006756025225207251],
        [-0.06723454002062011, -0.00555367682297924],
    ]
    @test sol.z ≈
        [
        [-0.06856709022761191, 0.20547630560640365],
        [0.034916316989299055, -0.030490125519643224],
        [0.0414594477647271, -0.06215886919798015],
        [0.08614040809827415, -0.040311314885592704],
        [0.0034755874208198837, -0.08053882074804589],
        [-0.07921183287013331, -0.16087605412196193],
    ]
    @test sol.W ≈
        [[-0.3597289068234817], [1.0872084924285859], [-0.4195896169388487], [0.7189099374659392], [0.4202471777937789]]
    @test sol.logpdf == 0.0
end

@testset "no noise (B=zeros) vs observation noise" begin
    T = 20
    B_no_noise = zeros(2, 2)
    u0 = [1.0, 0.5]
    prob_no_noise = LinearStateSpaceProblem(
        A_rbc, B_no_noise, u0, (0, T); C = C_rbc,
        syms = [:a, :b]
    )

    sol_no_noise = solve(prob_no_noise)

    prob_obs_noise = LinearStateSpaceProblem(
        A_rbc, B_no_noise, u0, (0, T); C = C_rbc,
        syms = [:a, :b], observables_noise = D_rbc
    )
    sol_obs_noise = solve(prob_obs_noise)
    @inferred solve(prob_obs_noise)

    sol_tiny_obs_noise = solve(
        LinearStateSpaceProblem(
            A_rbc, B_no_noise, u0, (0, T);
            C = C_rbc,
            syms = [:a, :b],
            observables_noise = [1.0e-16, 1.0e-16]
        )
    )
    @test maximum(maximum.(sol_tiny_obs_noise.z - sol_no_noise.z)) < 1.0e-7
    @test maximum(maximum.(sol_tiny_obs_noise.z - sol_no_noise.z)) > 0.0
end

@testset "B=nothing matches B=zeros" begin
    T = 5
    B_no_noise = zeros(2, 2)
    u0 = [1.0, 0.5]
    sol_no_noise = solve(
        LinearStateSpaceProblem(
            A_rbc, B_no_noise, u0, (0, T); C = C_rbc,
            syms = [:a, :b]
        )
    )

    prob = LinearStateSpaceProblem(
        A_rbc, nothing, u0, (0, T); C = C_rbc,
        syms = [:a, :b]
    )

    sol_nothing_noise = solve(prob)
    @inferred solve(prob)

    @test sol_no_noise.z ≈ sol_nothing_noise.z
    @test sol_no_noise.u ≈ sol_nothing_noise.u
    @test sol_nothing_noise.W === nothing
end

@testset "C=nothing, no observation process" begin
    Random.seed!(1234)
    T = 5
    u0 = [1.0, 0.5]
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0, (0, T); C = nothing,
        syms = [:a, :b]
    )
    sol = solve(prob)
    @inferred solve(prob)

    @test sol.z === nothing
    @test sol.u ≈ [
        [1.0, 0.5], [4.06152065180075, 0.10359728906823484],
        [4.5294797207351944, 0.009847372889361128],
        [4.395111394835915, 0.006165370747260727],
        [4.243680140369242, -0.005956025225207233],
        [4.023519148749289, -0.005393676822979223],
    ]
    @test sol.W ≈
        [[-0.3597289068234817], [1.0872084924285859], [-0.4195896169388487], [0.7189099374659392], [0.4202471777937789]]
    @test sol.logpdf == 0.0
end

# --- Joint Likelihood Tests ---

@testset "joint likelihood inference" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc_5));
        C = C_rbc,
        observables_noise = D_rbc, noise = noise_rbc_5,
        observables = observables_rbc_5
    )
    @inferred LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc_5));
        C = C_rbc, observables_noise = D_rbc,
        noise = noise_rbc_5,
        observables = observables_rbc_5
    )

    sol = solve(prob)
    @inferred solve(prob)

    DiffEqBase.get_concrete_problem(prob, false)
    @inferred DiffEqBase.get_concrete_problem(prob, false)

    joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc_5, observables_rbc_5, D_rbc)
    @inferred joint_likelihood_1(
        A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc_5, observables_rbc_5,
        D_rbc
    )
end

@testset "linear RBC joint likelihood value" begin
    @test joint_likelihood_1(
        A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc_5, observables_rbc_5,
        D_rbc
    ) ≈
        -690.9407412360038
    @inferred joint_likelihood_1(
        A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc_5, observables_rbc_5,
        D_rbc
    )
end

# --- FVGQ Data ---

A_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A.csv"), ',')
B_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
C_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C.csv"), ',')
D_FVGQ = ones(6) * 1.0e-3

observables_FVGQ_matrix = readdlm(
    joinpath(
        pkgdir(DifferenceEquations),
        "test/data/FVGQ20_observables.csv"
    ), ','
)' |> collect
observables_FVGQ = [observables_FVGQ_matrix[:, t] for t in 1:size(observables_FVGQ_matrix, 2)]

noise_FVGQ_matrix = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"),
    ','
)' |> collect
noise_FVGQ = [noise_FVGQ_matrix[:, t] for t in 1:size(noise_FVGQ_matrix, 2)]
u0_FVGQ = zeros(size(A_FVGQ, 1))

@testset "linear FVGQ joint likelihood" begin
    @test joint_likelihood_1(
        A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ, observables_FVGQ,
        D_FVGQ
    ) ≈ -1.4613614369686982e6
    @inferred joint_likelihood_1(
        A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ,
        observables_FVGQ,
        D_FVGQ
    )
end

# --- Primal Edge-Case Checks ---

@testset "z_sum primal" begin
    function z_sum(A, B, C, u0, noise, observables, D; kwargs...)
        problem = LinearStateSpaceProblem(
            A, B, u0, (0, length(observables)); C,
            observables_noise = D,
            noise, observables, kwargs...
        )
        sol = solve(problem)
        return sol.z[5][1] + sol.z[3][2]
    end
    @test z_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc_5, observables_rbc_5, D_rbc) ≈
        -0.09008162336682057
end

@testset "u_sum primal" begin
    function u_sum(A, B, C, u0, noise, observables, D; kwargs...)
        problem = LinearStateSpaceProblem(
            A, B, u0, (0, length(observables)); C,
            observables_noise = D,
            noise, observables, kwargs...
        )
        sol = solve(problem)
        return sol.u[3][1] + sol.u[3][2]
    end
    @test u_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc_5, observables_rbc_5, D_rbc) ≈
        -0.08780558376240931
end

@testset "no_observables_sum primal" begin
    function no_observables_sum(A, B, C, u0, noise; kwargs...)
        problem = LinearStateSpaceProblem(
            A, B, u0, (0, length(noise)); C, noise,
            kwargs...
        )
        sol = solve(problem)
        return sol.W[2][1] + sol.W[4][1] + sol.z[2][2]
    end
    @test no_observables_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc_5) ≈
        -0.08892781958364693
end

@testset "no_noise primal (B=nothing, C present)" begin
    function no_noise(A, C, u0; kwargs...)
        problem = LinearStateSpaceProblem(A, nothing, u0, (0, 5); C, kwargs...)
        sol = solve(problem)
        return sol.z[2][2]
    end
    u_nonzero = [1.1, 0.2]
    @test no_noise(A_rbc, C_rbc, u_nonzero) ≈ 2.2943928649664755
end

@testset "no_observation_equation primal (B=nothing, C=nothing)" begin
    function no_observation_equation(A, u0; kwargs...)
        problem = LinearStateSpaceProblem(A, nothing, u0, (0, 5); kwargs...)
        sol = solve(problem)
        return sol.u[2][2] + sol.u[4][1]
    end
    u_nonzero = [1.1, 0.2]
    @test no_observation_equation(A_rbc, u_nonzero) ≈ 2.4279222804056597
end

@testset "no_observation_equation_noise primal (B present, C=nothing)" begin
    function no_observation_equation_noise(A, B, u0; kwargs...)
        Random.seed!(1234)
        problem = LinearStateSpaceProblem(A, B, u0, (0, 5); kwargs...)
        sol = solve(problem)
        return sol.u[2][2] + sol.u[4][1]
    end
    u_nonzero = [1.1, 0.2]
    @test no_observation_equation_noise(A_rbc, B_rbc, u_nonzero) ≈ 2.3898508744331406
end

@testset "last_state with impulse noise" begin
    function last_state_pass_noise(A, B, C, u0, noise)
        problem = LinearStateSpaceProblem(
            A, B, u0, (0, length(noise)); C, noise,
            observables_noise = nothing, observables = nothing
        )
        sol = solve(problem)
        return sol.u[end][2]
    end
    T_imp = 20
    impulse_noise = [[i == 1 ? 1.0 : 0.0] for i in 1:T_imp]
    u_nonzero = [0.1, 0.2]
    val = last_state_pass_noise(A_rbc, B_rbc, C_rbc, u_nonzero, impulse_noise)
    @test isfinite(val)
end

@testset "last_observable with impulse noise" begin
    function last_observable_pass_noise(A, B, C, u0, noise)
        problem = LinearStateSpaceProblem(
            A, B, u0, (0, length(noise)); C, noise,
            observables_noise = nothing, observables = nothing
        )
        sol = solve(problem)
        return sol.z[end][2]
    end
    T_imp = 20
    impulse_noise = [[i == 1 ? 1.0 : 0.0] for i in 1:T_imp]
    u_nonzero = [0.1, 0.2]
    val = last_observable_pass_noise(A_rbc, B_rbc, C_rbc, u_nonzero, impulse_noise)
    @test isfinite(val)
end

# --- Workspace (init/solve!) tests ---

@testset "solve!() matches solve() — simulation with noise, C, and obs_noise" begin
    Random.seed!(1234)
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, 5); C = C_rbc,
        observables_noise = D_rbc
    )
    Random.seed!(1234)
    sol_direct = solve(prob)
    Random.seed!(1234)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.logpdf ≈ sol_direct.logpdf
end

@testset "solve!() matches solve() — joint likelihood (noise + obs + obs_noise)" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc_5));
        C = C_rbc, observables_noise = D_rbc,
        noise = noise_rbc_5, observables = observables_rbc_5
    )
    sol_direct = solve(prob)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.logpdf ≈ sol_direct.logpdf
end

@testset "solve!() matches solve() — no observables (noise + C, no obs/obs_noise)" begin
    Random.seed!(1234)
    prob = LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, 5); C = C_rbc)
    Random.seed!(1234)
    sol_direct = solve(prob)
    Random.seed!(1234)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.logpdf ≈ sol_direct.logpdf
end

@testset "solve!() matches solve() — no noise (B=nothing, C present)" begin
    u_nonzero = [1.1, 0.2]
    prob = LinearStateSpaceProblem(A_rbc, nothing, u_nonzero, (0, 5); C = C_rbc)
    sol_direct = solve(prob)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.W === nothing
    @test sol_ws.logpdf ≈ sol_direct.logpdf
end

@testset "solve!() matches solve() — no observation equation (B=nothing, C=nothing)" begin
    u_nonzero = [1.1, 0.2]
    prob = LinearStateSpaceProblem(A_rbc, nothing, u_nonzero, (0, 5))
    sol_direct = solve(prob)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z === nothing
    @test sol_ws.W === nothing
    @test sol_ws.logpdf ≈ sol_direct.logpdf
end

@testset "solve!() repeated — idempotent results" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc_5));
        C = C_rbc, observables_noise = D_rbc,
        noise = noise_rbc_5, observables = observables_rbc_5
    )
    ws = init(prob, DirectIteration())
    sol1 = solve!(ws)
    sol2 = solve!(ws)
    @test sol1.u ≈ sol2.u
    @test sol1.z ≈ sol2.z
    @test sol1.logpdf ≈ sol2.logpdf
end
