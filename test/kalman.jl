using DifferenceEquations, Distributions, LinearAlgebra, Test, DelimitedFiles, DiffEqBase
using DifferenceEquations: init, solve!

# --- Helpers ---

function solve_kalman(A, B, C, u0_prior_mean, u0_prior_var, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(
        A, B, u0_prior_mean, (0, length(observables)); C,
        observables_noise = D,
        u0_prior_mean, u0_prior_var,
        noise = nothing, observables, kwargs...
    )
    return solve(problem)
end

function unvech_5(v)
    return LowerTriangular(
        hcat(
            v[1:5],
            [
                zeros(1);
                v[6:9]
            ],
            [
                zeros(2);
                v[10:12]
            ],
            [
                zeros(3);
                v[13:14]
            ],
            [
                zeros(4);
                v[15]
            ]
        )
    )
end

function solve_kalman_cov(A, B, C, u0_mean, u0_variance_vech, observables, D; kwargs...)
    u0_variance_cholesky = unvech_5(u0_variance_vech)
    u0_variance = u0_variance_cholesky * u0_variance_cholesky'
    problem = LinearStateSpaceProblem(
        A, B, zeros(length(u0_mean)),
        (0, length(observables)); C,
        observables_noise = D,
        u0_prior_mean = u0_mean, u0_prior_var = u0_variance,
        noise = nothing, observables, kwargs...
    )
    return solve(problem)
end

get_matrix(R::AbstractVector) = Diagonal(R)
get_matrix(R::AbstractMatrix) = R

function solve_manual(observables, A, B, C, R_raw, u0_mean, u0_variance, tspan)
    T = tspan[2]
    @assert tspan[1] == 0
    @assert length(observables) == T

    # Gaussian prior
    B_prod = B * B'
    R = get_matrix(R_raw)

    u = Vector{Vector{Float64}}(undef, T + 1) # prior mean
    P = Vector{Matrix{Float64}}(undef, T + 1) # prior variance
    z = Vector{Vector{Float64}}(undef, T + 1) # mean observation

    u[1] = u0_mean
    P[1] = u0_variance
    z[1] = C * u[1]
    loglik = 0.0
    for i in 2:(T + 1)
        # Kalman iteration
        u[i] = A * u[i - 1]
        P[i] = A * P[i - 1] * A' + B_prod
        z[i] = C * u[i]

        CP_i = C * P[i]
        V_temp = CP_i * C' + R
        V = Symmetric((V_temp + V_temp') / 2)
        loglik += logpdf(MvNormal(z[i], V), observables[i - 1])
        K = CP_i' / V  # gain
        u[i] += K * (observables[i - 1] - z[i])
        P[i] -= K * CP_i
    end
    return z, u, P, loglik
end

function solve_manual_cov_lik(A, B, C, u0_mean, u0_variance_vech, observables, R_raw, tspan)
    T = tspan[2]
    @assert tspan[1] == 0
    @assert length(observables) == T

    # Gaussian prior — u0 prior taken from params
    u0_variance_cholesky = unvech_5(u0_variance_vech)
    u0_variance = u0_variance_cholesky * u0_variance_cholesky'
    B_prod = B * B'
    R = get_matrix(R_raw)

    u = u0_mean
    P = u0_variance
    z = C * u
    loglik = 0.0
    for i in 2:(T + 1)
        # Kalman iteration
        u = A * u
        P = A * P * A' + B_prod
        z = C * u

        CP_i = C * P
        V_temp = CP_i * C' + R
        V = (V_temp + V_temp') / 2
        loglik += logpdf(MvNormal(z, V), observables[i - 1])
        K = CP_i' / V  # gain
        u += K * (observables[i - 1] - z)
        P -= K * CP_i
    end
    return loglik
end

function kalman_likelihood(A, B, C, u0, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(
        A, B, u0, (0, length(observables)); C,
        observables_noise = D,
        u0_prior_mean = u0,
        u0_prior_var = diagm(ones(length(u0))),
        noise = nothing, observables, kwargs...
    )
    return solve(problem).logpdf
end

# --- Kalman test data (5x5 model) ---

A_kalman = [
    0.0495388 0.0109918 0.0960529 0.0767147 0.0404643;
    0.020344 0.0627784 0.00865501 0.0394004 0.0601155;
    0.0260677 0.039467 0.0344606 0.033846 0.00224089;
    0.0917289 0.081082 0.0341586 0.0591207 0.0411927;
    0.0837549 0.0515705 0.0429467 0.0209615 0.014668
]
B_kalman = [
    0.589064 0.97337 2.32677;
    0.864922 0.695811 0.618615;
    2.07924 1.11661 0.721113;
    0.995325 1.8416 2.30442;
    1.76884 1.56082 0.749023
]
C_kalman = [
    0.0979797 0.114992 0.0964536 0.110065 0.0946794;
    0.110095 0.0856981 0.0841296 0.0981172 0.0811817;
    0.109134 0.103406 0.112622 0.0925896 0.112384;
    0.0848231 0.0821602 0.099332 0.113586 0.115105
]
D_kalman = abs2.(ones(4) * 0.1)
u0_mean_kalman = zeros(5)
u0_var_kalman = diagm(ones(length(u0_mean_kalman)))

observables_kalman_matrix = readdlm(
    joinpath(
        pkgdir(DifferenceEquations),
        "test/data/Kalman_observables.csv"
    ), ','
)' |> collect
observables_kalman = [observables_kalman_matrix[:, t] for t in 1:size(observables_kalman_matrix, 2)]
T_kalman = 200

D_offdiag = [
    0.01 0.0 0.0 0.0;
    0.0 0.02 0.005 0.01;
    0.0 0.005 0.03 0.0;
    0.0 0.01 0.0 0.04
]

u0_mean = [0.0, 0.0, 0.0, 0.0, 0.0]
u0_var_vech = [
    1.1193770675024004, -0.1755391543370492, -0.8351442110561855,
    0.6799242624030147,
    -0.7627861222280011, 0.1346800868329039, 0.46537792458084976,
    -0.16223737917345768, 0.1772417632124954, 0.2722945202387173,
    -0.3971349857502508, -0.1474011998331263, 0.18113754883619412,
    0.13433861105247683, 0.029171596025489813,
]

R = [
    0.01 0.0 0.0 0.0;
    0.0 0.02 0.005 0.01;
    0.0 0.005 0.03 0.0;
    0.0 0.01 0.0 0.04
]

# --- RBC model data ---

A_rbc = [
    0.9568351489231076 6.209371005755285;
    3.0153731819288737e-18 0.20000000000000007
]
B_rbc = reshape([0.0; -0.01], 2, 1)
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
noise_rbc_matrix = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"),
    ','
)' |>
    collect
T_rbc = 5
observables_rbc = [observables_rbc_matrix[:, t] for t in 1:T_rbc]
noise_rbc = [noise_rbc_matrix[:, t] for t in 1:T_rbc]

# --- FVGQ model data ---

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
)' |>
    collect
noise_FVGQ = [noise_FVGQ_matrix[:, t] for t in 1:size(noise_FVGQ_matrix, 2)]
u0_FVGQ = zeros(size(A_FVGQ, 1))

# --- Tests ---

@testset "Kalman filter — non-square matrices" begin
    z, u, P, loglik = solve_manual(
        observables_kalman, A_kalman, B_kalman, C_kalman,
        D_kalman,
        u0_mean_kalman, u0_var_kalman, [0, T_kalman]
    )
    sol = solve_kalman(
        A_kalman, B_kalman, C_kalman, u0_mean_kalman, u0_var_kalman,
        observables_kalman, D_kalman
    )
    @inferred solve_kalman(
        A_kalman, B_kalman, C_kalman, u0_mean_kalman, u0_var_kalman,
        observables_kalman,
        D_kalman
    )
    @test sol.logpdf ≈ loglik
    @test sol.logpdf ≈ 329.7550738722514
    @test sol.z ≈ z
    @test sol.u ≈ u
    @test sol.P ≈ P
end

@testset "Kalman filter — off-diagonal D" begin
    z, u, P, loglik = solve_manual(
        observables_kalman, A_kalman, B_kalman, C_kalman,
        D_offdiag,
        u0_mean_kalman, u0_var_kalman, [0, T_kalman]
    )
    sol = solve_kalman(
        A_kalman, B_kalman, C_kalman, u0_mean_kalman, u0_var_kalman,
        observables_kalman, D_offdiag
    )
    @inferred solve_kalman(
        A_kalman, B_kalman, C_kalman, u0_mean_kalman, u0_var_kalman,
        observables_kalman,
        D_offdiag
    )
    @test sol.logpdf ≈ loglik
    @test sol.logpdf ≈ 124.86949661078718
    @test sol.z ≈ z
    @test sol.u ≈ u
    @test sol.P ≈ P
end

@testset "Kalman filter — covariance prior likelihood" begin
    loglik = solve_manual_cov_lik(
        A_kalman, B_kalman, C_kalman, u0_mean, u0_var_vech,
        observables_kalman,
        R, [0, T_kalman]
    )
    sol = solve_kalman_cov(
        A_kalman, B_kalman, C_kalman, u0_mean, u0_var_vech,
        observables_kalman,
        R
    )
    @test sol.logpdf ≈ loglik
end

@testset "Kalman inference — RBC" begin
    prob = LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc));
        C = C_rbc,
        observables_noise = D_rbc, observables = observables_rbc,
        u0_prior_mean = u0_rbc,
        u0_prior_var = diagm(ones(length(u0_rbc)))
    )
    @inferred LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, length(observables_rbc));
        C = C_rbc,
        observables_noise = D_rbc,
        observables = observables_rbc,
        u0_prior_mean = u0_rbc,
        u0_prior_var = diagm(ones(length(u0_rbc)))
    )

    sol = solve(prob)
    @inferred solve(prob)

    prob_concrete = DiffEqBase.get_concrete_problem(prob, false)
    @inferred DiffEqBase.get_concrete_problem(prob, false)

    kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc)
    @inferred kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc)
end

@testset "Kalman likelihood — RBC" begin
    @test kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc) ≈
        -607.3698273765538
    @inferred kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc)
end

@testset "Kalman likelihood — FVGQ" begin
    @test kalman_likelihood(
        A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, observables_FVGQ,
        D_FVGQ
    ) ≈
        2253.0905386483046
end

@testset "Kalman failure — ill-conditioned A" begin
    A = [1.0e20 0.0; 1.0e20 0.0]
    u0_prior_var = diagm(1.0e10 * ones(length(u0_rbc)))
    prob = LinearStateSpaceProblem(
        A, B_rbc, u0_rbc, (0, length(observables_rbc));
        C = C_rbc,
        observables_noise = D_rbc, observables = observables_rbc,
        u0_prior_mean = u0_rbc, u0_prior_var
    )
    @test_throws Exception solve(prob)
end

# --- Workspace (init/solve!) tests ---

@testset "solve!() matches solve() — basic Kalman (5x5, non-square)" begin
    z_ref, u_ref, P_ref, loglik_ref = solve_manual(
        observables_kalman, A_kalman, B_kalman, C_kalman,
        D_kalman, u0_mean_kalman, u0_var_kalman, [0, T_kalman]
    )
    prob = LinearStateSpaceProblem(
        A_kalman, B_kalman, u0_mean_kalman, (0, length(observables_kalman));
        C = C_kalman, observables_noise = D_kalman,
        u0_prior_mean = u0_mean_kalman, u0_prior_var = u0_var_kalman,
        noise = nothing, observables = observables_kalman
    )
    ws = init(prob, KalmanFilter())
    sol_ws = solve!(ws)
    @test sol_ws.logpdf ≈ loglik_ref
    @test sol_ws.logpdf ≈ 329.7550738722514
    @test sol_ws.z ≈ z_ref
    @test sol_ws.u ≈ u_ref
    @test sol_ws.P ≈ P_ref
end

@testset "solve!() matches solve() — off-diagonal D" begin
    sol_direct = solve_kalman(
        A_kalman, B_kalman, C_kalman, u0_mean_kalman, u0_var_kalman,
        observables_kalman, D_offdiag
    )
    prob = LinearStateSpaceProblem(
        A_kalman, B_kalman, u0_mean_kalman, (0, length(observables_kalman));
        C = C_kalman, observables_noise = D_offdiag,
        u0_prior_mean = u0_mean_kalman, u0_prior_var = u0_var_kalman,
        noise = nothing, observables = observables_kalman
    )
    ws = init(prob, KalmanFilter())
    sol_ws = solve!(ws)
    @test sol_ws.logpdf ≈ sol_direct.logpdf
    @test sol_ws.logpdf ≈ 124.86949661078718
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.P ≈ sol_direct.P
end

@testset "solve!() matches solve() — covariance prior likelihood" begin
    sol_direct = solve_kalman_cov(
        A_kalman, B_kalman, C_kalman, u0_mean, u0_var_vech,
        observables_kalman, R
    )
    u0_variance_cholesky = unvech_5(u0_var_vech)
    u0_variance = u0_variance_cholesky * u0_variance_cholesky'
    prob = LinearStateSpaceProblem(
        A_kalman, B_kalman, zeros(length(u0_mean)),
        (0, length(observables_kalman));
        C = C_kalman, observables_noise = R,
        u0_prior_mean = u0_mean, u0_prior_var = u0_variance,
        noise = nothing, observables = observables_kalman
    )
    ws = init(prob, KalmanFilter())
    sol_ws = solve!(ws)
    @test sol_ws.logpdf ≈ sol_direct.logpdf
    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.P ≈ sol_direct.P
end

@testset "solve!() repeated — idempotent Kalman results" begin
    prob = LinearStateSpaceProblem(
        A_kalman, B_kalman, u0_mean_kalman, (0, length(observables_kalman));
        C = C_kalman, observables_noise = D_kalman,
        u0_prior_mean = u0_mean_kalman, u0_prior_var = u0_var_kalman,
        noise = nothing, observables = observables_kalman
    )
    ws = init(prob, KalmanFilter())
    sol1 = solve!(ws)
    sol2 = solve!(ws)
    @test sol1.u ≈ sol2.u
    @test sol1.z ≈ sol2.z
    @test sol1.P ≈ sol2.P
    @test sol1.logpdf ≈ sol2.logpdf
end
