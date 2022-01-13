using ChainRulesTestUtils, DifferenceEquations, Distributions, DistributionsAD, LinearAlgebra, Test, Zygote
using CSV, DataFrames
using FiniteDiff: finite_difference_gradient

# Matrices from RBC
A_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = [0.1, 0.1]
u0_rbc = zeros(2)

# Data and Noise
T = 5
# observables = [randn(2) for _ in 1:T]
observables_rbc = [[0.06065740756093304, 1.8761799441716314], [-1.9307042321264407, -0.1454342675762673], [0.5520924380514393, -1.5018846999636526], [0.10231137533697832, 1.0545174362282907], [0.15468839844927526, 0.280347305699153]]
# noise = [randn(1) for _ in 1:T]
noise_rbc = [[0.25588177046623284], [0.0414605537784824], [-0.4828412176701489], [0.9852386982044963], [1.2118950851729577]]

# joint case
function joint_likelihood_1(A, B, C, u0, noise, observables, D)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, length(noise)); obs_noise = TuringDiagMvNormal(zeros(length(D)), D), noise, observables)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end

@testset "linear rbc joint likelihood" begin
    @test joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) ≈ -536.0372569648741
    @inferred joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) # would this catch inference problems in the solve?
    test_rrule(Zygote.ZygoteRuleConfig(), joint_likelihood_1, A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc; rrule_f = rrule_via_ad, check_inferred = false)

    res = gradient(joint_likelihood_1, A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc)
    @test res ≈ ([-2.4360645393856517 0.4516572166327964; -19.52943478490269 0.07594715659996228], [219.3752613486913; -286.4024053583998], [1.2115987849879737 0.08821398833652092; 2.68844038727624 -2.5440332554602327], [142.27544836238908, 872.6057912350849], [[2.5344570451477524], [0.05891403400153479], [-9.415566573103447], [-2.2282218173643584], [-0.11373358884039465]], [[-6.23838085127432, -187.61799441716312], [192.85571471832162, 12.954561911192375], [-55.09671860305764, 148.092971334607], [-10.749097877737764, -104.57369352549641], [-16.857238200340635, -32.73567460636627]], [4016.7602080912934, 6880.714243164064])

    # Redundant struct on those matrices
    x = (; A = A_rbc, B = B_rbc, C = C_rbc, u0 = u0_rbc, noise = noise_rbc, observables = observables_rbc, D = D_rbc)
    @test joint_likelihood_1(x.A, x.B, x.C, x.u0, x.noise, x.observables, x.D) ≈ -536.0372569648741
    @inferred joint_likelihood_1(x.A, x.B, x.C, x.u0, x.noise, x.observables, x.D) 
end

# If you are going to see the speed...
# T = 500
# observables = [randn(2) for _ in 1:T]
# noise = [randn(1) for _ in 1:T]
# @time joint_likelihood_1(A, B, C, u0, noise, observables, D)
# @time gradient(joint_likelihood_1, A, B, C, u0, noise, observables, D)

# Kalman only
function kalman_likelihood(A, B, C, u0, observables, D)
    problem = LinearStateSpaceProblem(A, B, C, TuringDenseMvNormal(zeros(length(u0)), cholesky(diagm(ones(length(u0))))), (0, length(observables)); noise = nothing, obs_noise = TuringDiagMvNormal(zeros(length(D)), D), observables)
    return solve(problem, KalmanFilter(); save_everystep = false).loglikelihood
end

@testset "linear rbc kalman likelihood" begin
    @test kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc) ≈ -377.89835856536564
    @inferred kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc) # would this catch inference problems in the solve?
    test_rrule(Zygote.ZygoteRuleConfig(), kalman_likelihood, A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc; rrule_f = rrule_via_ad, check_inferred = false)
end

# Load FVGQ data for checks
path = joinpath(pkgdir(DifferenceEquations), "test", "data")
file_prefix = "FVGQ20"
A = Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_A.csv"); header = false)))
B = Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_B.csv"); header = false)))
C = Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_C.csv"); header = false)))
# D_raw = Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_D.csv"); header = false)))
D = ones(6) * 1e-3
observables_raw = Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_observables.csv"); header = false)))
noise_raw = Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_noise.csv"); header = false)))
observables = [observables_raw[i, :] for i in 1:size(observables_raw, 1)]
noise = [noise_raw[i, :] for i in 1:size(noise_raw, 1)]
u0 = zeros(size(A, 1))

@testset "linear FVGQ joint likelihood" begin
    # The likelihood number is huge, so we just test the gradients.
    # Note: test_rrule struggles on D. Hence we only test a subset of the inputs with FiniteDiff
    res = gradient(joint_likelihood_1, A, B, C, u0, noise, observables, D)
    @test finite_difference_gradient(A -> joint_likelihood_1(A, B, C, u0, noise, observables, D), A) ≈ res[1] rtol = 1e-3
    @test finite_difference_gradient(B -> joint_likelihood_1(A, B, C, u0, noise, observables, D), B) ≈ res[2] rtol = 1e-3
    @test finite_difference_gradient(C -> joint_likelihood_1(A, B, C, u0, noise, observables, D), C) ≈ res[3] rtol = 1e-3
    @test finite_difference_gradient(u0 -> joint_likelihood_1(A, B, C, u0, noise, observables, D), u0) ≈ res[4] rtol = 1e-3
    noise_grad = finite_difference_gradient(noise_mat -> joint_likelihood_1(A, B, C, u0, [noise_mat[i, :] for i in 1:size(noise_mat, 1)], observables, D), noise_raw)
    @test [noise_grad[i, :] for i in 1:size(noise_raw, 1)] ≈ res[5] rtol = 1e-5
end

@testset "linear FVGQ Kalman" begin
    res = gradient(kalman_likelihood, A, B, C, u0, observables, D)
    @test finite_difference_gradient(A -> kalman_likelihood(A, B, C, u0, observables, D), A) ≈ res[1] rtol = 1e-3
    @test finite_difference_gradient(B -> kalman_likelihood(A, B, C, u0, observables, D), B) ≈ res[2] rtol = 1e-3
    @test finite_difference_gradient(C -> kalman_likelihood(A, B, C, u0, observables, D), C) ≈ res[3] rtol = 1e-3
    @test finite_difference_gradient(u0 -> kalman_likelihood(A, B, C, u0, observables, D), u0) ≈ res[4] rtol = 1e-3
end
