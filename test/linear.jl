using ChainRulesTestUtils, DifferenceEquations, Distributions, DistributionsAD, LinearAlgebra, Test, Zygote
using CSV, DataFrames
using FiniteDiff: finite_difference_gradient

# Matrices from RBC
A_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = [0.1, 0.1]
u0_rbc = zeros(2)

path = joinpath(pkgdir(DifferenceEquations), "test", "data")
file_prefix = "RBC"
observables = collect(Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_observables.csv"); header = false)))')
noise = collect(Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_noise.csv"); header = false)))')

# Data and Noise
T = 5
observables_rbc = observables[:, 1:T]
noise_rbc = noise[:, 1:T]

# joint case
function joint_likelihood_1(A, B, C, u0, noise, observables, D)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, size(noise, 2)); obs_noise = TuringDiagMvNormal(zeros(length(D)), D), noise, observables)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end

@testset "linear rbc joint likelihood" begin
    @test joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) ≈ -690.9407412360038
    @inferred joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) # would this catch inference problems in the solve?
    # We only test A, B, C, and noise
    f = (A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc) -> joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc)
    @test f(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc) ≈ -690.9407412360038
    test_rrule(Zygote.ZygoteRuleConfig(), f, A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)
    # Redundant struct on those matrices
    x = (; A = A_rbc, B = B_rbc, C = C_rbc, u0 = u0_rbc, noise = noise_rbc, observables = observables_rbc, D = D_rbc)
    @test joint_likelihood_1(x.A, x.B, x.C, x.u0, x.noise, x.observables, x.D) ≈ -690.9407412360038
    @inferred joint_likelihood_1(x.A, x.B, x.C, x.u0, x.noise, x.observables, x.D) 
end

# Kalman only
function kalman_likelihood(A, B, C, u0, observables, D)
    problem = LinearStateSpaceProblem(A, B, C, TuringDenseMvNormal(zeros(length(u0)), cholesky(diagm(ones(length(u0))))), (0, size(observables, 2)); noise = nothing, obs_noise = TuringDiagMvNormal(zeros(length(D)), D), observables)
    return solve(problem, KalmanFilter(); save_everystep = false).loglikelihood
end

@testset "linear rbc kalman likelihood" begin
    @test kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc) ≈ -607.3698273765538
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
observables = collect(Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_observables.csv"); header = false)))')
noise = collect(Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_noise.csv"); header = false)))')
u0 = zeros(size(A, 1))

@testset "linear FVGQ joint likelihood" begin
    @test joint_likelihood_1(A, B, C, u0, noise, observables, D) ≈ -1.4648817357717388e9
    @inferred joint_likelihood_1(A, B, C, u0, noise, observables, D)
    f = (A, B, C, noise) -> joint_likelihood_1(A, B, C, u0, noise, observables, D)
    test_rrule(Zygote.ZygoteRuleConfig(), f, A, B, C, noise; rrule_f = rrule_via_ad, check_inferred = false)
end

@testset "linear FVGQ Kalman" begin
    # Note: set rtol to be higher than the default case because of huge gradient numbers
    @test kalman_likelihood(A, B, C, u0, observables, D) ≈ -108.52706300389917
    # test_rrule(Zygote.ZygoteRuleConfig(), kalman_likelihood, A, B, C, u0, observables, D; rrule_f = rrule_via_ad, check_inferred = false, rtol = 1e-5)
    # res = gradient(kalman_likelihood, A, B, C, u0, observables, D)
    # @test finite_difference_gradient(A -> kalman_likelihood(A, B, C, u0, observables, D), A) ≈ res[1] rtol = 1e-3
    # @test finite_difference_gradient(B -> kalman_likelihood(A, B, C, u0, observables, D), B) ≈ res[2] rtol = 1e-3
    # @test finite_difference_gradient(C -> kalman_likelihood(A, B, C, u0, observables, D), C) ≈ res[3] rtol = 1e-3
    # @test finite_difference_gradient(u0 -> kalman_likelihood(A, B, C, u0, observables, D), u0) ≈ res[4] rtol = 1e-3
end
