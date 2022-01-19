#Benchmarking of RBC and FVGQ variants
using DifferenceEquations, BenchmarkTools
using CSV, DataFrames, DistributionsAD, Zygote
const LINEAR = BenchmarkGroup()

# Matrices from RBC
const A_rbc = [0.9568351489231076 6.209371005755285;
               3.0153731819288737e-18 0.20000000000000007]
const B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
const C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
const D_rbc = [0.1, 0.1]
const u0_rbc = zeros(2)

const observables_rbc = Matrix(DataFrame(CSV.File(joinpath(pkgdir(DifferenceEquations),
                                                           "test/data/RBC_observables.csv");
                                                  header = false)))' |> collect
const noise_rbc = Matrix(DataFrame(CSV.File(joinpath(pkgdir(DifferenceEquations),
                                                     "test/data/RBC_noise.csv"); header = false)))' |>
                  collect

# Matrices from FVGQ
# Load FVGQ data for checks
const A_FVGQ = Matrix(DataFrame(CSV.File(joinpath(pkgdir(DifferenceEquations),
                                                  "test/data/FVGQ20_A.csv"); header = false)))
const B_FVGQ = Matrix(DataFrame(CSV.File(joinpath(pkgdir(DifferenceEquations),
                                                  "test/data/FVGQ20_B.csv"); header = false)))
const C_FVGQ = Matrix(DataFrame(CSV.File(joinpath(pkgdir(DifferenceEquations),
                                                  "test/data/FVGQ20_C.csv"); header = false)))
const D_FVGQ = ones(6) * 1e-3

const observables_FVGQ = Matrix(DataFrame(CSV.File(joinpath(pkgdir(DifferenceEquations),
                                                            "test/data/FVGQ20_observables.csv");
                                                   header = false)))' |> collect

const noise_FVGQ = Matrix(DataFrame(CSV.File(joinpath(pkgdir(DifferenceEquations),
                                                      "test/data/FVGQ20_noise.csv");
                                             header = false)))' |> collect
const u0_FVGQ = zeros(size(A_FVGQ, 1))

# General likelihood calculation
function joint_likelihood_1(A, B, C, u0, noise, observables, D)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, size(noise, 2));
                                      obs_noise = TuringDiagMvNormal(zeros(length(D)), D), noise,
                                      observables)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end

# Kalman only
function kalman_likelihood(A, B, C, u0, observables, D)
    problem = LinearStateSpaceProblem(A, B, C,
                                      TuringDenseMvNormal(zeros(length(u0)),
                                                          cholesky(diagm(ones(length(u0))))),
                                      (0, size(observables, 2)); noise = nothing,
                                      obs_noise = TuringDiagMvNormal(zeros(length(D)), D),
                                      observables)
    return solve(problem, KalmanFilter(); save_everystep = false).loglikelihood
end

# RBC sized specific tests
# executing gradients once to avoid compilation time in benchmarking
gradient(joint_likelihood_1, A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc)
gradient(kalman_likelihood, A_rbc, B_rbc, C_rbc, u0_rbc, observables_rbc, D_rbc)
gradient(joint_likelihood_1, A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ, observables_FVGQ, D_FVGQ)
gradient(kalman_likelihood, A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, observables_FVGQ, D_FVGQ)

const LINEAR["rbc"] = BenchmarkGroup()
const LINEAR["rbc"]["joint_1"] = @benchmarkable joint_likelihood_1($A_rbc, $B_rbc, $C_rbc, $u0_rbc,
                                                                   $noise_rbc, $observables_rbc,
                                                                   $D_rbc)
const LINEAR["rbc"]["joint_1_gradient"] = @benchmarkable gradient(joint_likelihood_1, $A_rbc,
                                                                  $B_rbc, $C_rbc, $u0_rbc,
                                                                  $noise_rbc, $observables_rbc,
                                                                  $D_rbc)
const LINEAR["rbc"] = BenchmarkGroup()
const LINEAR["rbc"]["joint_1"] = @benchmarkable joint_likelihood_1($A_rbc, $B_rbc, $C_rbc, $u0_rbc,
                                                                   $noise_rbc, $observables_rbc,
                                                                   $D_rbc)
const LINEAR["rbc"]["joint_1_gradient"] = @benchmarkable gradient(joint_likelihood_1, $A_rbc,
                                                                  $B_rbc, $C_rbc, $u0_rbc,
                                                                  $noise_rbc, $observables_rbc,
                                                                  $D_rbc)
const LINEAR["rbc"]["kalman"] = @benchmarkable kalman_likelihood($A_rbc, $B_rbc, $C_rbc, $u0_rbc,
                                                                 $observables_rbc, $D_rbc)
const LINEAR["rbc"]["kalman_gradient"] = @benchmarkable gradient(kalman_likelihood, $A_rbc, $B_rbc,
                                                                 $C_rbc, $u0_rbc, $observables_rbc,
                                                                 $D_rbc)

# FVGQ sized specific test
const LINEAR["FVGQ"] = BenchmarkGroup()
const LINEAR["FVGQ"]["joint_1"] = @benchmarkable joint_likelihood_1($A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                    $u0_FVGQ, $noise_FVGQ,
                                                                    $observables_FVGQ, $D_FVGQ)
const LINEAR["FVGQ"]["joint_1_gradient"] = @benchmarkable gradient(joint_likelihood_1, $A_FVGQ,
                                                                   $B_FVGQ, $C_FVGQ, $u0_FVGQ,
                                                                   $noise_FVGQ, $observables_FVGQ,
                                                                   $D_FVGQ)
const LINEAR["FVGQ"]["kalman"] = @benchmarkable kalman_likelihood($A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                  $u0_FVGQ, $observables_FVGQ,
                                                                  $D_FVGQ)
const LINEAR["FVGQ"]["kalman_gradient"] = @benchmarkable gradient(kalman_likelihood, $A_FVGQ,
                                                                  $B_FVGQ, $C_FVGQ, $u0_FVGQ,
                                                                  $observables_FVGQ, $D_FVGQ)

# return for the test suite
LINEAR