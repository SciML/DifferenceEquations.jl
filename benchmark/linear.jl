#Benchmarking of RBC and FVGQ variants
using DifferenceEquations, BenchmarkTools
using DelimitedFiles, Distributions, Zygote, LinearAlgebra

const LINEAR = BenchmarkGroup()

# Matrices from RBC
const A_rbc = [0.9568351489231076 6.209371005755285;
               3.0153731819288737e-18 0.20000000000000007]
const B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
const C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
const D_rbc = MvNormal(Diagonal(abs2.([0.1, 0.1])))
const u0_rbc = zeros(2)
const u0_prior_rbc = MvNormal(diagm(ones(length(u0_rbc))))

const observables_rbc = readdlm(joinpath(pkgdir(DifferenceEquations),
                                         "test/data/RBC_observables.csv"), ',')' |> collect
const noise_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ',')' |>
                  collect
const cache_1_rbc = LinearStateSpaceProblemCache{Float64}(size(A_rbc, 1), size(B_rbc, 2),
                                                          size(observables_rbc, 1),
                                                          size(observables_rbc, 2) + 1)
# Matrices from FVGQ
# Load FVGQ data for checks
const A_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A.csv"), ',')
const B_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
const C_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C.csv"), ',')
const D_FVGQ = MvNormal(Diagonal(abs2.(ones(6) * 1e-3)))

const observables_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations),
                                          "test/data/FVGQ20_observables.csv"), ',')' |> collect

const noise_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"),
                           ',')' |> collect
const u0_FVGQ = zeros(size(A_FVGQ, 1))
const u0_prior_FVGQ = MvNormal(diagm(ones(length(u0_FVGQ))))
const cache_1_FVGQ = LinearStateSpaceProblemCache{Float64}(size(A_FVGQ, 1), size(B_FVGQ, 2),
                                                           size(observables_FVGQ, 1),
                                                           size(observables_FVGQ, 2) + 1)

# Specific tests with const arguments bound 
function kalman_rbc(A, B, C, u0_prior)
    problem = LinearStateSpaceProblem(A, B, C, u0_prior, (0, size(observables_rbc, 2), Val(true));
                                      noise = nothing, obs_noise = D_rbc,
                                      observables = observables_rbc)
    return solve(problem, KalmanFilter(); save_everystep = false).loglikelihood
end
function joint_1_rbc(A, B, C, u0, noise)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, size(observables_rbc, 2), Val(false)); noise,
                                      obs_noise = D_rbc, observables = observables_rbc)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end
function kalman_rbc(A, B, C, u0_prior)
    problem = LinearStateSpaceProblem(A, B, C, u0_prior, (0, size(observables_rbc, 2), Val(true));
                                      noise = nothing, obs_noise = D_rbc,
                                      observables = observables_rbc, cache = cache_1_rbc)
    return solve(problem, KalmanFilter(); save_everystep = false).loglikelihood
end
function joint_1_rbc(A, B, C, u0, noise)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, size(observables_rbc, 2), Val(false)); noise,
                                      obs_noise = D_rbc, observables = observables_rbc,
                                      cache = cache_1_rbc)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end
# executing gradients once to avoid compilation time in benchmarking
kalman_rbc(A_rbc, B_rbc, C_rbc, u0_prior_rbc)
gradient(kalman_rbc, A_rbc, B_rbc, C_rbc, u0_prior_rbc)
kalman_rbc(A_rbc, B_rbc, C_rbc, u0_prior_rbc)
gradient(kalman_rbc, A_rbc, B_rbc, C_rbc, u0_prior_rbc)
joint_1_rbc(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc)
gradient(joint_1_rbc, A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc)
joint_1_rbc(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc)
gradient(joint_1_rbc, A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc)

function kalman_FVGQ(A, B, C, u0_prior)
    problem = LinearStateSpaceProblem(A, B, C, u0_prior, (0, size(observables_FVGQ, 2), Val(true));
                                      noise = nothing, obs_noise = D_FVGQ,
                                      observables = observables_FVGQ)
    return solve(problem, KalmanFilter(); save_everystep = false).loglikelihood
end
function joint_1_FVGQ(A, B, C, u0, noise)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, size(observables_FVGQ, 2), Val(false));
                                      noise, obs_noise = D_FVGQ, observables = observables_FVGQ)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end
function kalman_FVGQ(A, B, C, u0_prior)
    problem = LinearStateSpaceProblem(A, B, C, u0_prior, (0, size(observables_FVGQ, 2), Val(true));
                                      noise = nothing, obs_noise = D_FVGQ,
                                      observables = observables_FVGQ, cache = cache_1_FVGQ)
    return solve(problem, KalmanFilter(); save_everystep = false).loglikelihood
end
function joint_1_FVGQ(A, B, C, u0, noise)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, size(observables_FVGQ, 2), Val(false));
                                      noise, obs_noise = D_FVGQ, observables = observables_FVGQ,
                                      cache = cache_1_FVGQ)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end
# executing gradients once to avoid compilation time in benchmarking
kalman_FVGQ(A_FVGQ, B_FVGQ, C_FVGQ, u0_prior_FVGQ)
gradient(kalman_FVGQ, A_FVGQ, B_FVGQ, C_FVGQ, u0_prior_FVGQ)
kalman_FVGQ(A_FVGQ, B_FVGQ, C_FVGQ, u0_prior_FVGQ)
gradient(kalman_FVGQ, A_FVGQ, B_FVGQ, C_FVGQ, u0_prior_FVGQ)
joint_1_FVGQ(A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ)
gradient(joint_1_FVGQ, A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ)
joint_1_FVGQ(A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ)
gradient(joint_1_FVGQ, A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ)

####### Tests

const LINEAR["rbc"] = BenchmarkGroup()

const LINEAR["rbc"]["joint_1_no_cache"] = @benchmarkable joint_1_rbc($A_rbc, $B_rbc, $C_rbc,
                                                                     $u0_rbc, $noise_rbc)
const LINEAR["rbc"]["joint_1_gradient_no_cache"] = @benchmarkable gradient(joint_1_rbc, $A_rbc,
                                                                           $B_rbc, $C_rbc, $u0_rbc,
                                                                           $noise_rbc)
const LINEAR["rbc"]["kalman_no_cache"] = @benchmarkable kalman_rbc($A_rbc, $B_rbc, $C_rbc,
                                                                   $u0_prior_rbc)
const LINEAR["rbc"]["kalman_gradient_no_cache"] = @benchmarkable gradient(kalman_rbc, $A_rbc,
                                                                          $B_rbc, $C_rbc,
                                                                          $u0_prior_rbc)
const LINEAR["rbc"]["joint_1"] = @benchmarkable joint_1_rbc($A_rbc, $B_rbc, $C_rbc, $u0_rbc,
                                                            $noise_rbc)
const LINEAR["rbc"]["joint_1_gradient"] = @benchmarkable gradient(joint_1_rbc, $A_rbc, $B_rbc,
                                                                  $C_rbc, $u0_rbc, $noise_rbc)
const LINEAR["rbc"]["kalman"] = @benchmarkable kalman_rbc($A_rbc, $B_rbc, $C_rbc, $u0_prior_rbc)
const LINEAR["rbc"]["kalman_gradient"] = @benchmarkable gradient(kalman_rbc, $A_rbc, $B_rbc, $C_rbc,
                                                                 $u0_prior_rbc)

# FVGQ sized specific test
const LINEAR["FVGQ"] = BenchmarkGroup()
const LINEAR["FVGQ"]["joint_1_no_cache"] = @benchmarkable joint_1_FVGQ($A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                       $u0_FVGQ, $noise_FVGQ)
const LINEAR["FVGQ"]["joint_1_gradient_no_cache"] = @benchmarkable gradient(joint_1_FVGQ, $A_FVGQ,
                                                                            $B_FVGQ, $C_FVGQ,
                                                                            $u0_FVGQ, $noise_FVGQ)
const LINEAR["FVGQ"]["kalman_no_cache"] = @benchmarkable kalman_FVGQ($A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                     $u0_prior_FVGQ)
const LINEAR["FVGQ"]["kalman_gradient_no_cache"] = @benchmarkable gradient(kalman_FVGQ, $A_FVGQ,
                                                                           $B_FVGQ, $C_FVGQ,
                                                                           $u0_prior_FVGQ)
const LINEAR["FVGQ"]["joint_1"] = @benchmarkable joint_1_FVGQ($A_FVGQ, $B_FVGQ, $C_FVGQ, $u0_FVGQ,
                                                              $noise_FVGQ)
const LINEAR["FVGQ"]["joint_1_gradient"] = @benchmarkable gradient(joint_1_FVGQ, $A_FVGQ, $B_FVGQ,
                                                                   $C_FVGQ, $u0_FVGQ, $noise_FVGQ)
const LINEAR["FVGQ"]["kalman"] = @benchmarkable kalman_FVGQ($A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                            $u0_prior_FVGQ)
const LINEAR["FVGQ"]["kalman_gradient"] = @benchmarkable gradient(kalman_FVGQ, $A_FVGQ, $B_FVGQ,
                                                                  $C_FVGQ, $u0_prior_FVGQ)

# return for the test suite
LINEAR