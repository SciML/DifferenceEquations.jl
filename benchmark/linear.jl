#Benchmarking of RBC and FVGQ variants
using DifferenceEquations, BenchmarkTools
using DelimitedFiles, Distributions, Zygote, LinearAlgebra

# for benchmarking construction itself
function make_problem_1(A, B, C, u0, noise, observables, D; kwargs...)
    prob = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                   noise, observables, kwargs...)
    return prob.A[1, 1] + prob.B[1, 1]
end
function make_problem_kalman(A, B, C, u0_prior, observables, D; kwargs...)
    prob = LinearStateSpaceProblem(A, B, u0_prior, (0, size(observables, 2)); C, u0_prior,
                                   observables_noise = D, noise = nothing, observables, kwargs...)
    return prob.A[1, 1] + prob.B[1, 1]
end

# for benchmarking likelihoods
function joint_likelihood_1(A, B, C, u0, noise, observables, D; kwargs...)
    prob = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                   noise, observables, kwargs...)
    return solve(prob).logpdf
end
function kalman_likelihood(A, B, C, u0_prior, observables, D; kwargs...)
    prob = LinearStateSpaceProblem(A, B, u0_prior, (0, size(observables, 2)); C, u0_prior,
                                   observables_noise = D, noise = nothing, observables, kwargs...)
    return solve(prob).logpdf
end

function simulate_model_no_noise_1(A, B, C, u0, observables, D; kwargs...)
    prob = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                   observables, kwargs...)
    sol = solve(prob)
    return sol.retcode
end

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

# executing gradients once to avoid compilation time in benchmarking

make_problem_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc)
gradient((args...) -> make_problem_1(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc, u0_rbc,
         noise_rbc)

make_problem_kalman(A_rbc, B_rbc, C_rbc, u0_prior_rbc, observables_rbc, D_rbc)
gradient((args...) -> make_problem_kalman(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
         u0_prior_rbc)

kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_prior_rbc, observables_rbc, D_rbc)
gradient((args...) -> kalman_likelihood(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
         u0_prior_rbc)
joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc)
gradient((args...) -> joint_likelihood_1(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
         u0_rbc, noise_rbc)

kalman_likelihood(A_FVGQ, B_FVGQ, C_FVGQ, u0_prior_FVGQ, observables_FVGQ, D_FVGQ)
gradient((args...) -> kalman_likelihood(args..., observables_FVGQ, D_FVGQ), A_FVGQ, B_FVGQ, C_FVGQ,
         u0_prior_FVGQ)
joint_likelihood_1(A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ, observables_FVGQ, D_FVGQ)
gradient((args...) -> joint_likelihood_1(args..., observables_FVGQ, D_FVGQ), A_FVGQ, B_FVGQ, C_FVGQ,
         u0_FVGQ, noise_FVGQ)

####### Becnmarks

const LINEAR = BenchmarkGroup()

const LINEAR["rbc"] = BenchmarkGroup()

const LINEAR["rbc"]["make_problem_1"] = @benchmarkable make_problem_1($A_rbc, $B_rbc, $C_rbc,
                                                                      $u0_rbc, $noise_rbc,
                                                                      $observables_rbc, $D_rbc)
const LINEAR["rbc"]["make_problem_1_gradient"] = @benchmarkable gradient((args...) -> make_problem_1(args...,
                                                                                                     $observables_rbc,
                                                                                                     $D_rbc),
                                                                         $A_rbc, $B_rbc, $C_rbc,
                                                                         $u0_rbc, $noise_rbc)

const LINEAR["rbc"]["simulate_model_no_noise_1"] = @benchmarkable simulate_model_no_noise_1($A_rbc,
                                                                                            $B_rbc,
                                                                                            $C_rbc,
                                                                                            $u0_rbc,
                                                                                            $observables_rbc,
                                                                                            $D_rbc)
const LINEAR["rbc"]["joint_1"] = @benchmarkable joint_likelihood_1($A_rbc, $B_rbc, $C_rbc, $u0_rbc,
                                                                   $noise_rbc, $observables_rbc,
                                                                   $D_rbc)
const LINEAR["rbc"]["joint_1_gradient"] = @benchmarkable gradient((args...) -> joint_likelihood_1(args...,
                                                                                                  $observables_rbc,
                                                                                                  $D_rbc),
                                                                  $A_rbc, $B_rbc, $C_rbc, $u0_rbc,
                                                                  $noise_rbc)
const LINEAR["rbc"]["make_problem_kalman"] = @benchmarkable make_problem_kalman($A_rbc, $B_rbc,
                                                                                $C_rbc,
                                                                                $u0_prior_rbc,
                                                                                $observables_rbc,
                                                                                $D_rbc)
const LINEAR["rbc"]["make_problem_kalman_gradient"] = @benchmarkable gradient((args...) -> make_problem_kalman(args...,
                                                                                                               $observables_rbc,
                                                                                                               $D_rbc),
                                                                              $A_rbc, $B_rbc,
                                                                              $C_rbc, $u0_prior_rbc)
const LINEAR["rbc"]["kalman"] = @benchmarkable kalman_likelihood($A_rbc, $B_rbc, $C_rbc,
                                                                 $u0_prior_rbc, $observables_rbc,
                                                                 $D_rbc)
const LINEAR["rbc"]["kalman_gradient"] = @benchmarkable gradient((args...) -> kalman_likelihood(args...,
                                                                                                $observables_rbc,
                                                                                                $D_rbc),
                                                                 $A_rbc, $B_rbc, $C_rbc,
                                                                 $u0_prior_rbc)

# FVGQ
const LINEAR["FVGQ"] = BenchmarkGroup()
const LINEAR["FVGQ"]["make_problem_1"] = @benchmarkable make_problem_1($A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                       $u0_FVGQ, $noise_FVGQ,
                                                                       $observables_FVGQ, $D_FVGQ)
const LINEAR["FVGQ"]["make_problem_1_gradient"] = @benchmarkable gradient((args...) -> make_problem_1(args...,
                                                                                                      $observables_FVGQ,
                                                                                                      $D_FVGQ),
                                                                          $A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                          $u0_FVGQ, $noise_FVGQ)
const LINEAR["FVGQ"]["simulate_model_no_noise_1"] = @benchmarkable simulate_model_no_noise_1($A_FVGQ,
                                                                                             $B_FVGQ,
                                                                                             $C_FVGQ,
                                                                                             $u0_FVGQ,
                                                                                             $observables_FVGQ,
                                                                                             $D_FVGQ)
const LINEAR["FVGQ"]["joint_1"] = @benchmarkable joint_likelihood_1($A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                    $u0_FVGQ, $noise_FVGQ,
                                                                    $observables_FVGQ, $D_FVGQ)
const LINEAR["FVGQ"]["joint_1_gradient"] = @benchmarkable gradient((args...) -> joint_likelihood_1(args...,
                                                                                                   $observables_FVGQ,
                                                                                                   $D_FVGQ),
                                                                   $A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                   $u0_FVGQ, $noise_FVGQ)
const LINEAR["FVGQ"]["make_problem_kalman"] = @benchmarkable make_problem_kalman($A_FVGQ, $B_FVGQ,
                                                                                 $C_FVGQ,
                                                                                 $u0_prior_FVGQ,
                                                                                 $observables_FVGQ,
                                                                                 $D_FVGQ)
const LINEAR["FVGQ"]["make_problem_kalman_gradient"] = @benchmarkable gradient((args...) -> make_problem_kalman(args...,
                                                                                                                $observables_FVGQ,
                                                                                                                $D_FVGQ),
                                                                               $A_FVGQ, $B_FVGQ,
                                                                               $C_FVGQ,
                                                                               $u0_prior_FVGQ)
const LINEAR["FVGQ"]["kalman"] = @benchmarkable kalman_likelihood($A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                  $u0_prior_FVGQ, $observables_FVGQ,
                                                                  $D_FVGQ)
const LINEAR["FVGQ"]["kalman_gradient"] = @benchmarkable gradient((args...) -> kalman_likelihood(args...,
                                                                                                 $observables_FVGQ,
                                                                                                 $D_FVGQ),
                                                                  $A_FVGQ, $B_FVGQ, $C_FVGQ,
                                                                  $u0_prior_FVGQ)

# return for the test suite
LINEAR