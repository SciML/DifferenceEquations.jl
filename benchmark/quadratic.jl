#Benchmarking of RBC and FVGQ variants
using DifferenceEquations, BenchmarkTools
using DelimitedFiles, Distributions, Zygote
const QUADRATIC = BenchmarkGroup()

# Matrices from RBC
const A_0_rbc = [-7.824904812740593e-5, 0.0]
const A_1_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
const A_2_rbc = cat([-0.00019761505863889124 0.03375055315837927; 0.0 0.0],
                    [0.03375055315837913 3.128758481817603; 0.0 0.0]; dims = 3)
const B_2_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
const C_0_rbc = [7.824904812740593e-5, 0.0]
const C_1_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
const C_2_rbc = cat([-0.00018554166974717046 0.0025652363153049716; 0.0 0.0],
                    [0.002565236315304951 0.3132705036896446; 0.0 0.0]; dims = 3)
const D_2_rbc = [0.1, 0.1]
const u0_2_rbc = zeros(2)

const observables_2_rbc = readdlm(joinpath(pkgdir(DifferenceEquations),
                                           "test/data/RBC_observables.csv"), ',')' |> collect
const noise_2_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"),
                            ',')' |> collect

# Matrices from FVGQ
# Load FVGQ data for checks
A_0_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_0.csv"), ',')
const A_0_FVGQ = vec(A_0_raw)
const A_1_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_1.csv"), ',')
A_2_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_2.csv"), ',')
const A_2_FVGQ = reshape(A_2_raw, length(A_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
const B_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
C_0_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_0.csv"), ',')
const C_0_FVGQ = vec(C_0_raw)
const C_1_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_1.csv"), ',')
C_2_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_2.csv"), ',')
const C_2_FVGQ = reshape(C_2_raw, length(C_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
# D_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "FVGQ_D.csv"); header = false)))
D_2_FVGQ = ones(6) * 1e-3
u0_2_FVGQ = zeros(size(A_1_FVGQ, 1))

const observables_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations),
                                            "test/data/FVGQ20_observables.csv"), ',')' |> collect

const noise_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"),
                             ',')' |> collect

# General likelihood calculation
function joint_likelihood_2(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, observables, D)
    problem = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, (0, size(noise, 2));
                                         obs_noise = MvNormal(Diagonal(abs2.(D))), noise,
                                         observables)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end

# RBC sized specific tests
# Verifying code prior to benchmark
# executing gradients once to avoid compilation time in benchmarking
gradient(joint_likelihood_2, A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc,
         u0_2_rbc, noise_2_rbc, observables_2_rbc, D_2_rbc)
gradient(joint_likelihood_2, A_0_FVGQ, A_1_FVGQ, A_2_FVGQ, B_2_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ,
         u0_2_FVGQ, noise_2_FVGQ, observables_2_FVGQ, D_2_FVGQ)

const QUADRATIC["rbc"] = BenchmarkGroup()
const QUADRATIC["rbc"]["joint_2"] = @benchmarkable joint_likelihood_2($A_0_rbc, $A_1_rbc, $A_2_rbc,
                                                                      $B_2_rbc, $C_0_rbc, $C_1_rbc,
                                                                      $C_2_rbc, $u0_2_rbc,
                                                                      $noise_2_rbc,
                                                                      $observables_2_rbc, $D_2_rbc)
const QUADRATIC["rbc"]["joint_2_gradient"] = @benchmarkable gradient(joint_likelihood_2, $A_0_rbc,
                                                                     $A_1_rbc, $A_2_rbc, $B_2_rbc,
                                                                     $C_0_rbc, $C_1_rbc, $C_2_rbc,
                                                                     $u0_2_rbc, $noise_2_rbc,
                                                                     $observables_2_rbc, $D_2_rbc)

# FVGQ sized specific test
const QUADRATIC["FVGQ"] = BenchmarkGroup()
const QUADRATIC["FVGQ"]["joint_2"] = @benchmarkable joint_likelihood_2($A_0_FVGQ, $A_1_FVGQ,
                                                                       $A_2_FVGQ, $B_2_FVGQ,
                                                                       $C_0_FVGQ, $C_1_FVGQ,
                                                                       $C_2_FVGQ, $u0_2_FVGQ,
                                                                       $noise_2_FVGQ,
                                                                       $observables_2_FVGQ,
                                                                       $D_2_FVGQ)
const QUADRATIC["FVGQ"]["joint_2_gradient"] = @benchmarkable gradient(joint_likelihood_2, $A_0_FVGQ,
                                                                      $A_1_FVGQ, $A_2_FVGQ,
                                                                      $B_2_FVGQ, $C_0_FVGQ,
                                                                      $C_1_FVGQ, $C_2_FVGQ,
                                                                      $u0_2_FVGQ, $noise_2_FVGQ,
                                                                      $observables_2_FVGQ,
                                                                      $D_2_FVGQ)
# return for the test suite
QUADRATIC