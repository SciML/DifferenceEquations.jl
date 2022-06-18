using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

# Matrices from RBC
A_rbc = [0.9568351489231076 6.209371005755285;
         3.0153731819288737e-18 0.20000000000000007]
B_rbc = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = abs2.([0.1, 0.1])
u0_rbc = zeros(2)
observables_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"),
                          ',')' |> collect
noise_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ',')' |>
            collect
# Data and Noise
T = 5
observables_rbc = observables_rbc[:, 1:T]
noise_rbc = noise_rbc[:, 1:T]

function observables_sum(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                      noise, observables, kwargs...)
    sol = solve(problem, DirectIteration())
    return sol.z[5][1] + sol.z[3][2]
end

@test observables_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) ≈
      -0.09008162336682057
gradient((args...) -> observables_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
         u0_rbc, noise_rbc)
test_rrule(Zygote.ZygoteRuleConfig(),
           (args...) -> observables_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc,
           C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)

function u_sum(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                      noise, observables, kwargs...)
    sol = solve(problem, DirectIteration())
    return sol[3][1] + sol[3][2] # + sol.u[2][2] + sol[2,1]
end
@test u_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) ≈ -0.08780558376240931
# BROKEN?  ZYGOTE BUG?  Seems to give the wrong Δsol type when calling the pullback
# gradient((args...) -> u_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
#          u0_rbc, noise_rbc)

function W_sum(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                      noise, observables, kwargs...)
    sol = solve(problem, DirectIteration())
    return sol.W[1, 2] + sol.W[1, 4] + sol.z[2][2]
end

gradient((args...) -> W_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
         u0_rbc, noise_rbc)
test_rrule(Zygote.ZygoteRuleConfig(),
           (args...) -> W_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc,
           C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)