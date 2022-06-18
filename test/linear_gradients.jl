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

function z_sum(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                      noise, observables, kwargs...)
    sol = solve(problem, DirectIteration())
    return sol.z[5][1] + sol.z[3][2]
end
@testset "mean_z test" begin
    @test z_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) ≈
          -0.09008162336682057
    gradient((args...) -> z_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
             u0_rbc, noise_rbc)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> z_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc,
               C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)
end
function u_sum(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                      noise, observables, kwargs...)
    sol = solve(problem, DirectIteration())
    u = sol.u  # Zygote bug, must use separate name, also passes Nothing for Δsol so requires workarounds
    return u[3][1] + u[3][2]
    # BROKEN?  ZYGOTE BUG?  Seems to give the wrong Δsol type when calling the pullback
    #    return sol.u[3][1] + sol.u[3][2]  #+  sol[3][1] + sol[3][2] + sol[2,1]
end
@testset "u test" begin
    @test u_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc) ≈
          -0.08780558376240931
    gradient((args...) -> u_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
             u0_rbc, noise_rbc)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> u_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc,
               C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)
end
function W_sum(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C, observables_noise = D,
                                      noise, observables, kwargs...)
    sol = solve(problem, DirectIteration())
    return sol.W[1, 2] + sol.W[1, 4] + sol.z[2][2]
end
@testset "W test" begin
    gradient((args...) -> W_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc, C_rbc,
             u0_rbc, noise_rbc)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> W_sum(args..., observables_rbc, D_rbc), A_rbc, B_rbc,
               C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)
end

# Versions without observations
function no_observables_sum(A, B, C, u0, noise; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(noise_rbc, 2)); C, noise, kwargs...)
    sol = solve(problem, DirectIteration())
    return sol.W[1, 2] + sol.W[1, 4] + sol.z[2][2]
end
@test no_observables_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc) ≈
      -0.08892781958364693
gradient((args...) -> no_observables_sum(args...), A_rbc, B_rbc, C_rbc,
         u0_rbc, noise_rbc)
test_rrule(Zygote.ZygoteRuleConfig(),
           (args...) -> no_observables_sum(args...), A_rbc, B_rbc,
           C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)