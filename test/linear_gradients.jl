using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote,
      Random, ChainRulesCore
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
observables_rbc = readdlm(joinpath(pkgdir(DifferenceEquations),
                                   "test/data/RBC_observables.csv"),
                          ',')' |> collect
noise_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"),
                    ',')' |>
            collect
# Data and Noise
T = 5
observables_rbc = observables_rbc[:, 1:T]
noise_rbc = noise_rbc[:, 1:T]

function z_sum(A, B, C, u0, noise, observables, D; kwargs...)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C,
                                      observables_noise = D,
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
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C,
                                      observables_noise = D,
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
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(observables, 2)); C,
                                      observables_noise = D,
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
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(noise_rbc, 2)); C, noise,
                                      kwargs...)
    sol = solve(problem, DirectIteration())
    return sol.W[1, 2] + sol.W[1, 4] + sol.z[2][2]
end
@testset "no observables gradient" begin
    @test no_observables_sum(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc) ≈
          -0.08892781958364693
    gradient((args...) -> no_observables_sum(args...), A_rbc, B_rbc, C_rbc,
             u0_rbc, noise_rbc)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> no_observables_sum(args...), A_rbc, B_rbc,
               C_rbc, u0_rbc, noise_rbc; rrule_f = rrule_via_ad, check_inferred = false)
end
function no_noise(A, C, u0; kwargs...)
    problem = LinearStateSpaceProblem(A, nothing, u0, (0, 5); C, kwargs...)
    sol = solve(problem, DirectIteration())
    # u = sol.u # bugs with u
    return sol.z[2][2]# + u[2][2]
end
@testset "no noise" begin
    u_nonzero = [1.1, 0.2]
    @test no_noise(A_rbc, C_rbc, u_nonzero) ≈ 2.2943928649664755
    gradient((args...) -> no_noise(args...), A_rbc, C_rbc,
             u_nonzero)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> no_noise(args...), A_rbc, C_rbc, u_nonzero;
               rrule_f = rrule_via_ad,
               check_inferred = false)
end

function no_observation_equation(A, u0; kwargs...)
    problem = LinearStateSpaceProblem(A, nothing, u0, (0, 5); kwargs...)
    sol = solve(problem, DirectIteration())
    u = sol.u # bugs with u
    return u[2][2] + u[4][1]
end
@testset "no observation equation" begin
    u_nonzero = [1.1, 0.2]
    @test no_observation_equation(A_rbc, u_nonzero) ≈ 2.4279222804056597
    gradient((args...) -> no_observation_equation(args...), A_rbc,
             u_nonzero)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> no_observation_equation(args...), A_rbc, u_nonzero;
               rrule_f = rrule_via_ad,
               check_inferred = false)
end

# Hack to set seeds within equation for finite-diff reproducibility
# Makes it ignore the derivative
setseed(x) = Random.seed!(x)
function ChainRulesCore.rrule(::typeof(setseed), x)
    Random.seed!(x)
    pb(ȳ) = (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    return nothing, pb
end

function no_observation_equation_noise(A, B, u0; kwargs...)
    setseed(1234)  # hack for reproducibility with finite diff
    problem = LinearStateSpaceProblem(A, B, u0, (0, 5); kwargs...)
    sol = solve(problem, DirectIteration())
    u = sol.u # bugs with u
    return u[2][2] + u[4][1]
end
@testset "no observation equation" begin
    u_nonzero = [1.1, 0.2]
    @test no_observation_equation_noise(A_rbc, B_rbc, u_nonzero) ≈ 2.3898508744331406
    gradient((args...) -> no_observation_equation_noise(args...), A_rbc, B_rbc,
             u_nonzero)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> no_observation_equation_noise(args...), A_rbc, B_rbc, u_nonzero;
               rrule_f = rrule_via_ad,
               check_inferred = false)
end

function last_state_pass_noise(A, B, C, u0, noise)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(noise, 2)); C, noise,
                                      observables_noise = nothing, observables = nothing)
    sol = solve(problem, DirectIteration())
    return sol.u[end][2]
end

@testset "last state with noise, no observable noise" begin
    T = 20
    noise = Matrix([1.0; zeros(T - 1)]')  # impulse
    u_nonzero = [0.1, 0.2]
    last_state_pass_noise(A_rbc, B_rbc, C_rbc, u_nonzero, noise)
    gradient(last_state_pass_noise, A_rbc, B_rbc, C_rbc, u_nonzero, noise)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (u_nonzero) -> last_state_pass_noise(A_rbc, B_rbc, C_rbc, u_nonzero, noise),
               u_nonzero;
               rrule_f = rrule_via_ad,
               check_inferred = false)
end
function last_observable_pass_noise(A, B, C, u0, noise)
    problem = LinearStateSpaceProblem(A, B, u0, (0, size(noise, 2)); C, noise,
                                      observables_noise = nothing, observables = nothing)
    sol = solve(problem, DirectIteration())
    return sol.z[end][2]
end
@testset "last observable with noise, no observable noise" begin
    T = 20
    noise = Matrix([1.0; zeros(T - 1)]')  # impulse
    u_nonzero = [0.1, 0.2]
    last_observable_pass_noise(A_rbc, B_rbc, C_rbc, u_nonzero, noise)
    gradient(last_observable_pass_noise, A_rbc, B_rbc, C_rbc, u_nonzero, noise)
    test_rrule(Zygote.ZygoteRuleConfig(),
               (u_nonzero) -> last_observable_pass_noise(A_rbc, B_rbc, C_rbc, u_nonzero,
                                                         noise),
               u_nonzero;
               rrule_f = rrule_via_ad,
               check_inferred = false)
end