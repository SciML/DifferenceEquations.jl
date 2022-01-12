using ChainRulesTestUtils, DifferenceEquations, Distributions, DistributionsAD, LinearAlgebra, Test, Zygote

# Matrices from RBC
A = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
B = reshape([0.0; -0.01], 2, 1) # make sure B is a matrix
C = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D = [0.1, 0.1]
u0 = zeros(2)

# Data and Noise
T = 5
# observables = [randn(2) for _ in 1:T]
observables = [[0.06065740756093304, 1.8761799441716314], [-1.9307042321264407, -0.1454342675762673], [0.5520924380514393, -1.5018846999636526], [0.10231137533697832, 1.0545174362282907], [0.15468839844927526, 0.280347305699153]]
# noise = [randn(1) for _ in 1:T]
noise = [[0.25588177046623284], [0.0414605537784824], [-0.4828412176701489], [0.9852386982044963], [1.2118950851729577]]

function joint_likelihood_1(A, B, C, u0, noise, observables, D)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, length(noise)); obs_noise = TuringDiagMvNormal(zeros(length(D)), D), noise, observables)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end


@testset "linear rbc joint likelihood" begin
    @test joint_likelihood_1(A, B, C, u0, noise, observables, D) ≈ -536.0372569648741
    @inferred joint_likelihood_1(A, B, C, u0, noise, observables, D) # would this catch inference problems in the solve?
    # NOTE: inference fails for the next line
    # test_rrule(Zygote.ZygoteRuleConfig(), joint_likelihood_1, A, B, C, u0, noise, observables, D; rrule_f = rrule_via_ad)

    # Redundant struct on those matrices
    x = (; A, B, C, u0, noise, observables, D)
    @test joint_likelihood_1(x.A, x.B, x.C, x.u0, x.noise, x.observables, x.D) ≈ -536.0372569648741
    @inferred joint_likelihood_1(x.A, x.B, x.C, x.u0, x.noise, x.observables, x.D) 
end
