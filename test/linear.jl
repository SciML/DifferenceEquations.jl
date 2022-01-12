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

# joint case
function joint_likelihood_1(A, B, C, u0, noise, observables, D)
    problem = LinearStateSpaceProblem(A, B, C, u0, (0, length(noise)); obs_noise = TuringDiagMvNormal(zeros(length(D)), D), noise, observables)
    return solve(problem, NoiseConditionalFilter(); save_everystep = false).loglikelihood
end

@testset "linear rbc joint likelihood" begin
    @test joint_likelihood_1(A, B, C, u0, noise, observables, D) ≈ -536.0372569648741
    @inferred joint_likelihood_1(A, B, C, u0, noise, observables, D) # would this catch inference problems in the solve?
    test_rrule(Zygote.ZygoteRuleConfig(), joint_likelihood_1, A, B, C, u0, noise, observables, D; rrule_f = rrule_via_ad, check_inferred = false)

    res = gradient(joint_likelihood_1, A, B, C, u0, noise, observables, D)
    @test res ≈ ([-2.4360645393856517 0.4516572166327964; -19.52943478490269 0.07594715659996228], [219.3752613486913; -286.4024053583998], [1.2115987849879737 0.08821398833652092; 2.68844038727624 -2.5440332554602327], [142.27544836238908, 872.6057912350849], [[2.5344570451477524], [0.05891403400153479], [-9.415566573103447], [-2.2282218173643584], [-0.11373358884039465]], [[-6.23838085127432, -187.61799441716312], [192.85571471832162, 12.954561911192375], [-55.09671860305764, 148.092971334607], [-10.749097877737764, -104.57369352549641], [-16.857238200340635, -32.73567460636627]], [4016.7602080912934, 6880.714243164064])

    # Redundant struct on those matrices
    x = (; A, B, C, u0, noise, observables, D)
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
    @test kalman_likelihood(A, B, C, u0, observables, D) ≈ -377.89835856536564
    @inferred kalman_likelihood(A, B, C, u0, observables, D) # would this catch inference problems in the solve?
    test_rrule(Zygote.ZygoteRuleConfig(), kalman_likelihood, A, B, C, u0, observables, D; rrule_f = rrule_via_ad, check_inferred = false)
end
