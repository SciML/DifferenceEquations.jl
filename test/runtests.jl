using DifferenceEquations
using Distributions
using Optim
using Random
using LinearAlgebra
using Test

# function ar2_transition(u, p, t) # f
#     A = [p[1] p[2]; 1 0]
#     return A * u
# end

# function ar2_noise(u, p, t) # g
#     return [p[3], 0.0]
# end

# function ar2_observation(u, p, t) # h
#     return [1 0] * u
# end

# Random.seed!(1)

# p = [0.5, -0.25]
# tspan = (1, 100)

# phi1 = 0.5
# phi2 = -0.25
# sigma = 1
# true_theta = [phi1, phi2, sigma]

# Y = zeros(tspan[2])
# n = randn(tspan[2])
# Y += sigma .* n

# for t in 3:length(Y)
#     Y[t] += phi1 * Y[t-1] + phi2 * Y[t-2]
# end

# u0 = [Y[2]; Y[1]]

# @testset "AR2" begin
#     @testset "Provided observations" begin
#         prob = StateSpaceProblem(
#             ar2_transition, # f
#             ar2_noise, # g
#             ar2_observation, # h
#             n,
#             [0.0, 0.0], # u0
#             tspan, # timespan
#             nothing, # observation noise
#             Y # observables
#         )

#         trans_x(x) = [x[1], x[2], exp(x[3])]
#         target(x) = -DifferenceEquations._solve(prob, ConditionalGaussian(), trans_x(x)).likelihood
#         res = optimize(target, [0.0, 0.0, 1.0], Optim.Options(iterations=1000))
#         display(res)
#         println(-res.minimum)
#         println(trans_x(res.minimizer))

#         # Calculate whether we inferred phi1 and phi2
#         @test isapprox(res.minimizer[1], phi1, atol=0.01)
#         @test isapprox(res.minimizer[2], phi2, atol=0.01)
#         @test isapprox(exp(res.minimizer[3]), sigma, atol=0.01)
#     end

#     @testset "No observables" begin
#         prob = StateSpaceProblem(
#             ar2_transition, # f
#             ar2_noise, # g
#             ar2_observation, # h
#             n,
#             [0.0, 0.0], # u0
#             tspan, # timespan
#             nothing,#n, # observation noise
#             nothing#Y # observables
#         )

#         sol1 = DifferenceEquations._solve(prob, ConditionalGaussian(), true_theta)
#         sol2 = DifferenceEquations._solve(prob, ConditionalGaussian(), [0.0, 0.0, 1.0])
#         @test sol1.likelihood > sol2.likelihood
#     end
# end

@testset "Linear model" begin

    A = [0.8 0.0; 0.1 0.7]
    B = Diagonal([0.1, 0.5])
    C = [0.5 0.5] # one observable
    R = [0.01]
    
    # Simulate data
    T = 10
    u0 =[0.0, 0.1]
    tspan = (1, T)
    
    prob1 = LinearStateSpaceProblem(A, B, C, u0, tspan, R=R)
    sol1 = solve(prob1, LinearGaussian())
    
    prob2 = LinearStateSpaceProblem(A, B, C, u0, tspan, R=R, observables=sol1.z)
    sol2 = solve(prob2, LinearGaussian())
    
    prob3 = LinearStateSpaceProblem(A, B, C, u0, tspan, R=R, observables=sol2.z, noise=DefinedNoise(sol2.n))
    sol3 = solve(prob3, LinearGaussian())
    
    @test sol2.n == sol3.n
end

@testset "Nonlinear" begin
    
end