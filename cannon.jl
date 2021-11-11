using Revise
using DifferenceEquations
using Distributions
using LinearAlgebra
using SciMLBase

using DifferenceEquations: LinearStateSpaceProblem, LinearGaussian, solve, 
    StandardGaussian, DefinedNoise


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

@assert sol2.n == sol3.n