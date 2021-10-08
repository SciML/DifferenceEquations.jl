using DifferenceEquations
using Distributions
using Optim
using Random
using Test

function ar2_transition(u, p, t) # f
    A = [p[1] p[2]; 1 0]
    return A * u
end

function ar2_noise(u, p, t) # g
    return [1.0, 0.0]
end

function ar2_observation(u, p, t) # h
    # @info "Observation" u
    return [1 0] * u
end

Random.seed!(1)

p = [0.5, -0.25]
tspan = (1, 100)

phi1 = 0.5
phi2 = -0.25
sigma = 5.0

Y = zeros(tspan[2])
n = rand(Normal(0, sigma), length(Y))
Y += n

for t in 3:length(Y)
    Y[t] += phi1 * Y[t-1] + phi2 * Y[t-2]
end

u0 = [Y[2]; Y[1]]

prob = StateSpaceProblem(
    ar2_transition,
    ar2_noise,
    ar2_observation,
    MvNormal([1],[1]),
    [0.0, 0.0],
    tspan,
    n,
    Y
)

sol = DifferenceEquations._solve(
    prob,
    ConditionalGaussian(),
    [phi1, phi2, sigma]
)

trans_x(x) = [x[1], x[2], exp(x[3])]
target(x) = -DifferenceEquations._solve(prob, trans_x(x)).likelihood
res = optimize(target, [0.0, 0.0, 1.0])

# Calculate whether we inferred phi1 and phi2
@test isapprox(res.minimizer[1], phi1, atol=0.01)
@test isapprox(res.minimizer[2], phi2, atol=0.01)
