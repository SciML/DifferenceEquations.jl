using Revise
using DifferenceEquations
using DifferentiableStateSpaceModels
using DifferentiableStateSpaceModels.Examples


m = @include_example_module(Examples.rbc_observables)
p_f = (ρ=0.2, δ=0.02, σ=0.01, Ω_1=0.1)
p_d = (α=0.5, β=0.95)

c = SolverCache(m, Val(1), p_d)
sol = generate_perturbation(m, p_d, p_f; cache = c)

T = 9
eps_value = [[0.22], [0.01], [0.14], [0.03], [0.15], [0.21], [0.22], [0.05], [0.18]]
x0 = zeros(m.n_x)

problem = StateSpaceProblem(
    DifferentiableStateSpaceModels.dssm_evolution,
    DifferentiableStateSpaceModels.dssm_volatility,
    DifferentiableStateSpaceModels.dssm_observation,
    x0,
    (1,T),
    sol
)

simul = DifferenceEquations.solve(problem)
