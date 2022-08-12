# Quadratic State Space Examples

Second-order state-space models here have pruning as in [Andreasen, Fernandez-Villaverde, and Rubio-Ramirez (2017)](https://www.sas.upenn.edu/~jesusfv/Pruning.pdf).


At this point, the package only supports linear time-invariant models without a separate `p` vector.  The canonical form is

```math
u_{n+1} = A_0 + A_1 u_n + u_n^{\top} A_2 u_n + B w_{n+1}
```

with

```math
z_n = C_0 + C_1 u_n + u_n^{\top} C_2 u_n +  v_n
```

and optionally $v_n \sim N(0, D)$ and $w_{n+1} \sim N(0,I)$.  If you pass noise into the solver, it no longer needs to be Gaussian.


!!! note
    Quadratic state-space models do not have the full feature coverage relative to the linear models.  In particular the auto-differentiation rules are only currently implemented for the `logpdf` required for estimation, and the simulation doesn't have much flexibility on which model elements can be missing.

## Simulating a Quadratic (and Time-Invariant) State Space Model

Creating a `QuadraticStateSpaceModel` is similar to the Linear version described previously.
I 
using DifferenceEquations, LinearAlgebra, Distributions, Random, Plots, DataFrames, Zygote, DiffEqBase
A_0 =  [-7.824904812740593e-5, 0.0]
A_1 = [0.95 6.2;
     0.0  0.2]
A_2 = cat([-0.0002 0.0334; 0.0 0.0],
              [0.034 3.129; 0.0 0.0]; dims = 3)
B = [0.0; 0.01;;] # matrix
C_0 = [7.8e-5, 0.0]
C_1 = [0.09 0.67;
     1.00 0.00]
C_2 = cat([-0.00019 0.0026; 0.0 0.0],
    [0.0026 0.313; 0.0 0.0]; dims = 3)
D = [0.01, 0.01] # diagonal observation noise
u0 = zeros(2)
T = 30

prob = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, T); C_0, C_1, C_2, observables_noise = D, syms = [:a, :b])
sol = solve(prob)
```

As in the linear case, this model can be simulated and plotted
```@example 2
plot(sol)
```

And the observables and noise can be stored
```@example 2
observables = hcat(sol.z...)  # Observables required to be matrix.  Issue #55 
observables = observables[:, 2:end] # see note above on likelihood and timing
noise = sol.W
```

Ensembles work as well,

```@example 2
trajectories = 50
u0_dist = MvNormal([1.0 0.1; 0.1 1.0])  # mean zero initial conditions
prob = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0_dist, (0, T); C_0, C_1, C_2, observables_noise = D, syms = [:a, :b])
ens_sol = solve(EnsembleProblem(prob), DirectIteration(), EnsembleThreads(); trajectories)
summ = EnsembleSummary(ens_sol)  # calculate summarize statistics such as quantiles
plot(summ)
```

## Joint Likelihood with Noise
To calculate the likelihood, the Kalman Filter is no longer applicable.  However, we can still calculate the joint likelihood we did in the linear examples.  Using the simulated observables and noise,

```@example 2
function joint_likelihood_quad(A_0, A_1, A_2, B, C_0, C_1, C_2, D, u0, noise, observables)
    prob = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, size(observables,2)); C_0, C_1, C_2, observables, observables_noise = D, noise)
    return solve(prob).logpdf
end
u0 = [0.0, 0.0]
joint_likelihood_quad(A_0, A_1, A_2, B, C_0, C_1, C_2, D, u0, noise, observables)
```
Which, in turn can itself be differentiated.

```@example 2
gradient((A_0, A_1, A_2, B, C_0, C_1, C_2, noise) -> joint_likelihood_quad(A_0, A_1, A_2, B, C_0, C_1, C_2, D, u0, noise, observables), A_0, A_1, A_2, B, C_0, C_1, C_2, noise)
```
Note that this is calculating the gradient of the likelihood with respect to the underlying canonical representations for the quadratic state space form, but also the entire noise vector.

As in the linear case, this likelihood calculation can nested such that a separate differentiable function could generate the quadratic state space model and the gradients could be over a smaller set of structural parameters.