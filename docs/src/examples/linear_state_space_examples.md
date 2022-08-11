# Linear State Space Examples

This tutorial provides describes the support for linear and linear gaussian state space models.

At this point, the package only supports linear time-invariant models without a separate `p` vector.  The canonical form of the linear model is

```math
u_{n+1} = A u_n + B w_{n+1}
```

with

```math
z_n = C u_n +  v_n
```

and optionally $v_n \sim N(0, D)$ and $w_{n+1} \sim N(0,I)$.  If you pass noise into the solver, it no longer needs to be gaussian.


## Simulating a Linear (and Time-Invariant) State Space Model

Creating a `LinearStateSpaceProblem` and simulating it for a simple, linear equation.

```@example 1
using DifferenceEquations, LinearAlgebra, Distributions, Random, Plots, DataFrames, Zygote
A = [0.95 6.2;
     0.0  0.2]
B = [0.0; 0.01;;] # matrix
C = [0.09 0.67;
     1.00 0.00]
D = [0.1, 0.1] # diagonal observation noise
u0 = zeros(2)
T = 10

prob = LinearStateSpaceProblem(A, B, u0, (0, T); C, observables_noise = D, syms = [:a, :b])
sol = solve(prob)
```

The `u` vector of the simulated solution can be plotted using standard recipes, including use of the optional `syms`.  See [SciML docs](https://diffeq.sciml.ai/latest/basics/plot/) for more options.

```@example 1
plot(sol)
```

By default the solution provides an interface to access the simulated `u`.  That is, `sol.u[...] = sol[...]`,
```@example 1
sol[2]
```
Or to get the first element of the last step

```@example 1
sol[end][1] #first element of last step
```

Finally, to extract the full vector
```@example 1
@show sol[2,:];  # whole second vector
```

The results for all of `sol.u` can be loaded in a dataframe, where the column names will be the (optionally) provided symbols.

```@example 1
df = DataFrame(sol)
```

Other results, such as the simulated noise and observables can be extracted from the solution
```@example 1
sol.z # observables
```
```@example 1
sol.W # Simulated Noise
```

We can also solve the model passing in fixed noise, which will be useful for joint likelihoods.  First lets extract the noise from the previous solution, then rerun the simulation but with a different initial value

```@example 1
noise = sol.W
u0_2 = [0.1, 0.0]
prob2 = LinearStateSpaceProblem(A, B, u0_2, (0, T); C, observables_noise = D, syms = [:a, :b], noise)
sol2 = solve(prob2)
plot(sol2)
```


To construct an IRF we can take the model and perturb just the first element of the noise,
```@example 1
function irf(A, B, C, T = 20)
    noise = Matrix([1.0; zeros(T-1)]')
    problem = LinearStateSpaceProblem(A, B, zeros(2), (0, T); C, noise, syms = [:a, :b])
    return solve(problem)
end
plot(irf(A, B, C))
```

Lets find the 2nd observable at the end of the IRF.

```@example 1
function last_observable_irf(A, B, C)
    sol = irf(A, B, C)
    return sol.z[end][2]  # return 2nd argument of last observable
end
last_observable_irf(A, B, C)
```

But everything in this package is differentiable.  Lets differentiate the observable of the IRF with respect to all of the parameters using `Zygote.jl`,

```@example 1
gradient(last_observable_irf, A, B, C)  # calculates gradient wrt all arguments
```

Gradients of other model elements (e.g. `.u` are also possible.  With this in mind, lets find the gradient of the mean of the 1st element of the IRF of the solution with respect to a particular noise vector.

```@example 1
function mean_u_1(A, B, C, noise, u0, T)
    problem = LinearStateSpaceProblem(A, B, u0, (0, T); noise, syms = [:a, :b]) # leaving off observation equation
    sol = solve(problem)
    u = sol.u # weird bug where temporary necessary to differentiate    
    return mean( u[i][1] for i in 1:T) # can have nontrivial functions and even non-mutating loops
end
u0 = [0.0, 0.0]
noise = sol.W # from simulation above
mean_u_1(A, B, C, noise, u0, T)
gradient((noise, u0)-> mean_u_1(A, B, C, noise, u0, T), noise, u0) # dropping a few arguments from derviative
```

## Simulating Ensembles and Fixing Noise
If you pass in a distribution for the initial condition, it will draw an initial condition.  Below we will simulate from a deterministic evolution equation and without any observation noise.

```@example 1
using Distributions, DiffEqBase
u0 = MvNormal([1.0 0.1; 0.1 1.0])  # mean zero initial conditions
prob = LinearStateSpaceProblem(A, nothing, u0, (0, T); C)
sol = solve(prob)
plot(sol)
```

With this, we can simulate an ensemble of solutions from different initial conditions (and we will turn back on the noise).  The `EnsembleSummary` calculates a set of quantiles by default.

```@example 1
T = 10
trajectories = 50
prob = LinearStateSpaceProblem(A, B, u0, (0, T); C)
sol = solve(EnsembleProblem(prob), DirectIteration(), EnsembleThreads(); trajectories)
summ = EnsembleSummary(sol)  #calculate summarize statistics from the
plot(summ)  # shows quantiles by default
```

## Observables and Marginal Likelihood using a Kalman Filter
If you provide `observables` and provide a distribution for the `observables_noise` then the model can provide a calculation of the likelihood.  

The simplest case is if you use a gaussian prior and have gaussian observation noise.  First, lets simulate some data with included observation noise.  If passing in a matrix or vector, the `observables_noise` argument is intended to be the cholesky of the covariance matrix.  At this point, only diagonal observation noise is allowed.


```@example 1
u0 = MvNormal([1.0 0.1; 0.1 1.0])  # draw from mean zero initial conditions
T = 10
prob = LinearStateSpaceProblem(A, B, u0, (0, T); C, observables_noise = D, syms = [:a, :b])
sol = solve(prob)
sol.z # simulated observables with observation noise
```

Next we will find the log likelihood of these simulated observables using the `u0` as a prior and with the true parameters.
```@example 1
observables = hcat(sol.z...)  # Right now, observables required to be matrix.  See https://github.com/SciML/DifferenceEquations.jl/issues/55 
u0_prior_var = cov(u0)  # use covariance of distribution we drew from.  Mean zero by default.

# new arguments: u0_prior_variance and observables.  The u0 argument is ignored for the filter
prob = LinearStateSpaceProblem(A, B, u0, (0, size(observables,)); C, observables, observables_noise = D, syms = [:a, :b], u0_prior_var)
solve(prob, KalmanFilter())  # the KalmanFilter() is redundant here since it can select it manually given the priors.
```