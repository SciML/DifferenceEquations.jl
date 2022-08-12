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

and optionally $v_n \sim N(0, D)$ and $w_{n+1} \sim N(0,I)$.  If you pass noise into the solver, it no longer needs to be Gaussian.  More generally, support could be added for $u_{n+1} = A(p,n) u_n + B(p,n) w_{n+1}$ where $p$ is a vector of differentiable parameters, and the $A$ and $B$ are potentially matrix-free operators.


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
    problem = LinearStateSpaceProblem(A, B, u0, (0, T); noise, syms = [:a, :b])
    sol = solve(problem)
    u = sol.u # see issue #75 workaround   
    # can have nontrivial functions and even non-mutating loops 
    return mean( u[i][1] for i in 1:T)
end
u0 = [0.0, 0.0]
noise = sol.W # from simulation above
mean_u_1(A, B, C, noise, u0, T)
# dropping a few arguments from derivative
gradient((noise, u0)-> mean_u_1(A, B, C, noise, u0, T), noise, u0) 
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

The new arguments we pass to the problem creation are `u0_prior_variance, u0_prior_mean,` and `observables`.  The `u0` is ignored for the filtering problem but must match the size.  The `KalmanFilter()` argument to the `solve` is unnecessary since it can select it manually given the priors and observables.

!!! note
    The timing convention is such that `observables` are expected to match the predictions starting at the second time period.  As the likelihood of the first element `u0` comes from a prior, the `observables` start at the next element, and hence the observables and noise sequences should be 1 less than the tspan 

```@example 1
observables = hcat(sol.z...)  # Observables required to be matrix.  Issue #55 
observables = observables[:, 2:end] # see note above on likelihood and timing
noise = copy(sol.W) # save for later
u0_prior_mean = [0.0, 0.0]
# use covariance of distribution we drew from
u0_prior_var = cov(u0)  

prob = LinearStateSpaceProblem(A, B, u0, (0, size(observables,2)); C, observables, observables_noise = D, syms = [:a, :b], u0_prior_var, u0_prior_mean)
sol = solve(prob, KalmanFilter())  
# plot(sol) The `u` is the sequence of posterior means.
sol.logpdf
```

Hence the `logpdf` provides the log likelihood marginalizing out the latent noise variables.

As before, we can differentiate the kalman filter itself.
```@example 1
function kalman_likelihood(A, B, C, D, u0_prior_mean, u0_prior_var, observables)
    prob = LinearStateSpaceProblem(A, B, u0, (0, size(observables,2)); C, observables, observables_noise = D, syms = [:a, :b], u0_prior_var, u0_prior_mean)
    return solve(prob).logpdf  
end
kalman_likelihood(A, B, C, D, u0_prior_mean, u0_prior_var, observables)
# Find the gradient wrt the A, B, C and priors variance.
gradient((A, B, C, u0_prior_var) -> kalman_likelihood(A, B, C, D, u0_prior_mean, u0_prior_var, observables), A, B, C, u0_prior_var)
```

!!! note
    Some of the gradients, such as those for `observables`, have not been implemented so test carefully.  This is a general theme with gradients and `Zygote.jl` in general.  Your best friend in this process is the spectacular [ChainRulesTestUtils.jl](https://github.com/JuliaDiff/ChainRulesTestUtils.jl) package. See `test_rrule` usage in the [linear unit tests](https://github.com/SciML/DifferenceEquations.jl/blob/main/test/linear_gradients.jl).


## Joint Likelihood with Noise
A key application of these methods is to find the joint likelihood of the latent variables (i.e., the `noise`) and the model definition.

The actual calculation of the likelihood is trivial in that case, and just requires iteration of the linear system while accumulating the likelihood given the observation noise.

Crucially, the differentiability with respect to the high-dimensional noise vector enables gradient-based sampling and estimation methods which would otherwise be infeasible.

```@example 1
function joint_likelihood(A, B, C, D, u0, noise, observables)
    prob = LinearStateSpaceProblem(A, B, u0, (0, size(observables,2)); C, observables, observables_noise = D, noise)
    return solve(prob).logpdf
end
u0 = [0.0, 0.0]
joint_likelihood(A, B, C, D, u0, noise, observables)
```

And as always, this can be differentiated with respect to the state-space matrices and the noise.  Choosing a few parameters,
```@example 1
gradient((A, u0, noise) -> joint_likelihood(A, B, C, D, u0, noise, observables), A, u0, noise)
```

## Composition of State Space Models and AD
While the above gradients have been with respect to the full state space objects `A, B`, etc. those themselves could be generated through a separate procedure and the whole object differentiated.  For example, lets repeat the above examples where we generate the `A` matrix from some sort of deep parameters.

First we will generate some observations with a `generate_model` proxy---which could be replaced with something more complicated but still differentiable

```@example 1
function generate_model(β)
    A = [β 6.2;
        0.0  0.2]
    B = Matrix([0.0  0.001]') # [0.0; 0.001;;] gives a zygote bug
    C = [0.09 0.67;
        1.00 0.00]
    D = [0.01, 0.01]
    return (;A,B,C,D)
end

function simulate_model(β, u0;T = 200)
    mod = generate_model(β)
    prob = LinearStateSpaceProblem(mod.A, mod.B, u0, (0, T); mod.C, observables_noise = mod.D)
    sol = solve(prob) # simulates
    observables = hcat(sol.z...)
    observables = observables[:, 2:end] # see note above on likelihood and timing
    return observables, sol.W
end

# Fix a "pseudo-true" and generate noise and observables
β = 0.95
u0 = [0.0, 0.0]
observables, noise = simulate_model(β, u0)
```

Next, we will evaluate the marginal likelihood using the kalman filter for a particular `β` value,
```@example 1
function kalman_model_likelihood(β, u0_prior_mean, u0_prior_var, observables)
    mod = generate_model(β) # generate model from structural parameters
    prob = LinearStateSpaceProblem(mod.A, mod.B, u0, (0, size(observables,2)); mod.C, observables,      observables_noise = mod.D, u0_prior_var, u0_prior_mean)
    return solve(prob).logpdf
end
u0_prior_mean = [0.0, 0.0]
u0_prior_var = [1e-10 0.0;
                0.0 1e-10]  # starting with degenerate prior
kalman_model_likelihood(β, u0_prior_mean, u0_prior_var, observables)
```

Given the observation error we would not expect the pseudo-true to exactly maximimize the log likelihood.  To show this, we can optimize it using using the Optim package and using a gradient-based optimization routine

```@example 1
using Optimization, OptimizationOptimJL
# Create a function to minimize only of β and use Zygote based gradients
kalman_objective(β,p) = -kalman_model_likelihood(β, u0_prior_mean, u0_prior_var, observables)
kalman_objective(0.95, nothing)
gradient(β ->kalman_objective(β, nothing),β) # Verifying it can be differentiated


optf = OptimizationFunction(kalman_objective, Optimization.AutoZygote())
β0 = [0.91] # start off of the pseudotrue
optprob = OptimizationProblem(optf, β0)
optsol = solve(optprob,LBFGS())  # reverse-mode AD is overkill here
```

In this way, this package composes with others such as [DifferentiableStateSpaceModels.jl](https://github.com/HighDimensionalEconLab/DifferentiableStateSpaceModels.jl) which take a set of structural parameters and an expectational difference equation and generate a state-space model.


Similarly, we can find the joint likelihood for a particular `β` value and noise.  Here we will add in prior.  Some form of a prior or regularization is generally necessary for these sorts of nonlinear models.
```@example 1
function joint_model_posterior(β, u0, noise, observables, noise_prior, β_prior)
    mod = generate_model(β) # generate model from structural parameters
    prob = LinearStateSpaceProblem(mod.A, mod.B, u0, (0, size(observables,2)); mod.C, observables,      observables_noise = mod.D, noise)
    return solve(prob).logpdf + sum(logpdf.(noise_prior, noise)) + logpdf(β_prior, β) # posterior
end
u0 = [0.0, 0.0]
noise_prior = Normal(0.0, 1.0)
β_prior = Normal(β, 0.03) # prior local to the true value
joint_model_posterior(β, u0, noise, observables, noise_prior, β_prior)
```

Which we can turn into a differntiable objective adding in a prior on the noise
```@example 1
joint_model_objective(x, p) = -joint_model_posterior(x[1], u0, Matrix(x[2:end]'), observables, noise_prior, β_prior) # extract noise and parameeter from vector
x0 = vcat([0.95], noise[1,:])  # starting at the true noise
joint_model_objective(x0, nothing)
gradient(x ->joint_model_objective(x, nothing),x0) # Verifying it can be differentiated

# optimize
optf = OptimizationFunction(joint_model_objective, Optimization.AutoZygote())
optprob = OptimizationProblem(optf, x0)
optsol = solve(optprob,LBFGS())
```

This "solves" the problem relatively quickly, despite the high-dimensionality.  However, from a statistics perspective note that this last optimization process does not do especially well in recovering the pseudotrue if you increase the prior variance on the `β` parameter.  Maximizing the posterior is usually the wrong thing to do in high-dimensions because the mode is not a typical set.


## Caveats on Gradients and Performance

A few notes on performance and gradients:
1. As this is using reverse-mode AD it will be efficient for fairly large systems as long as the ultimate value of your differentiable program.  With a little extra work and unit tests, it could support structured matrices/etc. as well.
2. Getting to much higher scales, where the `A,B,C,D` are so large that matrix-free operators is necessary, is feasible but will require generalizing those to LinearOperators.  This would be reasonably easy for the joint likelihood and feasible but possible for the Kalman filter
3. At this point, there is no support for forward-mode auto-differentiation.  For smaller systems with a kalman filter, this should dominate the alternatives, and efficient forward-mode AD rules for the kalman filter exist (see the supplementary materials in the the [Differentiable State Space Models](https://github.com/HighDimensionalEconLab/DifferentiableStateSpaceModels.jl) paper).  However, it would be a significant amount of work to add end-to-end support and fulfill standard SciML interfaces, and perhaps waiting for [Enzyme](https://enzyme.mit.edu/julia/) or similar AD systems that provide both forward/reverse/mixed mode makes sense.
4. Forward-mode AD is likely inappropriate for the joint-likelihood based models since the dimensionality of the noise is always large.
5. The gradient rules are written using [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) so in theory they will work with any supporting AD.  In practice, though, Zygote is the most tested and other systems have inconsistent support on Julia at this time.