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


## Example 1: Simulating a Linear (and Time-Invariant) State Space Model

Creating a `LinearStateSpaceProblem` and simulating it for a simple, linear equation.

```jldoctest
using DifferenceEquations, LinearAlgebra, Distributions, Random, Plots
A = [0.95 6.2;
     0.0  0.2]
B = [0.0; 0.01;;] # matrix
C = [0.09 0.67;
     1.00 0.00]
D = [0.1, 0.1] # diagonal observation noise
u0 = zeros(2)
T = 20

prob = LinearStateSpaceProblem(A, B, u0, (0, T); C, observables_noise = D, syms = [:a, :b])
sol = solve(prob)

# output
retcode: Success
Interpolation: Piecewise constant interpolation
t: 0:20
u: 21-element Vector{Vector{Float64}}:
 [0.0, 0.0]
 [0.0, -0.008864254379266133]
 [-0.05495837715145002, 0.005465150040881311]
 [-0.018326528040413393, 0.020951951538475705]
 [0.11249189790015665, -0.002745935744685962]
 [0.08984250138809584, -0.005629896869022084]
 [0.05044501573075411, 0.0068082770610265395]
 [0.09013408272258096, -0.0008312122657583546]
 [0.08047386253875011, -0.016595754904355234]
 [-0.026443510995189858, 0.005105441829045822]
 ⋮
 [0.07688586855845986, 0.007591554290850411]
 [0.12010921173380941, 0.013155429954010542]
 [0.1956674168619843, 0.005674032887483874]
 [0.22106304992128512, -0.0026107099494257764]
 [0.19382349573878105, 0.0044787229272008145]
 [0.21190040310048702, 0.010381269649028056]
 [0.2656692547694366, 0.013439353870622119]
 [0.33570978602882184, -0.002902332506551836]
 [0.30092983518675936, -0.014208567084427727]
```

We can `solve` the model to simulate a path.  Since we have not provided the $w_t$ or $v_t$ sequence, it will simulate it using the default Gaussian draws.  The use of the algorithm `LinearGaussian()` is a specialization

```julia
sol = solve(prob, LinearGaussian())  # default algorithm is linear-gaussian iteration
@show sol[1]  # This is the observation at the first time period.
@show sol[1,:]  # the observation of the first value for all periods

plot(sol)  # or plot(sol.z)
```

The `u` state is not-observable in the primary output.  To access the simulated values,

```julia
plot(sol.u)
```

!!! note

    Since the output of the state space model is the observables, `sol[i,j]` refers to `sol.z[i,j]` instead of `sol.u[i,j]`

Assuming that you chose to save the `noise` and the `observational_noise` (i.e. `solve(prob; save_noise = true, save_observational_noise = true`) are the defaults, then you can also access them through

```julia
plot(sol.W)  # noise on the evolution equation
plot(sol.V)  # observational noise, if it exists.
````

## Example 2: Kalman Filter for LTI System

Here, we will setup a `LinearStateSpaceProblem` with a prior, and calculate the likelihood of the observables using the `KalmanFilter` (which will be exact in this case since we provide a linear gaussian model).

```julia
p = [0.8, 0.05, 0.01]
T = 10
u₀_prior = MvNormal([0.0, 0.1], Diagonal([0.01, 0.01])
tspan = (0, T)

prob = LinearStateSpaceProblem(A, B, u₀_prior, tspan; C = C, R = R)  # prior for initial condition
# Simulate some observables, where u0 is drawn from the prior
sol_sim = solve(prob, LinearGaussian())
z = sol_sim
```

Then, attach the observables, and calculate the likelihood,

```julia
prob = LinearStateSpaceProblem(A, B, u₀_prior, tspan; C = C, R = R, observables = z)
sol = solve(prob, KalmanFilter(); save_everystep = true)
@show sol.logpdf
```

Or, we can use the `sol` to extract the sequence of posteriors.

```julia
# Or to extract the posteriors
plot(sol.t, [mean(posterior) for posterior in sol.posteriors])
plot(sol.t, [cov(posterior) for posterior in  sol.posteriors])  # posterior covariance

# TODO: add recipe of some sort?  posterior mean and the 5th/95th quantiles around it?
plot(sol)
```

To run the filter without storing the intermediate values (e.g. if you only need the log likelihood), use the standard `solve` options - where `save_everystep = false` is the default.

```julia
sol = solve(prob, KalmanFilter(); save_everystep = false, save_posteriors = false)
sol.t  # does not save all time periods and saves no posteriors
```

Note: If we wish to run a smoother, we can replace the algorithm with `KalmanSmoother()`, which will update the `u` and `posteriors` on its back-pass.


## Example 3: Differentiating the Kalman Filter

We are then able to differentiate the filter.
```julia
function logpdf_KF(z, u₀, p)
    # parameterize matrices
    A = [p[1] 0.0; 0.1 0.7]
    B = Diagonal([0.1, p[2]])
    C = [0.5 0.5]
    R = [p[3]]  # one observable

    prob = LinearStateSpaceProblem(A, B, u₀,tspan; C=C, R=R)  # Gaussian noise
    return solve(prob, KalmanFilter(), save_everystep = false).logpdf
end
gradient(p -> logpdf_KF(z, u₀, p), p)
```

Note that we have bound the initial prior but could have parameterized and differentiated that as well.
