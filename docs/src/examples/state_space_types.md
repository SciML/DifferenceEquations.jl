### Constructors

- `LinearStateSpaceProblem{isinplace}(f,g,u0,tspan,p=NullParameters();kwargs...)` :
  Defines the discrete problem with the specified functions for a linear-operator.

Parameters are optional, and if not given then a `NullParameters()` singleton
will be used which will throw nice errors if you try to index non-existent
parameters.

### Fields

* `f`: The evolution function.
* `g`: The noise function.
* `h`: The observation function. Defaults perfect observability of the underlying state (i.e. $h(u, p, t_n) = u$)
* `D`: The observation noise distribution. Defaults `nothing`.
* `u0`: The initial condition, which might be a vector or a prior distribution.
* `tspan`: The timespan for the problem.
* `p`: The parameters for the problem. Defaults to `NullParameters`
* `noise`: The noise process applied to the noise upon generation. Only
  Gaussian white noise is currently supported unless values are specified directly
* `observables`: The observables to match when filtering.  Defaults to `nothing`
* `kwargs`: The keyword arguments passed onto the solves.

#### Note About Timing

Note that if no `dt` nor `tstops` is given, it's assumed that `dt=1` and thus
`tspan=(0,T)` will solve for `T+1` iterations. If in the solver `dt` is given, then
the number of iterations will change. And if `tstops` is not empty, the solver will
revert to the standard behavior of fixed timestep methods, which is "step to each
tstop".

## Likelihood and Filtering Calculations
Certain `solve` algorithms will run a filter on the unobservable `u` states and compare to the `observables` if provided.  In that case, it might do so (1) with unobservable `w_t` noise; or (2) conditioning on a particular sequence of $w_{t+1}$ shocks, where the likelihood depends on the unknown observational error `v_t`.

If an algorithm is given for the filtering, then the return type of `solve` will have access to a `logpdf` for the log likelihood.  In addition, the solution will provide information on the sequence of posteriors (and smoothed values, if required).

### Joint Likelihood
In the case of a joint-likelihood where the `noise` (i.e. $w_t$) is given it is not a hidden markov model and the log likelihood simply accumulates the likelihood of each observation.
$$
\mathcal{L}(z, u_0, w) = \sum_{n=1}^N \log P\left(v_t, t_n\right) 
$$
where
$$
v_t = z_n - h(u_n, p, t_n)\\
u_{n+1} = f(u_n,p,t_n) + g(u_n,p,t_n) w_{n+1}
$$
The density is In the case of the typical Gaussian errors, it would be
$$
z_n - h(u_n, p, t_n) \sim D
$$
where a common case is $D = N(0, R)$

### Linear Filtering for the Marginal Likelihood
When the system is linear, there is an exact likelihood for the marginal likelihood using the [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter#Marginal_likelihood).  Unlike the previous example, this is a marginal likelihood and not conditional on the noise, $w$,i.e.  $\mathcal{L}(z, u_0)$.

### Nonlinear Filtering for the Marginal Likelihood
The marginal likelihood can be calculated through a particle filter or unscented Kalman Filter.


## Solution Types
The solutions provided by the `solve` are either a `StateSpaceSolution` or `StateSpaceFilterSolution` depending on the algorithm choice.  It provided as consistent as possible of an interface with `RODESolution` as possible, subject to the difference between observables and latent states in a hidden Markov model.

The type does not have the `interp` and other features not used by a discrete-time model, but adds:

* `observables`: the observable
* `V`: the observational noise (if appropriate and not calculating a likelihood) 
* `logpdf`: the likelihood of the set of observations.  This may be calculated as the accumulated likelihood given observational noise if the `noise` is passed in as an argument (i.e. if it conditions on a particular set of $w_t$, it is not a hidden markov model)

In the case of the `StateSpaceFilterSolution` type, which represents the solution to filtering with a latent state,

* `u`: is the posterior mean of the latent state, potentially smoothed if those algorithms are chosen.
* `posteriors`: A sequence of `Distributions` if a hidden Markov model.
* Neither `W` nor `V` are available as they have been marginalized.