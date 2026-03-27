# FAQ

## When should I use the Kalman filter vs. joint likelihood?

- **Kalman filter**: Use for linear Gaussian models when you want the marginal likelihood, integrating out the latent noise sequence. This is the standard approach for maximum likelihood estimation (MLE) of parameters.
- **Joint likelihood**: Use when conditioning on a specific noise realization. This is useful for Bayesian methods where the noise sequence is sampled as part of inference (e.g., particle MCMC, HMC on latent variables).

## Why does Enzyme require all arguments to be Duplicated?

Enzyme tracks activity at the struct level. When constructing a `LinearStateSpaceProblem`, all matrix arguments (e.g., `A`, `B`, `C`) flow into a single struct. If any argument is active (i.e., being differentiated), Enzyme needs shadow copies for all arguments in the struct. Passing some arguments as `Const` while others are `Duplicated` triggers an `EnzymeRuntimeActivityError`. The solution is to mark all arguments as `Duplicated`.

## What is the observables timing convention?

The `tspan` `(0, T)` produces `T+1` states: ``u_0, u_1, \ldots, u_T``. Observations ``z_n`` correspond to state ``u_n``. The `observables` keyword expects `T` vectors corresponding to ``z_1, z_2, \ldots, z_T`` (skipping ``z_0``). So when passing simulated data, use `sol.z[2:end]`.

## What does `observables_noise` represent?

The `observables_noise` keyword specifies the observation noise **covariance matrix** (entries are variances and covariances, not standard deviations). It must be an `AbstractMatrix` — use `Diagonal([σ₁², σ₂², …])` for diagonal noise or a full `Matrix`/`Symmetric(H * H')` for correlated noise.

Its behavior depends on context:

- **During simulation** (when `observables` is not provided): used to generate synthetic measurement noise added to the clean observations `sol.z`.
- **During likelihood computation** (when `observables` is provided): used as the observation noise covariance in the log-likelihood calculation.

## How do I differentiate with respect to only some parameters?

- **Enzyme**: All arguments to the likelihood function must be marked `Duplicated` (see [Enzyme AD](@ref)). However, you only need to *read* the shadow of the parameter you care about — the other shadows are computed but can be discarded. There is no performance cost to ignoring unused shadows.
- **ForwardDiff**: Only the parameter passed through `ForwardDiff.gradient`'s vector argument is differentiated. Other parameters are captured as constants in the closure, so no derivatives are computed for them.

## How do I choose between QuadraticStateSpaceProblem and PrunedQuadraticStateSpaceProblem?

- **`PrunedQuadraticStateSpaceProblem`**: Use for second-order perturbation solutions of DSGE models. The pruning prevents explosive dynamics by applying the quadratic term to a separate linear-part state rather than the full nonlinear state.
- **`QuadraticStateSpaceProblem`**: Use if you specifically need the unpruned quadratic form (e.g., for comparison or when the system is known to be stable).
