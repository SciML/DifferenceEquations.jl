using SciMLBase: AbstractDEAlgorithm, keyword_arg_silent

"""
    AbstractDifferenceEquationAlgorithm <: AbstractDEAlgorithm

Developer supertype for algorithms accepted by DifferenceEquations' state-space
solver. The package currently provides [`DirectIteration`](@ref),
[`KalmanFilter`](@ref), and [`ConditionalLikelihood`](@ref).

Custom subtypes are not a user-facing extension point unless they implement the
internal allocation, transition, observation, and solve hooks documented on the
developer API page.
"""
abstract type AbstractDifferenceEquationAlgorithm <: AbstractDEAlgorithm end

"""
    DirectIteration()

Forward iteration algorithm for state-space problems. Iterates the state transition
equation forward in time, computing the state trajectory `u`, observations `z`,
noise history `W`, and (if `observables` are provided) the joint log-likelihood `logpdf`.

This is the default algorithm for all problem types.

See also: [`KalmanFilter`](@ref).

# Returns

- `DirectIteration`: An algorithm object selecting forward state propagation.

# Fields

`DirectIteration` is stateless and has no fields.

# Examples

```jldoctest
julia> using DifferenceEquations

julia> DirectIteration() isa DifferenceEquations.AbstractDifferenceEquationAlgorithm
true
```
"""
struct DirectIteration <: AbstractDifferenceEquationAlgorithm end

"""
    KalmanFilter()

Kalman filter algorithm for [`LinearStateSpaceProblem`](@ref). Computes filtered
state estimates, posterior covariances, and the marginal log-likelihood.

Automatically selected when the problem provides:
- `u0_prior_mean` and `u0_prior_var` (Gaussian prior),
- `observables` (observed data),
- `observables_noise` (observation noise covariance),
- `noise = nothing` (latent noise is not fixed).

The solution contains filtered means in `sol.u`, posterior covariances in `sol.P`,
predicted observations in `sol.z`, and the marginal log-likelihood in `sol.logpdf`.

See also: [`DirectIteration`](@ref).

# Returns

- `KalmanFilter`: An algorithm object selecting Gaussian state estimation.

# Fields

`KalmanFilter` is stateless and has no fields.

# Examples

```jldoctest
julia> using DifferenceEquations

julia> KalmanFilter() isa DifferenceEquations.AbstractDifferenceEquationAlgorithm
true
```
"""
struct KalmanFilter <: AbstractDifferenceEquationAlgorithm end

"""
    ConditionalLikelihood()

Conditional likelihood (prediction error decomposition) algorithm for
fully-observed state-space models. At each step, predicts the next
observation from the *observed* current state using the transition equation,
and accumulates the Gaussian log-likelihood of the innovation.

Works with all problem types (`LinearStateSpaceProblem`, `StateSpaceProblem`,
`QuadraticStateSpaceProblem`, `PrunedQuadraticStateSpaceProblem`). The only
requirement is additive Gaussian observation noise.

Requires:
- `observables` (observed data y₁, …, y_T),
- `observables_noise` (innovation covariance R).

The solution contains predicted observations in `sol.z` (when an observation
equation is present), the conditional log-likelihood in `sol.logpdf`, and the
state trajectory (clamped to observables) in `sol.u`.

See also: [`DirectIteration`](@ref), [`KalmanFilter`](@ref).

# Returns

- `ConditionalLikelihood`: An algorithm object selecting prediction-error likelihood.

# Fields

`ConditionalLikelihood` is stateless and has no fields.

# Examples

```jldoctest
julia> using DifferenceEquations

julia> ConditionalLikelihood() isa DifferenceEquations.AbstractDifferenceEquationAlgorithm
true
```
"""
struct ConditionalLikelihood <: AbstractDifferenceEquationAlgorithm end

"""
    default_alg(prob::AbstractStateSpaceProblem)

Select the algorithm used by [`solve`](@ref) when the caller does not provide one.
The generic fallback returns [`DirectIteration`](@ref); eligible linear Gaussian
problems use [`KalmanFilter`](@ref) through a more specific method.

This is a developer extension point. A new problem type must either implement a
matching `default_alg(prob)` method or require callers to pass an algorithm
explicitly. The selected algorithm must have matching `alloc_sol` and `alloc_cache`
methods.
"""
default_alg(prob::AbstractStateSpaceProblem) = DirectIteration()

# If a normal prior, normal observational noise, no noise given, and observables provided then can use a kalman filter
function default_alg(
        prob::LinearStateSpaceProblem{
            uType, uPriorMeanType, uPriorVarType,
            tType, P, NP, F, AType, BType, CType,
            RType, ObsType, OS, K,
        }
    ) where {
        uType,
        uPriorMeanType,
        uPriorVarType <:
        AbstractMatrix,
        tType, P,
        NP <: Nothing,
        F,
        AType <:
        AbstractMatrix,
        BType <:
        AbstractMatrix,
        CType <:
        AbstractMatrix,
        RType <:
        AbstractMatrix,
        ObsType <:
        AbstractVector,
        OS, K,
    }
    return KalmanFilter()
end

"""
    solve(prob::AbstractStateSpaceProblem; kwargs...)

Solve a state-space problem using its automatically selected algorithm.

For a [`LinearStateSpaceProblem`](@ref), this selects [`KalmanFilter`](@ref) when the
problem provides a Gaussian initial-state prior, Gaussian observation noise, observed
data, matrix-valued `A`, `B`, and `C`, and leaves `noise = nothing`. Otherwise it uses
[`DirectIteration`](@ref). Pass an algorithm explicitly to override this selection.

# Arguments

  - `prob`: State-space problem to solve.

# Keyword Arguments

- `save_everystep::Bool = true`: Store the complete trajectory when `true`; retain
  only the initial and final states when `false`.
- `perturb_diagonal`: Diagonal perturbation used when factoring observation
  covariance matrices.
- `kwargs...`: Additional options forwarded to the selected algorithm.

# Returns

- `StateSpaceSolution`: The simulated or filtered state-space solution.

# Throws

- `ArgumentError`: If problem dimensions, noise lengths, or observation lengths do
  not match the time span.

# Examples

```jldoctest
julia> using DifferenceEquations

julia> A = [0.95 0.1; 0.0 0.2];

julia> B = [0.0; 0.01;;];

julia> prob = LinearStateSpaceProblem(A, B, zeros(2), (0, 10));

julia> sol = solve(prob);

julia> length(sol.u)
11
```
"""
function CommonSolve.solve(prob::AbstractStateSpaceProblem; kwargs...)
    return CommonSolve.solve(
        prob,
        default_alg(prob);
        kwargshandle = keyword_arg_silent,
        kwargs...
    )
end
function CommonSolve.solve(prob::AbstractStateSpaceProblem, alg::Nothing, args...; kwargs...)
    return CommonSolve.solve(
        prob,
        default_alg(prob),
        args...;
        kwargshandle = keyword_arg_silent,
        kwargs...
    )
end

function CommonSolve.solve(
        prob::AbstractStateSpaceProblem, alg::AbstractDifferenceEquationAlgorithm, args...;
        kwargs...
    )
    return SciMLBase.__solve(prob, alg, args...; kwargs...)
end
