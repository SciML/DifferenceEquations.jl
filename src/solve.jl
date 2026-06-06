using DiffEqBase: DEAlgorithm, KeywordArgSilent

abstract type AbstractDifferenceEquationAlgorithm <: DEAlgorithm end

"""
    DirectIteration()

Forward iteration algorithm for state-space problems. Iterates the state transition
equation forward in time, computing the state trajectory `u`, observations `z`,
noise history `W`, and (if `observables` are provided) the joint log-likelihood `logpdf`.

This is the default algorithm for all problem types.

See also: [`KalmanFilter`](@ref).
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
"""
struct ConditionalLikelihood <: AbstractDifferenceEquationAlgorithm end

# The typical algorithm in discrete-time is DirectIteration()
# Unlike continuous time, there aren't many simple variations
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

# Select default algorithm if not provided
function CommonSolve.solve(prob::AbstractStateSpaceProblem; kwargs...)
    return CommonSolve.solve(
        prob,
        default_alg(prob);
        kwargshandle = KeywordArgSilent,
        kwargs...
    )
end
function CommonSolve.solve(prob::AbstractStateSpaceProblem, alg::Nothing, args...; kwargs...)
    return CommonSolve.solve(
        prob,
        default_alg(prob),
        args...;
        kwargshandle = KeywordArgSilent,
        kwargs...
    )
end
