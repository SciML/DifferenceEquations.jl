"""
    AbstractStateSpaceProblem <: DEProblem

Abstract supertype for all discrete-time state-space problems in DifferenceEquations.jl.

Subtypes include [`LinearStateSpaceProblem`](@ref), [`QuadraticStateSpaceProblem`](@ref),
[`PrunedQuadraticStateSpaceProblem`](@ref), and [`StateSpaceProblem`](@ref).
"""
abstract type AbstractStateSpaceProblem <: DEProblem end

# TODO: Can add in more checks on the algorithm choice
DiffEqBase.check_prob_alg_pairing(prob::AbstractStateSpaceProblem, alg) = nothing

# Perturbation problems don't have f, g
# In discrete time, tspan should not have a sensitivity so the concretization is less obvious
function DiffEqBase.get_concrete_problem(
        prob::AbstractStateSpaceProblem, isadapt;
        kwargs...
    )
    p = get_concrete_p(prob, kwargs)
    tspan = prob.tspan #get_concrete_tspan(prob, isadapt, kwargs, p)
    u0 = get_concrete_u0(prob, isadapt, tspan[1], kwargs)
    u0_promote = promote_u0(u0, p, tspan[1])

    if isconcreteu0(prob, tspan[1], kwargs) &&
            typeof(u0_promote) === typeof(prob.u0) &&
            p === prob.p &&
            return prob
    else
        return remake(prob; u0 = u0_promote, p)
    end
end

SciMLBase.isinplace(prob::AbstractStateSpaceProblem) = false  # necessary for the get_concrete_u0 overloads

"""
    LinearStateSpaceProblem(A, B, u0, tspan[, p]; kwargs...)

Define a linear time-invariant state-space model:

```math
u_{n+1} = A \\, u_n + B \\, w_{n+1}
```

with optional observation equation ``z_n = C \\, u_n + v_n``.

# Positional Arguments
- `A`: Transition matrix (n×n).
- `B`: Noise input matrix (n×k), or `nothing` for deterministic dynamics.
- `u0`: Initial state vector, or a `Distribution` for random initial conditions.
- `tspan`: Time span as `(t0, t_end)` with integer distance (e.g., `(0, T)`).
- `p`: Parameters (default: `NullParameters()`).

# Keyword Arguments
- `C`: Observation matrix (m×n). If `nothing`, no observations are computed.
- `observables_noise`: Observation noise covariance. A `Vector` is treated as a diagonal.
- `observables`: Observed data as `Vector{Vector{T}}` for likelihood computation.
- `noise`: Fixed noise sequence as `Vector{Vector{T}}`. If `nothing`, noise is drawn randomly.
- `u0_prior_mean`: Prior mean for Kalman filtering.
- `u0_prior_var`: Prior covariance matrix for Kalman filtering.
- `syms`: State variable names (e.g., `(:x, :y)`) for symbolic indexing.
- `obs_syms`: Observation variable names for symbolic indexing.

# Notes
- Providing `u0_prior_mean`, `u0_prior_var`, `observables`, and `observables_noise`
  (with `noise = nothing`) triggers automatic selection of [`KalmanFilter`](@ref).
- The `observables` timing convention: observations correspond to ``z_1, z_2, \\ldots``
  (starting from the second state), so pass `T` observations for a `tspan` of `(0, T)`.

See also: [`StateSpaceProblem`](@ref), [`QuadraticStateSpaceProblem`](@ref),
[`DirectIteration`](@ref), [`KalmanFilter`](@ref).
"""
struct LinearStateSpaceProblem{
        uType, uPriorMeanType, uPriorVarType, tType, P, NP, F, AType,
        BType, CType,
        RType, ObsType, OS, K,
    } <:
    AbstractStateSpaceProblem
    f::F # HACK: used only for standard interfaces/syms/etc., not used in calculations
    A::AType
    B::BType
    C::CType
    observables_noise::RType
    observables::ObsType
    u0::uType
    u0_prior_mean::uPriorMeanType
    u0_prior_var::uPriorVarType
    tspan::tType
    p::P
    noise::NP
    obs_syms::OS
    kwargs::K
    @add_kwonly function LinearStateSpaceProblem{iip}(
            A, B, u0, tspan, p = NullParameters();
            u0_prior_mean = nothing,
            u0_prior_var = nothing, C = nothing,
            observables_noise = nothing,
            observables = nothing,
            noise = nothing,
            syms = nothing,
            obs_syms = nothing,
            f = nothing,
            kwargs...
        ) where {iip}
        if f === nothing
            f = ODEFunction{false}(
                (u, p, t) -> error("not implemented");
                sys = SymbolCache(syms)
            )
        end
        _tspan = promote_tspan(tspan)
        # _observables = promote_vv(observables)
        _observables = observables

        # Require integer distances between time periods for now.  Later could check with dt != 1
        @assert round(_tspan[2] - _tspan[1]) - (_tspan[2] - _tspan[1]) ≈ 0.0

        return new{
            typeof(u0), typeof(u0_prior_mean), typeof(u0_prior_var), typeof(_tspan),
            typeof(p),
            typeof(noise), typeof(f),
            typeof(A), typeof(B), typeof(C), typeof(observables_noise),
            typeof(_observables), typeof(obs_syms),
            typeof(kwargs),
        }(
            f, A, B, C, observables_noise, _observables, u0,
            u0_prior_mean,
            u0_prior_var,
            _tspan, p, noise, obs_syms, kwargs
        )
    end
end
# just forwards to a iip = false case
function LinearStateSpaceProblem(args...; kwargs...)
    return LinearStateSpaceProblem{false}(args...; kwargs...)
end

"""
    StateSpaceProblem(transition, observation, u0, tspan[, p]; n_shocks, kwargs...)

Define a generic state-space model with user-provided callback functions:

```math
u_{n+1} = f(u_n, w_{n+1}, p, t_n), \\quad z_n = g(u_n, p, t_n)
```

# Positional Arguments
- `transition`: Callback `f!!(x_next, x, w, p, t) -> x_next`. For mutable arrays, mutate
  `x_next` in place and return it; for immutable arrays (e.g., `SVector`), return a new value.
- `observation`: Callback `g!!(y, x, p, t) -> y`, or `nothing` for no observations.
- `u0`: Initial state vector, or a `Distribution` for random initial conditions.
- `tspan`: Time span as `(t0, t_end)` with integer distance.
- `p`: Parameters passed to the callbacks (default: `NullParameters()`).

# Keyword Arguments
- `n_shocks::Int`: Number of noise dimensions (required).
- `n_obs::Int`: Number of observation dimensions (default: `0`).
- `observables_noise`: Observation noise covariance for likelihood computation.
- `observables`: Observed data as `Vector{Vector{T}}`.
- `noise`: Fixed noise sequence as `Vector{Vector{T}}`.
- `syms`: State variable names for symbolic indexing.
- `obs_syms`: Observation variable names for symbolic indexing.

See also: [`LinearStateSpaceProblem`](@ref), [`DirectIteration`](@ref).
"""
struct StateSpaceProblem{
        uType, tType, P, NP, TF, GF, F,
        RType, ObsType, OS, K,
    } <: AbstractStateSpaceProblem
    f::F # HACK: used only for standard interfaces/syms/etc., not used in calculations
    transition::TF     # f!!(x_next, x, w, p, t) -> x_next
    observation::GF    # g!!(y, x, p, t) -> y (or nothing)
    observables_noise::RType
    observables::ObsType
    u0::uType
    tspan::tType
    p::P
    noise::NP
    n_shocks::Int
    n_obs::Int         # 0 if no observation equation
    obs_syms::OS
    kwargs::K
    @add_kwonly function StateSpaceProblem{iip}(
            transition, observation, u0, tspan, p = NullParameters();
            n_shocks,
            n_obs = 0,
            observables_noise = nothing,
            observables = nothing,
            noise = nothing,
            syms = nothing,
            obs_syms = nothing,
            f = nothing,
            kwargs...
        ) where {iip}
        if f === nothing
            f = ODEFunction{false}(
                (u, p, t) -> error("not implemented");
                sys = SymbolCache(syms)
            )
        end
        _tspan = promote_tspan(tspan)
        _observables = observables

        # Require integer distances between time periods for now.
        @assert round(_tspan[2] - _tspan[1]) - (_tspan[2] - _tspan[1]) ≈ 0.0

        return new{
            typeof(u0), typeof(_tspan), typeof(p), typeof(noise),
            typeof(transition), typeof(observation), typeof(f),
            typeof(observables_noise), typeof(_observables), typeof(obs_syms),
            typeof(kwargs),
        }(
            f, transition, observation, observables_noise, _observables,
            u0, _tspan, p, noise, n_shocks, n_obs, obs_syms, kwargs
        )
    end
end
# just forwards to a iip = false case
function StateSpaceProblem(args...; kwargs...)
    return StateSpaceProblem{false}(args...; kwargs...)
end
