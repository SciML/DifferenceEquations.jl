# Quadratic state-space problem types
# Two variants: unpruned (quad on x) and pruned (quad on linear-part u_f)
# Union type for shared dispatch (cache allocation, noise matrix, etc.)

# --- Unpruned quadratic ---
# x[t+1] = A_0 + A_1 * x[t] + quad(A_2, x[t]) + B * w[t]
# z[t]   = C_0 + C_1 * x[t] + quad(C_2, x[t])

"""
    QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, tspan[, p]; kwargs...)

Define a second-order (quadratic) state-space model:

```math
u_{n+1} = A_0 + A_1 \\, u_n + u_n^\\top A_2 \\, u_n + B \\, w_{n+1}
```

with optional observation equation
``z_n = C_0 + C_1 \\, u_n + u_n^\\top C_2 \\, u_n + v_n``.

# Arguments
- `A_0`: Constant drift vector (length n).
- `A_1`: Linear transition matrix (n×n).
- `A_2`: Quadratic transition tensor (n×n×n). Entry `A_2[i,:,:]` gives the matrix
  for the `i`-th element of the quadratic term.
- `B`: Noise input matrix (n×k), or `nothing`.
- `u0`: Initial state vector.
- `tspan`: Time span as `(t0, t_end)` with integer distance.
- `p`: Parameters passed through the SciML problem interface (default:
  `NullParameters()`).

# Keyword Arguments
- `C_0`: Constant observation term, or `nothing` when observations are disabled.
- `C_1`: Linear observation matrix, or `nothing` when observations are disabled.
- `C_2`: Quadratic observation tensor, or `nothing` when observations are disabled.
- `observables_noise`: Observation covariance matrix, or `nothing`.
- `observables`: Observed data used for likelihood calculations, or `nothing`.
- `noise`: Fixed process-noise sequence, or `nothing` for generated noise.
- `syms`: State variable names for symbolic indexing, or `nothing`.
- `obs_syms`: Observation variable names for symbolic indexing, or `nothing`.
- `kwargs...`: Additional constructor keywords retained in the problem.

# References
- Andreasen, Fernandez-Villaverde, and Rubio-Ramirez (2017),
  "The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications."

See also: [`PrunedQuadraticStateSpaceProblem`](@ref), [`LinearStateSpaceProblem`](@ref).

# Fields

- `A_0`, `A_1`, `A_2`: Constant, linear, and quadratic transition coefficients.
- `f`: Internal `SciMLBase.ODEFunction` bridge used for SciML interfaces and
  symbolic indexing.
- `B`: Noise input matrix or `nothing`.
- `C_0`, `C_1`, `C_2`: Optional observation coefficients.
- `u0`: Initial state.
- `tspan`: Integer-step time span.
- `p`: User parameters.
- `observables_noise`, `observables`, `noise`: Observation and simulation data.
- `syms`, `obs_syms`: Optional state and observation names for symbolic indexing.
- `kwargs`: Additional constructor keyword arguments retained for remaking.

# Returns

- `QuadraticStateSpaceProblem`: A quadratic discrete-time state-space problem.

# Throws

- `ArgumentError`: If `tspan` does not have an integer distance.

# Examples

```jldoctest
julia> using DifferenceEquations

julia> prob = QuadraticStateSpaceProblem([0.0], [1.0;;], zeros(1, 1, 1), nothing, [1.0], (0, 2));

julia> length(solve(prob).u)
3
```
"""
@concrete struct QuadraticStateSpaceProblem <: AbstractStateSpaceProblem
    f           # ODEFunction (SciML interface/syms only)
    A_0         # Constant drift vector
    A_1         # Linear transition matrix
    A_2         # Quadratic transition tensor (N, N, N)
    B           # Noise input matrix (or nothing)
    C_0         # Observation constant (or nothing)
    C_1         # Observation linear matrix (or nothing)
    C_2         # Observation quadratic tensor (or nothing)
    observables_noise
    observables
    u0
    tspan
    p
    noise
    obs_syms
    kwargs
end

function QuadraticStateSpaceProblem(
        A_0, A_1, A_2, B, u0, tspan, p = NullParameters();
        C_0 = nothing, C_1 = nothing, C_2 = nothing,
        observables_noise = nothing, observables = nothing,
        noise = nothing, syms = nothing, obs_syms = nothing, kwargs...
    )
    f = ODEFunction{false}(
        (u, p, t) -> error("not implemented");
        sys = SymbolCache(syms)
    )
    _tspan = promote_tspan(tspan)
    _dt = _tspan[2] - _tspan[1]
    isinteger(_dt) || throw(ArgumentError("tspan must have integer distance, got $_dt"))
    return QuadraticStateSpaceProblem(
        f, A_0, A_1, A_2, B, C_0, C_1, C_2,
        observables_noise, observables, u0, _tspan, p, noise, obs_syms, kwargs
    )
end

# --- Pruned quadratic ---
# u_f[t+1] = A_1 * u_f[t] + B * w[t]
# x[t+1]   = A_0 + A_1 * x[t] + quad(A_2, u_f[t]) + B * w[t]
# z[t]     = C_0 + C_1 * x[t] + quad(C_2, u_f[t])

"""
    PrunedQuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, tspan[, p]; kwargs...)

Define a pruned second-order state-space model. Unlike [`QuadraticStateSpaceProblem`](@ref),
the quadratic terms operate on a separate linear-part state ``u_f`` rather than the full state:

```math
u_f^{n+1} = A_1 \\, u_f^n + B \\, w_{n+1}
```
```math
u_{n+1} = A_0 + A_1 \\, u_n + (u_f^n)^\\top A_2 \\, u_f^n + B \\, w_{n+1}
```

The observation equation similarly uses ``u_f``:
``z_n = C_0 + C_1 \\, u_n + (u_f^n)^\\top C_2 \\, u_f^n + v_n``.

This pruning approach prevents explosive dynamics in higher-order perturbation solutions.

# Arguments

- `A_0`: Constant drift vector.
- `A_1`: Linear transition matrix used by both the full and linear-part states.
- `A_2`: Quadratic transition tensor applied to the linear-part state.
- `B`: Noise input matrix, or `nothing`.
- `u0`: Initial full state and initial linear-part state.
- `tspan`: Integer-step time span.
- `p`: Parameters passed through the SciML problem interface (default:
  `NullParameters()`).

# Keyword Arguments

- `C_0`: Constant observation term, or `nothing`.
- `C_1`: Linear observation matrix, or `nothing`.
- `C_2`: Quadratic observation tensor, or `nothing`.
- `observables_noise`: Observation covariance matrix, or `nothing`.
- `observables`: Observed data used for likelihood calculations, or `nothing`.
- `noise`: Fixed process-noise sequence, or `nothing` for generated noise.
- `syms`: State variable names for symbolic indexing, or `nothing`.
- `obs_syms`: Observation variable names for symbolic indexing, or `nothing`.
- `kwargs...`: Additional constructor keywords retained in the problem.

# References
- Andreasen, Fernandez-Villaverde, and Rubio-Ramirez (2017),
  "The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications."

See also: [`QuadraticStateSpaceProblem`](@ref).

# Fields

- `A_0`, `A_1`, `A_2`: Constant, linear, and quadratic transition coefficients.
- `f`: Internal `SciMLBase.ODEFunction` bridge used for SciML interfaces and
  symbolic indexing.
- `B`: Noise input matrix or `nothing`.
- `C_0`, `C_1`, `C_2`: Optional observation coefficients.
- `u0`: Initial state and the initial value of the linear component `u_f`.
- `tspan`: Integer-step time span.
- `p`: User parameters.
- `observables_noise`, `observables`, `noise`: Observation and simulation data.
- `syms`, `obs_syms`: Optional state and observation names for symbolic indexing.
- `kwargs`: Additional constructor keyword arguments retained for remaking.

# Returns

- `PrunedQuadraticStateSpaceProblem`: A pruned quadratic state-space problem.

# Throws

- `ArgumentError`: If `tspan` does not have an integer distance.

# Examples

```jldoctest
julia> using DifferenceEquations

julia> prob = PrunedQuadraticStateSpaceProblem([0.0], [1.0;;], zeros(1, 1, 1), nothing, [1.0], (0, 2));

julia> length(solve(prob).u)
3
```
"""
@concrete struct PrunedQuadraticStateSpaceProblem <: AbstractStateSpaceProblem
    f           # ODEFunction (SciML interface/syms only)
    A_0         # Constant drift vector
    A_1         # Linear transition matrix
    A_2         # Quadratic transition tensor (N, N, N)
    B           # Noise input matrix (or nothing)
    C_0         # Observation constant (or nothing)
    C_1         # Observation linear matrix (or nothing)
    C_2         # Observation quadratic tensor (or nothing)
    observables_noise
    observables
    u0
    tspan
    p
    noise
    obs_syms
    kwargs
end

function PrunedQuadraticStateSpaceProblem(
        A_0, A_1, A_2, B, u0, tspan, p = NullParameters();
        C_0 = nothing, C_1 = nothing, C_2 = nothing,
        observables_noise = nothing, observables = nothing,
        noise = nothing, syms = nothing, obs_syms = nothing, kwargs...
    )
    f = ODEFunction{false}(
        (u, p, t) -> error("not implemented");
        sys = SymbolCache(syms)
    )
    _tspan = promote_tspan(tspan)
    _dt = _tspan[2] - _tspan[1]
    isinteger(_dt) || throw(ArgumentError("tspan must have integer distance, got $_dt"))
    return PrunedQuadraticStateSpaceProblem(
        f, A_0, A_1, A_2, B, C_0, C_1, C_2,
        observables_noise, observables, u0, _tspan, p, noise, obs_syms, kwargs
    )
end

# Union for shared dispatch (cache allocation, noise matrix, etc.)
const AnyQuadraticProblem = Union{QuadraticStateSpaceProblem, PrunedQuadraticStateSpaceProblem}
