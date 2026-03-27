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

# Positional Arguments
- `A_0`: Constant drift vector (length n).
- `A_1`: Linear transition matrix (n×n).
- `A_2`: Quadratic transition tensor (n×n×n). Entry `A_2[i,:,:]` gives the matrix
  for the `i`-th element of the quadratic term.
- `B`: Noise input matrix (n×k), or `nothing`.
- `u0`: Initial state vector.
- `tspan`: Time span as `(t0, t_end)`.

# Keyword Arguments
- `C_0`, `C_1`, `C_2`: Observation equation coefficients (analogous to `A_0`, `A_1`, `A_2`).
- `observables_noise`, `observables`, `noise`, `syms`, `obs_syms`: Same as
  [`LinearStateSpaceProblem`](@ref).

# References
- Andreasen, Fernandez-Villaverde, and Rubio-Ramirez (2017),
  "The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications."

See also: [`PrunedQuadraticStateSpaceProblem`](@ref), [`LinearStateSpaceProblem`](@ref).
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
Arguments are identical to [`QuadraticStateSpaceProblem`](@ref).

# References
- Andreasen, Fernandez-Villaverde, and Rubio-Ramirez (2017),
  "The Pruned State-Space System for Non-Linear DSGE Models: Theory and Empirical Applications."

See also: [`QuadraticStateSpaceProblem`](@ref).
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
