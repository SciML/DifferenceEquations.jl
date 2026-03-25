# Quadratic state-space problem types
# Two variants: unpruned (quad on x) and pruned (quad on linear-part u_f)
# Union type for shared dispatch (cache allocation, noise matrix, etc.)

# --- Unpruned quadratic ---
# x[t+1] = A_0 + A_1 * x[t] + quad(A_2, x[t]) + B * w[t]
# z[t]   = C_0 + C_1 * x[t] + quad(C_2, x[t])

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
        noise = nothing, syms = nothing, obs_syms = nothing, kwargs...)
    f = ODEFunction{false}(
        (u, p, t) -> error("not implemented");
        sys = SymbolCache(syms))
    _tspan = promote_tspan(tspan)
    @assert round(_tspan[2] - _tspan[1]) - (_tspan[2] - _tspan[1]) ≈ 0.0
    return QuadraticStateSpaceProblem(
        f, A_0, A_1, A_2, B, C_0, C_1, C_2,
        observables_noise, observables, u0, _tspan, p, noise, obs_syms, kwargs)
end

# --- Pruned quadratic ---
# u_f[t+1] = A_1 * u_f[t] + B * w[t]
# x[t+1]   = A_0 + A_1 * x[t] + quad(A_2, u_f[t]) + B * w[t]
# z[t]     = C_0 + C_1 * x[t] + quad(C_2, u_f[t])

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
        noise = nothing, syms = nothing, obs_syms = nothing, kwargs...)
    f = ODEFunction{false}(
        (u, p, t) -> error("not implemented");
        sys = SymbolCache(syms))
    _tspan = promote_tspan(tspan)
    @assert round(_tspan[2] - _tspan[1]) - (_tspan[2] - _tspan[1]) ≈ 0.0
    return PrunedQuadraticStateSpaceProblem(
        f, A_0, A_1, A_2, B, C_0, C_1, C_2,
        observables_noise, observables, u0, _tspan, p, noise, obs_syms, kwargs)
end

# Union for shared dispatch (cache allocation, noise matrix, etc.)
const AnyQuadraticProblem = Union{QuadraticStateSpaceProblem, PrunedQuadraticStateSpaceProblem}
