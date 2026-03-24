# Enzyme AD benchmarks for generic StateSpaceProblem with quadratic callbacks
# Returns QUAD_ENZYME BenchmarkGroup
#
# Two modes:
#   "simulation" — no observables, forward returns sol_out.u[end]
#   "likelihood" — with observables + obs_noise, forward returns (u[end], z[end])
#
# NOTE: The quadratic callbacks capture mutable state (u_f).  Enzyme may or may
# not handle these closures correctly.  Raw primal benchmarks are always active;
# forward/reverse AD benchmarks are commented out until verified.

using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, StateSpaceWorkspace

const QUAD_ENZYME = BenchmarkGroup()
QUAD_ENZYME["simulation"] = BenchmarkGroup()
QUAD_ENZYME["simulation"]["raw"] = BenchmarkGroup()
QUAD_ENZYME["simulation"]["forward"] = BenchmarkGroup()
QUAD_ENZYME["simulation"]["reverse"] = BenchmarkGroup()
QUAD_ENZYME["likelihood"] = BenchmarkGroup()
QUAD_ENZYME["likelihood"]["raw"] = BenchmarkGroup()
QUAD_ENZYME["likelihood"]["forward"] = BenchmarkGroup()
QUAD_ENZYME["likelihood"]["reverse"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

const p_quad_small = (; N = 2, K = 1, M = 2, T = 10)
const p_quad_large = (; N = 10, K = 3, M = 6, T = 50)

# =============================================================================
# Random matrix generation (self-contained, no CSV data)
# =============================================================================

function make_quadratic_matrices(N, K, M; seed = 42)
    Random.seed!(seed)
    A_1 = 0.3 * randn(N, N) / N  # scale to keep stable
    A_1 = 0.5 * A_1 / maximum(abs.(eigvals(A_1)))
    A_0 = 0.001 * randn(N)
    A_2 = 0.01 * randn(N, N, N) / N
    B = 0.1 * randn(N, K)
    C_0 = 0.001 * randn(M)
    C_1 = randn(M, N)
    C_2 = 0.01 * randn(M, N, N) / N
    return (; A_0, A_1, A_2, B, C_0, C_1, C_2)
end

# =============================================================================
# Quadratic callbacks (copied from test/direct_iteration.jl)
# =============================================================================

function make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
    n_x = length(u0)
    n_obs = length(C_0)
    u_f = copy(u0)          # tracks linear-part state, initialized to u0
    u_f_new = similar(u0)   # workspace for updating u_f

    function f!!(x_next, x, w, p, t)
        # Compute new linear-part: u_f_new = A_1 * u_f + B * w
        mul!(u_f_new, A_1, u_f)
        mul!(u_f_new, B, w, 1.0, 1.0)

        # Full transition: x_next = A_0 + A_1 * x + quad(A_2, u_f) + B * w
        copyto!(x_next, A_0)
        mul!(x_next, A_1, x, 1.0, 1.0)
        @inbounds for i in 1:n_x
            x_next[i] += dot(u_f, view(A_2, i, :, :), u_f)
        end
        mul!(x_next, B, w, 1.0, 1.0)

        # Advance u_f for next step
        copyto!(u_f, u_f_new)

        return x_next
    end

    function g!!(y, x, p, t)
        # y = C_0 + C_1 * x + quad(C_2, u_f)
        copyto!(y, C_0)
        mul!(y, C_1, x, 1.0, 1.0)
        @inbounds for i in 1:n_obs
            y[i] += dot(u_f, view(C_2, i, :, :), u_f)
        end
        return y
    end

    return f!!, g!!
end

# =============================================================================
# Problem constructors
# =============================================================================

function make_quad_prob(mats, u0, T, noise; observables = nothing, observables_noise = nothing)
    f!!, g!! = make_quadratic_callbacks(mats.A_0, mats.A_1, mats.A_2, mats.B,
        mats.C_0, mats.C_1, mats.C_2, u0)
    return StateSpaceProblem(f!!, g!!, u0, (0, T), nothing;
        n_shocks = size(mats.B, 2), n_obs = length(mats.C_0),
        noise, observables, observables_noise)
end

function make_quad_benchmark(p; seed = 42)
    (; N, K, M, T) = p
    mats = make_quadratic_matrices(N, K, M; seed)
    u0 = zeros(N)
    Random.seed!(seed + 1)
    noise = [randn(K) for _ in 1:T]

    # --- Simulation problem (no observables) ---
    prob_sim = make_quad_prob(mats, u0, T, noise)
    ws_sim = init(prob_sim, DirectIteration())
    sol_out_sim = ws_sim.output
    cache_sim = ws_sim.cache

    # --- Likelihood problem (observables + obs_noise) ---
    # Generate synthetic observations from a simulation run
    sim_sol = solve(make_quad_prob(mats, u0, T, noise))
    Random.seed!(seed + 2)
    H = 0.1 * randn(M, M)
    R = H * H'
    obs_noise_diag = diag(R)
    y = [sim_sol.z[t + 1] + H * randn(M) for t in 1:T]

    prob_lik = make_quad_prob(mats, u0, T, noise;
        observables = y, observables_noise = obs_noise_diag)
    ws_lik = init(prob_lik, DirectIteration())
    sol_out_lik = ws_lik.output
    cache_lik = ws_lik.cache

    # Shadow copies for AD (all Duplicated)
    dA_0 = make_zero(mats.A_0)
    dA_1 = make_zero(mats.A_1)
    dA_2 = make_zero(mats.A_2)
    dB = make_zero(mats.B)
    dC_0 = make_zero(mats.C_0)
    dC_1 = make_zero(mats.C_1)
    dC_2 = make_zero(mats.C_2)
    du0 = make_zero(u0)
    dnoise = [make_zero(noise[1]) for _ in 1:T]

    # Simulation shadows
    dprob_sim = make_zero(prob_sim)
    dsol_out_sim = make_zero(sol_out_sim)
    dcache_sim = make_zero(cache_sim)

    # Likelihood shadows
    dprob_lik = make_zero(prob_lik)
    dsol_out_lik = make_zero(sol_out_lik)
    dcache_lik = make_zero(cache_lik)
    dH = make_zero(H)
    dy = [make_zero(y[1]) for _ in 1:T]

    return (; mats, u0, noise, y, H, R, obs_noise_diag,
        prob_sim, sol_out_sim, cache_sim,
        prob_lik, sol_out_lik, cache_lik,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise,
        dprob_sim, dsol_out_sim, dcache_sim,
        dprob_lik, dsol_out_lik, dcache_lik, dH, dy)
end

# =============================================================================
# Instantiate problems
# =============================================================================

const quad_s = make_quad_benchmark(p_quad_small)
const quad_l = make_quad_benchmark(p_quad_large)

# =============================================================================
# Raw benchmarks — simulation (no observables)
# =============================================================================

function raw_quad_sim!(prob, sol_out, cache)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return sol_out.u[end]
end

# Warmup
raw_quad_sim!(quad_s.prob_sim, quad_s.sol_out_sim, quad_s.cache_sim)
raw_quad_sim!(quad_l.prob_sim, quad_l.sol_out_sim, quad_l.cache_sim)

QUAD_ENZYME["simulation"]["raw"]["small_mutable"] = @benchmarkable raw_quad_sim!(
    $(quad_s.prob_sim), $(quad_s.sol_out_sim), $(quad_s.cache_sim))
QUAD_ENZYME["simulation"]["raw"]["large_mutable"] = @benchmarkable raw_quad_sim!(
    $(quad_l.prob_sim), $(quad_l.sol_out_sim), $(quad_l.cache_sim))

# =============================================================================
# Raw benchmarks — likelihood (with observables + obs_noise)
# =============================================================================

function raw_quad_lik!(prob, sol_out, cache)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return solve!(ws).logpdf
end

# Warmup
raw_quad_lik!(quad_s.prob_lik, quad_s.sol_out_lik, quad_s.cache_lik)
raw_quad_lik!(quad_l.prob_lik, quad_l.sol_out_lik, quad_l.cache_lik)

QUAD_ENZYME["likelihood"]["raw"]["small_mutable"] = @benchmarkable raw_quad_lik!(
    $(quad_s.prob_lik), $(quad_s.sol_out_lik), $(quad_s.cache_lik))
QUAD_ENZYME["likelihood"]["raw"]["large_mutable"] = @benchmarkable raw_quad_lik!(
    $(quad_l.prob_lik), $(quad_l.sol_out_lik), $(quad_l.cache_lik))

# =============================================================================
# Forward / Reverse AD wrappers
#
# NOTE: These are commented out pending verification that Enzyme can
# differentiate through the mutable closures created by make_quadratic_callbacks.
# The callbacks capture `u_f` (mutable Vector) and mutate it each timestep.
# To enable, uncomment the blocks below and add warmup + @benchmarkable calls.
# =============================================================================

# --- Simulation forward wrapper ---
# function quad_sim_forward_bench!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise,
#         prob, sol_out, cache)
#     f!!, g!! = make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
#     prob_new = StateSpaceProblem(f!!, g!!, u0, prob.tspan, nothing;
#         n_shocks = size(B, 2), n_obs = length(C_0), noise)
#     ws = StateSpaceWorkspace(prob_new, DirectIteration(), sol_out, cache)
#     solve!(ws)
#     return sol_out.u[end]
# end

# --- Simulation reverse wrapper (scalar output: sum of final state) ---
# function quad_sim_reverse_bench!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise,
#         prob, sol_out, cache)
#     f!!, g!! = make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
#     prob_new = StateSpaceProblem(f!!, g!!, u0, prob.tspan, nothing;
#         n_shocks = size(B, 2), n_obs = length(C_0), noise)
#     ws = StateSpaceWorkspace(prob_new, DirectIteration(), sol_out, cache)
#     solve!(ws)
#     return sum(sol_out.u[end])
# end

# --- Likelihood forward wrapper ---
# function quad_lik_forward_bench!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, y, H,
#         prob, sol_out, cache)
#     R = H * H'
#     obs_noise_diag = diag(R)
#     f!!, g!! = make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
#     prob_new = StateSpaceProblem(f!!, g!!, u0, prob.tspan, nothing;
#         n_shocks = size(B, 2), n_obs = length(C_0),
#         noise, observables = y, observables_noise = obs_noise_diag)
#     ws = StateSpaceWorkspace(prob_new, DirectIteration(), sol_out, cache)
#     solve!(ws)
#     return (sol_out.u[end], sol_out.z[end])
# end

# --- Likelihood reverse wrapper (scalar output: logpdf) ---
# function quad_lik_reverse_bench!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, y, H,
#         prob, sol_out, cache)
#     R = H * H'
#     obs_noise_diag = diag(R)
#     f!!, g!! = make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
#     prob_new = StateSpaceProblem(f!!, g!!, u0, prob.tspan, nothing;
#         n_shocks = size(B, 2), n_obs = length(C_0),
#         noise, observables = y, observables_noise = obs_noise_diag)
#     ws = StateSpaceWorkspace(prob_new, DirectIteration(), sol_out, cache)
#     return solve!(ws).logpdf
# end

# =============================================================================
# Forward mode AD — simulation (commented out, pending Enzyme closure support)
# =============================================================================

# function forward_quad_sim_bench!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise,
#         prob, sol_out, cache,
#         dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise,
#         dprob, dsol_out, dcache)
#     # Zero all shadows
#     make_zero!(dprob); make_zero!(dsol_out); make_zero!(dcache)
#     make_zero!(dA_0); make_zero!(dA_1); make_zero!(dA_2); make_zero!(dB)
#     make_zero!(dC_0); make_zero!(dC_1); make_zero!(dC_2); make_zero!(du0)
#     @inbounds for i in eachindex(dnoise)
#         make_zero!(dnoise[i])
#     end
#     # Perturb A_1[1,1]
#     dA_1[1, 1] = 1.0
#
#     autodiff(Forward, quad_sim_forward_bench!,
#         Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
#         Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
#         Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
#         Duplicated(prob, dprob), Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
#     return nothing
# end

# =============================================================================
# Reverse mode AD — simulation (commented out, pending Enzyme closure support)
# =============================================================================

# function reverse_quad_sim_bench!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise,
#         prob, sol_out, cache,
#         dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise,
#         dprob, dsol_out, dcache)
#     # Zero all shadows
#     make_zero!(dprob); make_zero!(dsol_out); make_zero!(dcache)
#     make_zero!(dA_0); make_zero!(dA_1); make_zero!(dA_2); make_zero!(dB)
#     make_zero!(dC_0); make_zero!(dC_1); make_zero!(dC_2); make_zero!(du0)
#     @inbounds for i in eachindex(dnoise)
#         make_zero!(dnoise[i])
#     end
#
#     autodiff(Reverse, quad_sim_reverse_bench!, Active,
#         Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
#         Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
#         Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
#         Duplicated(prob, dprob), Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
#     return nothing
# end

# =============================================================================
# Forward mode AD — likelihood (commented out, pending Enzyme closure support)
# =============================================================================

# function forward_quad_lik_bench!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, y, H,
#         prob, sol_out, cache,
#         dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dy, dH,
#         dprob, dsol_out, dcache)
#     # Zero all shadows
#     make_zero!(dprob); make_zero!(dsol_out); make_zero!(dcache)
#     make_zero!(dA_0); make_zero!(dA_1); make_zero!(dA_2); make_zero!(dB)
#     make_zero!(dC_0); make_zero!(dC_1); make_zero!(dC_2); make_zero!(du0)
#     make_zero!(dH)
#     @inbounds for i in eachindex(dnoise)
#         make_zero!(dnoise[i])
#     end
#     @inbounds for i in eachindex(dy)
#         make_zero!(dy[i])
#     end
#     # Perturb A_1[1,1]
#     dA_1[1, 1] = 1.0
#
#     autodiff(Forward, quad_lik_forward_bench!,
#         Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
#         Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
#         Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
#         Duplicated(y, dy), Duplicated(H, dH),
#         Duplicated(prob, dprob), Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
#     return nothing
# end

# =============================================================================
# Reverse mode AD — likelihood (commented out, pending Enzyme closure support)
# =============================================================================

# function reverse_quad_lik_bench!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, y, H,
#         prob, sol_out, cache,
#         dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dy, dH,
#         dprob, dsol_out, dcache)
#     # Zero all shadows
#     make_zero!(dprob); make_zero!(dsol_out); make_zero!(dcache)
#     make_zero!(dA_0); make_zero!(dA_1); make_zero!(dA_2); make_zero!(dB)
#     make_zero!(dC_0); make_zero!(dC_1); make_zero!(dC_2); make_zero!(du0)
#     make_zero!(dH)
#     @inbounds for i in eachindex(dnoise)
#         make_zero!(dnoise[i])
#     end
#     @inbounds for i in eachindex(dy)
#         make_zero!(dy[i])
#     end
#
#     autodiff(Reverse, quad_lik_reverse_bench!, Active,
#         Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
#         Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
#         Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
#         Duplicated(y, dy), Duplicated(H, dH),
#         Duplicated(prob, dprob), Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
#     return nothing
# end

# =============================================================================
# Forward/reverse @benchmarkable registrations (commented out)
# =============================================================================

# Uncomment these once the AD wrappers above are verified to work with Enzyme.
# Follow the same warmup + @benchmarkable pattern as enzyme_direct_iteration.jl.

# --- Simulation forward ---
# forward_quad_sim_bench!(...)  # warmup small
# QUAD_ENZYME["simulation"]["forward"]["small_mutable"] = @benchmarkable forward_quad_sim_bench!(...)
# forward_quad_sim_bench!(...)  # warmup large
# QUAD_ENZYME["simulation"]["forward"]["large_mutable"] = @benchmarkable forward_quad_sim_bench!(...)

# --- Simulation reverse ---
# reverse_quad_sim_bench!(...)  # warmup small
# QUAD_ENZYME["simulation"]["reverse"]["small_mutable"] = @benchmarkable reverse_quad_sim_bench!(...)
# reverse_quad_sim_bench!(...)  # warmup large
# QUAD_ENZYME["simulation"]["reverse"]["large_mutable"] = @benchmarkable reverse_quad_sim_bench!(...)

# --- Likelihood forward ---
# forward_quad_lik_bench!(...)  # warmup small
# QUAD_ENZYME["likelihood"]["forward"]["small_mutable"] = @benchmarkable forward_quad_lik_bench!(...)
# forward_quad_lik_bench!(...)  # warmup large
# QUAD_ENZYME["likelihood"]["forward"]["large_mutable"] = @benchmarkable forward_quad_lik_bench!(...)

# --- Likelihood reverse ---
# reverse_quad_lik_bench!(...)  # warmup small
# QUAD_ENZYME["likelihood"]["reverse"]["small_mutable"] = @benchmarkable reverse_quad_lik_bench!(...)
# reverse_quad_lik_bench!(...)  # warmup large
# QUAD_ENZYME["likelihood"]["reverse"]["large_mutable"] = @benchmarkable reverse_quad_lik_bench!(...)

QUAD_ENZYME
