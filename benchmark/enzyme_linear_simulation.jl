# Enzyme AD benchmarks for Linear DirectIteration simulation (no observations/likelihood)
# Returns SIM_ENZYME BenchmarkGroup

using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, StateSpaceWorkspace

const SIM_ENZYME = BenchmarkGroup()
SIM_ENZYME["raw"] = BenchmarkGroup()
SIM_ENZYME["forward"] = BenchmarkGroup()
SIM_ENZYME["reverse"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

const p_sim_small = (; N = 5, M = 3, K = 2, T = 10)
const p_sim_large = (; N = 30, M = 10, K = 10, T = 100)

# =============================================================================
# Problem setup
# =============================================================================

function make_sim_benchmark(p; seed = 42)
    (; N, M, K, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    B = 0.1 * randn(N, K)
    C = randn(M, N)
    u0 = zeros(N)
    noise = [randn(K) for _ in 1:T]

    # Create problem and workspace (no observables, no observables_noise)
    prob = LinearStateSpaceProblem(A, B, u0, (0, T); C, noise)
    ws = init(prob, DirectIteration())
    sol_out = ws.output
    cache = ws.cache

    # Shadow copies for AD (all Duplicated)
    dprob = make_zero(prob)
    dsol_out = make_zero(sol_out)
    dcache = make_zero(cache)
    dA = make_zero(A)
    dB = make_zero(B)
    dC = make_zero(C)
    du0 = make_zero(u0)
    dnoise = [make_zero(noise[1]) for _ in 1:T]

    return (; A, B, C, u0, noise, prob, sol_out, cache,
        dprob, dsol_out, dcache, dA, dB, dC, du0, dnoise)
end

# =============================================================================
# Scalar wrapper for reverse mode (returns sum of terminal state)
# =============================================================================

function sim_scalar_bench!(A, B, C, u0, noise, prob, sol_out, cache)::Float64
    prob_new = remake(prob; A, B, C, u0, noise)
    ws = StateSpaceWorkspace(prob_new, DirectIteration(), sol_out, cache)
    return sum(solve!(ws).u[end])
end

# =============================================================================
# Forward wrapper (returns terminal state and observation)
# =============================================================================

function sim_forward_bench!(A, B, C, u0, noise, prob, sol_out, cache)
    prob_new = remake(prob; A, B, C, u0, noise)
    ws = StateSpaceWorkspace(prob_new, DirectIteration(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.z[end])
end

# =============================================================================
# Instantiate problems
# =============================================================================

const sim_s = make_sim_benchmark(p_sim_small)
const sim_l = make_sim_benchmark(p_sim_large)

# =============================================================================
# Raw benchmarks (primal solve through public API)
# =============================================================================

function raw_sim!(prob, sol_out, cache)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return sum(solve!(ws).u[end])
end

# Warmup
raw_sim!(sim_s.prob, sim_s.sol_out, sim_s.cache)
raw_sim!(sim_l.prob, sim_l.sol_out, sim_l.cache)

SIM_ENZYME["raw"]["small_mutable"] = @benchmarkable raw_sim!($(sim_s.prob), $(sim_s.sol_out), $(sim_s.cache))
SIM_ENZYME["raw"]["large_mutable"] = @benchmarkable raw_sim!($(sim_l.prob), $(sim_l.sol_out), $(sim_l.cache))

# =============================================================================
# Forward mode AD — perturb A[1,1], return terminal state and observation
# =============================================================================

function forward_sim_bench!(A, B, C, u0, noise, prob, sol_out, cache,
        dA, dB, dC, du0, dnoise, dprob, dsol_out, dcache)
    # Zero all shadows
    make_zero!(dprob)
    make_zero!(dsol_out)
    make_zero!(dcache)
    make_zero!(dA); make_zero!(dB); make_zero!(dC)
    make_zero!(du0)
    @inbounds for i in eachindex(dnoise)
        make_zero!(dnoise[i])
    end
    # Set perturbation direction
    dA[1, 1] = 1.0

    autodiff(Forward, sim_forward_bench!,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(prob, dprob), Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

# Warmup
forward_sim_bench!(
    copy(sim_s.A), copy(sim_s.B), copy(sim_s.C),
    copy(sim_s.u0), [copy(n) for n in sim_s.noise],
    sim_s.prob, sim_s.sol_out, sim_s.cache,
    sim_s.dA, sim_s.dB, sim_s.dC, sim_s.du0, sim_s.dnoise,
    sim_s.dprob, sim_s.dsol_out, sim_s.dcache)

SIM_ENZYME["forward"]["small_mutable"] = @benchmarkable forward_sim_bench!(
    $(copy(sim_s.A)), $(copy(sim_s.B)), $(copy(sim_s.C)),
    $(copy(sim_s.u0)), $([copy(n) for n in sim_s.noise]),
    $(sim_s.prob), $(sim_s.sol_out), $(sim_s.cache),
    $(sim_s.dA), $(sim_s.dB), $(sim_s.dC), $(sim_s.du0), $(sim_s.dnoise),
    $(sim_s.dprob), $(sim_s.dsol_out), $(sim_s.dcache))

# Warmup large
forward_sim_bench!(
    copy(sim_l.A), copy(sim_l.B), copy(sim_l.C),
    copy(sim_l.u0), [copy(n) for n in sim_l.noise],
    sim_l.prob, sim_l.sol_out, sim_l.cache,
    sim_l.dA, sim_l.dB, sim_l.dC, sim_l.du0, sim_l.dnoise,
    sim_l.dprob, sim_l.dsol_out, sim_l.dcache)

SIM_ENZYME["forward"]["large_mutable"] = @benchmarkable forward_sim_bench!(
    $(copy(sim_l.A)), $(copy(sim_l.B)), $(copy(sim_l.C)),
    $(copy(sim_l.u0)), $([copy(n) for n in sim_l.noise]),
    $(sim_l.prob), $(sim_l.sol_out), $(sim_l.cache),
    $(sim_l.dA), $(sim_l.dB), $(sim_l.dC), $(sim_l.du0), $(sim_l.dnoise),
    $(sim_l.dprob), $(sim_l.dsol_out), $(sim_l.dcache))

# =============================================================================
# Reverse mode AD — all Duplicated, scalar sum(u[end]) output
# =============================================================================

function reverse_sim_bench!(A, B, C, u0, noise, prob, sol_out, cache,
        dA, dB, dC, du0, dnoise, dprob, dsol_out, dcache)
    # Zero all shadows
    make_zero!(dprob)
    make_zero!(dsol_out)
    make_zero!(dcache)
    make_zero!(dA); make_zero!(dB); make_zero!(dC)
    make_zero!(du0)
    @inbounds for i in eachindex(dnoise)
        make_zero!(dnoise[i])
    end

    autodiff(Reverse, sim_scalar_bench!, Active,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(prob, dprob), Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

# Warmup
reverse_sim_bench!(
    copy(sim_s.A), copy(sim_s.B), copy(sim_s.C),
    copy(sim_s.u0), [copy(n) for n in sim_s.noise],
    sim_s.prob, sim_s.sol_out, sim_s.cache,
    sim_s.dA, sim_s.dB, sim_s.dC, sim_s.du0, sim_s.dnoise,
    sim_s.dprob, sim_s.dsol_out, sim_s.dcache)

SIM_ENZYME["reverse"]["small_mutable"] = @benchmarkable reverse_sim_bench!(
    $(copy(sim_s.A)), $(copy(sim_s.B)), $(copy(sim_s.C)),
    $(copy(sim_s.u0)), $([copy(n) for n in sim_s.noise]),
    $(sim_s.prob), $(sim_s.sol_out), $(sim_s.cache),
    $(sim_s.dA), $(sim_s.dB), $(sim_s.dC), $(sim_s.du0), $(sim_s.dnoise),
    $(sim_s.dprob), $(sim_s.dsol_out), $(sim_s.dcache))

# Warmup large
reverse_sim_bench!(
    copy(sim_l.A), copy(sim_l.B), copy(sim_l.C),
    copy(sim_l.u0), [copy(n) for n in sim_l.noise],
    sim_l.prob, sim_l.sol_out, sim_l.cache,
    sim_l.dA, sim_l.dB, sim_l.dC, sim_l.du0, sim_l.dnoise,
    sim_l.dprob, sim_l.dsol_out, sim_l.dcache)

SIM_ENZYME["reverse"]["large_mutable"] = @benchmarkable reverse_sim_bench!(
    $(copy(sim_l.A)), $(copy(sim_l.B)), $(copy(sim_l.C)),
    $(copy(sim_l.u0)), $([copy(n) for n in sim_l.noise]),
    $(sim_l.prob), $(sim_l.sol_out), $(sim_l.cache),
    $(sim_l.dA), $(sim_l.dB), $(sim_l.dC), $(sim_l.du0), $(sim_l.dnoise),
    $(sim_l.dprob), $(sim_l.dsol_out), $(sim_l.dcache))

# --- Edge cases: no noise, no observation equation (raw primal only) ---

SIM_ENZYME["raw"]["no_noise"] = let
    A = sim_s.A; C = sim_s.C; u0 = sim_s.u0
    prob = LinearStateSpaceProblem(A, nothing, u0, (0, p_sim_small.T); C)
    ws = init(prob, DirectIteration())
    @benchmarkable bench_nn!(ws_nn) setup=(ws_nn = $ws)
end

function bench_nn!(ws)
    solve!(ws)
    return nothing
end

SIM_ENZYME["raw"]["no_obs_eq"] = let
    A = sim_s.A; u0 = sim_s.u0
    prob = LinearStateSpaceProblem(A, nothing, u0, (0, p_sim_small.T))
    ws = init(prob, DirectIteration())
    @benchmarkable bench_nn!(ws_nn) setup=(ws_nn = $ws)
end

SIM_ENZYME
