# Enzyme AD benchmarks for QuadraticStateSpaceProblem / PrunedQuadraticStateSpaceProblem
# Two sub-groups: "unpruned" and "pruned", each with "raw", "forward", "reverse" × small/large
# Returns QUAD_ENZYME BenchmarkGroup

using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, StateSpaceWorkspace, fill_zero!!

const QUAD_ENZYME = BenchmarkGroup()
QUAD_ENZYME["unpruned"] = BenchmarkGroup()
QUAD_ENZYME["unpruned"]["raw"] = BenchmarkGroup()
QUAD_ENZYME["unpruned"]["forward"] = BenchmarkGroup()
QUAD_ENZYME["unpruned"]["reverse"] = BenchmarkGroup()
QUAD_ENZYME["pruned"] = BenchmarkGroup()
QUAD_ENZYME["pruned"]["raw"] = BenchmarkGroup()
QUAD_ENZYME["pruned"]["forward"] = BenchmarkGroup()
QUAD_ENZYME["pruned"]["reverse"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

const p_quad_small = (; N = 2, K = 1, M = 2, T = 10)
const p_quad_large = (; N = 10, K = 3, M = 6, T = 50)

# =============================================================================
# Problem setup
# =============================================================================

function make_quad_benchmark(; N, K, M, T, seed = 42, pruned = false)
    Random.seed!(seed)
    A_1_raw = randn(N, N)
    A_1 = 0.5 * A_1_raw / maximum(abs.(eigvals(A_1_raw)))
    A_0 = 0.001 * randn(N)
    A_2 = 0.01 * randn(N, N, N) / N
    B = 0.1 * randn(N, K)
    C_0 = 0.001 * randn(M)
    C_1 = randn(M, N)
    C_2 = 0.01 * randn(M, N, N) / N
    u0 = zeros(N)
    noise = [randn(K) for _ in 1:T]

    ProbType = pruned ? PrunedQuadraticStateSpaceProblem : QuadraticStateSpaceProblem
    prob = ProbType(A_0, A_1, A_2, B, u0, (0, T); C_0, C_1, C_2, noise)
    ws = init(prob, DirectIteration())

    # Shadows for AD (no dprob — prob constructed inside wrapper)
    dA_0 = make_zero(A_0); dA_1 = make_zero(A_1); dA_2 = make_zero(A_2)
    dB = make_zero(B); dC_0 = make_zero(C_0); dC_1 = make_zero(C_1); dC_2 = make_zero(C_2)
    du0 = make_zero(u0); dnoise = [make_zero(noise[1]) for _ in 1:T]
    dsol = make_zero(ws.output); dcache = make_zero(ws.cache)

    return (; A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, prob,
        sol = ws.output, cache = ws.cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dsol, dcache)
end

# =============================================================================
# Inner wrappers — construct prob inside (correct Enzyme pattern)
# =============================================================================

# --- Unpruned ---

function quad_fwd!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache)
    prob = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.z[end])
end

function quad_rev!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache)::Float64
    prob = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return sum(solve!(ws).u[end])
end

# --- Pruned ---

function pruned_quad_fwd!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache)
    prob = PrunedQuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.z[end])
end

function pruned_quad_rev!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache)::Float64
    prob = PrunedQuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return sum(solve!(ws).u[end])
end

# =============================================================================
# Outer bench functions — zero shadows, call autodiff
# =============================================================================

function forward_quad!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); make_zero!(dA_2)
    dB = fill_zero!!(dB); dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1)
    make_zero!(dC_2); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end
    dA_1[1, 1] = 1.0

    autodiff(Forward, quad_fwd!,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

function reverse_quad!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); make_zero!(dA_2)
    dB = fill_zero!!(dB); dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1)
    make_zero!(dC_2); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end

    autodiff(Reverse, quad_rev!, Active,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

function forward_pruned_quad!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); make_zero!(dA_2)
    dB = fill_zero!!(dB); dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1)
    make_zero!(dC_2); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end
    dA_1[1, 1] = 1.0

    autodiff(Forward, pruned_quad_fwd!,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

function reverse_pruned_quad!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); make_zero!(dA_2)
    dB = fill_zero!!(dB); dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1)
    make_zero!(dC_2); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end

    autodiff(Reverse, pruned_quad_rev!, Active,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

# =============================================================================
# Instantiate problems
# =============================================================================

const quad_us = make_quad_benchmark(; p_quad_small..., pruned = false)
const quad_ul = make_quad_benchmark(; p_quad_large..., pruned = false)
const quad_ps = make_quad_benchmark(; p_quad_small..., pruned = true)
const quad_pl = make_quad_benchmark(; p_quad_large..., pruned = true)

# =============================================================================
# Raw benchmarks (primal solve through public API)
# =============================================================================

function raw_quad!(prob, sol_out, cache)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return nothing
end

# Warmup
raw_quad!(quad_us.prob, quad_us.sol, quad_us.cache)
raw_quad!(quad_ul.prob, quad_ul.sol, quad_ul.cache)
raw_quad!(quad_ps.prob, quad_ps.sol, quad_ps.cache)
raw_quad!(quad_pl.prob, quad_pl.sol, quad_pl.cache)

QUAD_ENZYME["unpruned"]["raw"]["small_mutable"] = @benchmarkable raw_quad!(
    $(quad_us.prob), $(quad_us.sol), $(quad_us.cache))
QUAD_ENZYME["unpruned"]["raw"]["large_mutable"] = @benchmarkable raw_quad!(
    $(quad_ul.prob), $(quad_ul.sol), $(quad_ul.cache))
QUAD_ENZYME["pruned"]["raw"]["small_mutable"] = @benchmarkable raw_quad!(
    $(quad_ps.prob), $(quad_ps.sol), $(quad_ps.cache))
QUAD_ENZYME["pruned"]["raw"]["large_mutable"] = @benchmarkable raw_quad!(
    $(quad_pl.prob), $(quad_pl.sol), $(quad_pl.cache))

# =============================================================================
# Forward mode AD — unpruned
# =============================================================================

# Warmup small
forward_quad!(
    copy(quad_us.A_0), copy(quad_us.A_1), copy(quad_us.A_2),
    copy(quad_us.B), copy(quad_us.C_0), copy(quad_us.C_1), copy(quad_us.C_2),
    copy(quad_us.u0), [copy(n) for n in quad_us.noise],
    quad_us.sol, quad_us.cache,
    quad_us.dA_0, quad_us.dA_1, quad_us.dA_2, quad_us.dB,
    quad_us.dC_0, quad_us.dC_1, quad_us.dC_2, quad_us.du0, quad_us.dnoise,
    quad_us.dsol, quad_us.dcache)

QUAD_ENZYME["unpruned"]["forward"]["small_mutable"] = @benchmarkable forward_quad!(
    $(copy(quad_us.A_0)), $(copy(quad_us.A_1)), $(copy(quad_us.A_2)),
    $(copy(quad_us.B)), $(copy(quad_us.C_0)), $(copy(quad_us.C_1)), $(copy(quad_us.C_2)),
    $(copy(quad_us.u0)), $([copy(n) for n in quad_us.noise]),
    $(quad_us.sol), $(quad_us.cache),
    $(quad_us.dA_0), $(quad_us.dA_1), $(quad_us.dA_2), $(quad_us.dB),
    $(quad_us.dC_0), $(quad_us.dC_1), $(quad_us.dC_2), $(quad_us.du0), $(quad_us.dnoise),
    $(quad_us.dsol), $(quad_us.dcache))

# Warmup large
forward_quad!(
    copy(quad_ul.A_0), copy(quad_ul.A_1), copy(quad_ul.A_2),
    copy(quad_ul.B), copy(quad_ul.C_0), copy(quad_ul.C_1), copy(quad_ul.C_2),
    copy(quad_ul.u0), [copy(n) for n in quad_ul.noise],
    quad_ul.sol, quad_ul.cache,
    quad_ul.dA_0, quad_ul.dA_1, quad_ul.dA_2, quad_ul.dB,
    quad_ul.dC_0, quad_ul.dC_1, quad_ul.dC_2, quad_ul.du0, quad_ul.dnoise,
    quad_ul.dsol, quad_ul.dcache)

QUAD_ENZYME["unpruned"]["forward"]["large_mutable"] = @benchmarkable forward_quad!(
    $(copy(quad_ul.A_0)), $(copy(quad_ul.A_1)), $(copy(quad_ul.A_2)),
    $(copy(quad_ul.B)), $(copy(quad_ul.C_0)), $(copy(quad_ul.C_1)), $(copy(quad_ul.C_2)),
    $(copy(quad_ul.u0)), $([copy(n) for n in quad_ul.noise]),
    $(quad_ul.sol), $(quad_ul.cache),
    $(quad_ul.dA_0), $(quad_ul.dA_1), $(quad_ul.dA_2), $(quad_ul.dB),
    $(quad_ul.dC_0), $(quad_ul.dC_1), $(quad_ul.dC_2), $(quad_ul.du0), $(quad_ul.dnoise),
    $(quad_ul.dsol), $(quad_ul.dcache))

# =============================================================================
# Reverse mode AD — unpruned
# =============================================================================

# Warmup small
reverse_quad!(
    copy(quad_us.A_0), copy(quad_us.A_1), copy(quad_us.A_2),
    copy(quad_us.B), copy(quad_us.C_0), copy(quad_us.C_1), copy(quad_us.C_2),
    copy(quad_us.u0), [copy(n) for n in quad_us.noise],
    quad_us.sol, quad_us.cache,
    quad_us.dA_0, quad_us.dA_1, quad_us.dA_2, quad_us.dB,
    quad_us.dC_0, quad_us.dC_1, quad_us.dC_2, quad_us.du0, quad_us.dnoise,
    quad_us.dsol, quad_us.dcache)

QUAD_ENZYME["unpruned"]["reverse"]["small_mutable"] = @benchmarkable reverse_quad!(
    $(copy(quad_us.A_0)), $(copy(quad_us.A_1)), $(copy(quad_us.A_2)),
    $(copy(quad_us.B)), $(copy(quad_us.C_0)), $(copy(quad_us.C_1)), $(copy(quad_us.C_2)),
    $(copy(quad_us.u0)), $([copy(n) for n in quad_us.noise]),
    $(quad_us.sol), $(quad_us.cache),
    $(quad_us.dA_0), $(quad_us.dA_1), $(quad_us.dA_2), $(quad_us.dB),
    $(quad_us.dC_0), $(quad_us.dC_1), $(quad_us.dC_2), $(quad_us.du0), $(quad_us.dnoise),
    $(quad_us.dsol), $(quad_us.dcache))

# Warmup large
reverse_quad!(
    copy(quad_ul.A_0), copy(quad_ul.A_1), copy(quad_ul.A_2),
    copy(quad_ul.B), copy(quad_ul.C_0), copy(quad_ul.C_1), copy(quad_ul.C_2),
    copy(quad_ul.u0), [copy(n) for n in quad_ul.noise],
    quad_ul.sol, quad_ul.cache,
    quad_ul.dA_0, quad_ul.dA_1, quad_ul.dA_2, quad_ul.dB,
    quad_ul.dC_0, quad_ul.dC_1, quad_ul.dC_2, quad_ul.du0, quad_ul.dnoise,
    quad_ul.dsol, quad_ul.dcache)

QUAD_ENZYME["unpruned"]["reverse"]["large_mutable"] = @benchmarkable reverse_quad!(
    $(copy(quad_ul.A_0)), $(copy(quad_ul.A_1)), $(copy(quad_ul.A_2)),
    $(copy(quad_ul.B)), $(copy(quad_ul.C_0)), $(copy(quad_ul.C_1)), $(copy(quad_ul.C_2)),
    $(copy(quad_ul.u0)), $([copy(n) for n in quad_ul.noise]),
    $(quad_ul.sol), $(quad_ul.cache),
    $(quad_ul.dA_0), $(quad_ul.dA_1), $(quad_ul.dA_2), $(quad_ul.dB),
    $(quad_ul.dC_0), $(quad_ul.dC_1), $(quad_ul.dC_2), $(quad_ul.du0), $(quad_ul.dnoise),
    $(quad_ul.dsol), $(quad_ul.dcache))

# =============================================================================
# Forward mode AD — pruned
# =============================================================================

# Warmup small
forward_pruned_quad!(
    copy(quad_ps.A_0), copy(quad_ps.A_1), copy(quad_ps.A_2),
    copy(quad_ps.B), copy(quad_ps.C_0), copy(quad_ps.C_1), copy(quad_ps.C_2),
    copy(quad_ps.u0), [copy(n) for n in quad_ps.noise],
    quad_ps.sol, quad_ps.cache,
    quad_ps.dA_0, quad_ps.dA_1, quad_ps.dA_2, quad_ps.dB,
    quad_ps.dC_0, quad_ps.dC_1, quad_ps.dC_2, quad_ps.du0, quad_ps.dnoise,
    quad_ps.dsol, quad_ps.dcache)

QUAD_ENZYME["pruned"]["forward"]["small_mutable"] = @benchmarkable forward_pruned_quad!(
    $(copy(quad_ps.A_0)), $(copy(quad_ps.A_1)), $(copy(quad_ps.A_2)),
    $(copy(quad_ps.B)), $(copy(quad_ps.C_0)), $(copy(quad_ps.C_1)), $(copy(quad_ps.C_2)),
    $(copy(quad_ps.u0)), $([copy(n) for n in quad_ps.noise]),
    $(quad_ps.sol), $(quad_ps.cache),
    $(quad_ps.dA_0), $(quad_ps.dA_1), $(quad_ps.dA_2), $(quad_ps.dB),
    $(quad_ps.dC_0), $(quad_ps.dC_1), $(quad_ps.dC_2), $(quad_ps.du0), $(quad_ps.dnoise),
    $(quad_ps.dsol), $(quad_ps.dcache))

# Warmup large
forward_pruned_quad!(
    copy(quad_pl.A_0), copy(quad_pl.A_1), copy(quad_pl.A_2),
    copy(quad_pl.B), copy(quad_pl.C_0), copy(quad_pl.C_1), copy(quad_pl.C_2),
    copy(quad_pl.u0), [copy(n) for n in quad_pl.noise],
    quad_pl.sol, quad_pl.cache,
    quad_pl.dA_0, quad_pl.dA_1, quad_pl.dA_2, quad_pl.dB,
    quad_pl.dC_0, quad_pl.dC_1, quad_pl.dC_2, quad_pl.du0, quad_pl.dnoise,
    quad_pl.dsol, quad_pl.dcache)

QUAD_ENZYME["pruned"]["forward"]["large_mutable"] = @benchmarkable forward_pruned_quad!(
    $(copy(quad_pl.A_0)), $(copy(quad_pl.A_1)), $(copy(quad_pl.A_2)),
    $(copy(quad_pl.B)), $(copy(quad_pl.C_0)), $(copy(quad_pl.C_1)), $(copy(quad_pl.C_2)),
    $(copy(quad_pl.u0)), $([copy(n) for n in quad_pl.noise]),
    $(quad_pl.sol), $(quad_pl.cache),
    $(quad_pl.dA_0), $(quad_pl.dA_1), $(quad_pl.dA_2), $(quad_pl.dB),
    $(quad_pl.dC_0), $(quad_pl.dC_1), $(quad_pl.dC_2), $(quad_pl.du0), $(quad_pl.dnoise),
    $(quad_pl.dsol), $(quad_pl.dcache))

# =============================================================================
# Reverse mode AD — pruned
# =============================================================================

# Warmup small
reverse_pruned_quad!(
    copy(quad_ps.A_0), copy(quad_ps.A_1), copy(quad_ps.A_2),
    copy(quad_ps.B), copy(quad_ps.C_0), copy(quad_ps.C_1), copy(quad_ps.C_2),
    copy(quad_ps.u0), [copy(n) for n in quad_ps.noise],
    quad_ps.sol, quad_ps.cache,
    quad_ps.dA_0, quad_ps.dA_1, quad_ps.dA_2, quad_ps.dB,
    quad_ps.dC_0, quad_ps.dC_1, quad_ps.dC_2, quad_ps.du0, quad_ps.dnoise,
    quad_ps.dsol, quad_ps.dcache)

QUAD_ENZYME["pruned"]["reverse"]["small_mutable"] = @benchmarkable reverse_pruned_quad!(
    $(copy(quad_ps.A_0)), $(copy(quad_ps.A_1)), $(copy(quad_ps.A_2)),
    $(copy(quad_ps.B)), $(copy(quad_ps.C_0)), $(copy(quad_ps.C_1)), $(copy(quad_ps.C_2)),
    $(copy(quad_ps.u0)), $([copy(n) for n in quad_ps.noise]),
    $(quad_ps.sol), $(quad_ps.cache),
    $(quad_ps.dA_0), $(quad_ps.dA_1), $(quad_ps.dA_2), $(quad_ps.dB),
    $(quad_ps.dC_0), $(quad_ps.dC_1), $(quad_ps.dC_2), $(quad_ps.du0), $(quad_ps.dnoise),
    $(quad_ps.dsol), $(quad_ps.dcache))

# Warmup large
reverse_pruned_quad!(
    copy(quad_pl.A_0), copy(quad_pl.A_1), copy(quad_pl.A_2),
    copy(quad_pl.B), copy(quad_pl.C_0), copy(quad_pl.C_1), copy(quad_pl.C_2),
    copy(quad_pl.u0), [copy(n) for n in quad_pl.noise],
    quad_pl.sol, quad_pl.cache,
    quad_pl.dA_0, quad_pl.dA_1, quad_pl.dA_2, quad_pl.dB,
    quad_pl.dC_0, quad_pl.dC_1, quad_pl.dC_2, quad_pl.du0, quad_pl.dnoise,
    quad_pl.dsol, quad_pl.dcache)

QUAD_ENZYME["pruned"]["reverse"]["large_mutable"] = @benchmarkable reverse_pruned_quad!(
    $(copy(quad_pl.A_0)), $(copy(quad_pl.A_1)), $(copy(quad_pl.A_2)),
    $(copy(quad_pl.B)), $(copy(quad_pl.C_0)), $(copy(quad_pl.C_1)), $(copy(quad_pl.C_2)),
    $(copy(quad_pl.u0)), $([copy(n) for n in quad_pl.noise]),
    $(quad_pl.sol), $(quad_pl.cache),
    $(quad_pl.dA_0), $(quad_pl.dA_1), $(quad_pl.dA_2), $(quad_pl.dB),
    $(quad_pl.dC_0), $(quad_pl.dC_1), $(quad_pl.dC_2), $(quad_pl.du0), $(quad_pl.dnoise),
    $(quad_pl.dsol), $(quad_pl.dcache))

QUAD_ENZYME
