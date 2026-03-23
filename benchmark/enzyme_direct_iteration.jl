# Enzyme AD benchmarks for DirectIteration (joint likelihood)
# Returns DI_ENZYME BenchmarkGroup

using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, StateSpaceWorkspace

const DI_ENZYME = BenchmarkGroup()
DI_ENZYME["raw"] = BenchmarkGroup()
DI_ENZYME["forward"] = BenchmarkGroup()
DI_ENZYME["reverse"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

const p_di_small = (; N = 5, M = 2, K = 2, L = 2, T = 10)
const p_di_large = (; N = 30, M = 10, K = 10, L = 10, T = 100)

# =============================================================================
# Problem setup
# =============================================================================

function make_di_benchmark(p; seed = 42)
    (; N, M, K, L, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    B = 0.1 * randn(N, K)
    C = randn(M, N)
    H = 0.1 * randn(M, L)
    R = H * H'
    u0 = zeros(N)
    noise = [randn(K) for _ in 1:T]

    # Generate observations using package's solve
    sim = solve(LinearStateSpaceProblem(A, B, u0, (0, T); C, noise))
    y = [sim.z[t + 1] + H * randn(L) for t in 1:T]

    # Create problem and workspace
    prob = LinearStateSpaceProblem(A, B, u0, (0, T); C,
        observables_noise = R, observables = y, noise)
    ws = init(prob, DirectIteration())
    cache = ws.cache

    # Shadow copies for AD (all Duplicated)
    dprob = make_zero(prob)
    dcache = make_zero(cache)
    dA = make_zero(A)
    dB = make_zero(B)
    dC = make_zero(C)
    dH = make_zero(H)
    du0 = make_zero(u0)
    dnoise = [make_zero(noise[1]) for _ in 1:T]
    dy = [make_zero(y[1]) for _ in 1:T]

    return (; A, B, C, H, R, u0, noise, y, prob, cache,
        dprob, dcache, dA, dB, dC, dH, du0, dnoise, dy)
end

# =============================================================================
# Scalar wrapper for reverse mode (returns logpdf)
# =============================================================================

function di_loglik_bench!(A, B, C, u0, noise, y, H, prob, cache)
    R = H * H'
    prob_new = remake(prob; A, B, C, u0, observables_noise = R, observables = y, noise)
    ws = StateSpaceWorkspace(prob_new, DirectIteration(), cache)
    return solve!(ws).logpdf
end

# =============================================================================
# Forward wrapper (returns matrices from cache)
# =============================================================================

function di_forward_bench!(A, B, C, u0, noise, y, H, prob, cache)
    R = H * H'
    prob_new = remake(prob; A, B, C, u0, observables_noise = R, observables = y, noise)
    ws = StateSpaceWorkspace(prob_new, DirectIteration(), cache)
    solve!(ws)
    return (cache.u[end], cache.z[end])
end

# =============================================================================
# Instantiate problems
# =============================================================================

const di_s = make_di_benchmark(p_di_small)
const di_l = make_di_benchmark(p_di_large)

# =============================================================================
# Raw benchmarks (primal solve through public API)
# =============================================================================

function raw_di!(prob, cache)
    ws = StateSpaceWorkspace(prob, DirectIteration(), cache)
    return solve!(ws).logpdf
end

# Warmup
raw_di!(di_s.prob, di_s.cache)
raw_di!(di_l.prob, di_l.cache)

DI_ENZYME["raw"]["small_mutable"] = @benchmarkable raw_di!($(di_s.prob), $(di_s.cache))
DI_ENZYME["raw"]["large_mutable"] = @benchmarkable raw_di!($(di_l.prob), $(di_l.cache))

# =============================================================================
# Forward mode AD — perturb A[1,1], return computed matrices
# =============================================================================

function forward_di_bench!(A, B, C, u0, noise, y, H, prob, cache,
        dA, dB, dC, du0, dnoise, dy, dH, dprob, dcache)
    # Zero all shadows
    make_zero!(dprob)
    make_zero!(dcache)
    make_zero!(dA); make_zero!(dB); make_zero!(dC); make_zero!(dH)
    make_zero!(du0)
    @inbounds for i in eachindex(dnoise)
        make_zero!(dnoise[i])
    end
    @inbounds for i in eachindex(dy)
        make_zero!(dy[i])
    end
    # Set perturbation direction
    dA[1, 1] = 1.0

    autodiff(Forward, di_forward_bench!,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(noise, dnoise), Duplicated(y, dy),
        Duplicated(H, dH),
        Duplicated(prob, dprob), Duplicated(cache, dcache))
    return nothing
end

# Warmup
forward_di_bench!(
    copy(di_s.A), copy(di_s.B), copy(di_s.C),
    copy(di_s.u0), [copy(n) for n in di_s.noise], [copy(yi) for yi in di_s.y],
    copy(di_s.H), di_s.prob, di_s.cache,
    di_s.dA, di_s.dB, di_s.dC, di_s.du0, di_s.dnoise, di_s.dy, di_s.dH,
    di_s.dprob, di_s.dcache)

DI_ENZYME["forward"]["small_mutable"] = @benchmarkable forward_di_bench!(
    $(copy(di_s.A)), $(copy(di_s.B)), $(copy(di_s.C)),
    $(copy(di_s.u0)), $([copy(n) for n in di_s.noise]), $([copy(yi) for yi in di_s.y]),
    $(copy(di_s.H)), $(di_s.prob), $(di_s.cache),
    $(di_s.dA), $(di_s.dB), $(di_s.dC), $(di_s.du0), $(di_s.dnoise), $(di_s.dy), $(di_s.dH),
    $(di_s.dprob), $(di_s.dcache))

# Warmup large
forward_di_bench!(
    copy(di_l.A), copy(di_l.B), copy(di_l.C),
    copy(di_l.u0), [copy(n) for n in di_l.noise], [copy(yi) for yi in di_l.y],
    copy(di_l.H), di_l.prob, di_l.cache,
    di_l.dA, di_l.dB, di_l.dC, di_l.du0, di_l.dnoise, di_l.dy, di_l.dH,
    di_l.dprob, di_l.dcache)

DI_ENZYME["forward"]["large_mutable"] = @benchmarkable forward_di_bench!(
    $(copy(di_l.A)), $(copy(di_l.B)), $(copy(di_l.C)),
    $(copy(di_l.u0)), $([copy(n) for n in di_l.noise]), $([copy(yi) for yi in di_l.y]),
    $(copy(di_l.H)), $(di_l.prob), $(di_l.cache),
    $(di_l.dA), $(di_l.dB), $(di_l.dC), $(di_l.du0), $(di_l.dnoise), $(di_l.dy), $(di_l.dH),
    $(di_l.dprob), $(di_l.dcache))

# =============================================================================
# Reverse mode AD — all Duplicated, scalar logpdf output
# =============================================================================

function reverse_di_bench!(A, B, C, u0, noise, y, H, prob, cache,
        dA, dB, dC, du0, dnoise, dy, dH, dprob, dcache)
    # Zero all shadows
    make_zero!(dprob)
    make_zero!(dcache)
    make_zero!(dA); make_zero!(dB); make_zero!(dC); make_zero!(dH)
    make_zero!(du0)
    @inbounds for i in eachindex(dnoise)
        make_zero!(dnoise[i])
    end
    @inbounds for i in eachindex(dy)
        make_zero!(dy[i])
    end

    autodiff(Reverse, di_loglik_bench!, Active,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(noise, dnoise), Duplicated(y, dy),
        Duplicated(H, dH),
        Duplicated(prob, dprob), Duplicated(cache, dcache))
    return nothing
end

# Warmup
reverse_di_bench!(
    copy(di_s.A), copy(di_s.B), copy(di_s.C),
    copy(di_s.u0), [copy(n) for n in di_s.noise], [copy(yi) for yi in di_s.y],
    copy(di_s.H), di_s.prob, di_s.cache,
    di_s.dA, di_s.dB, di_s.dC, di_s.du0, di_s.dnoise, di_s.dy, di_s.dH,
    di_s.dprob, di_s.dcache)

DI_ENZYME["reverse"]["small_mutable"] = @benchmarkable reverse_di_bench!(
    $(copy(di_s.A)), $(copy(di_s.B)), $(copy(di_s.C)),
    $(copy(di_s.u0)), $([copy(n) for n in di_s.noise]), $([copy(yi) for yi in di_s.y]),
    $(copy(di_s.H)), $(di_s.prob), $(di_s.cache),
    $(di_s.dA), $(di_s.dB), $(di_s.dC), $(di_s.du0), $(di_s.dnoise), $(di_s.dy), $(di_s.dH),
    $(di_s.dprob), $(di_s.dcache))

# Warmup large
reverse_di_bench!(
    copy(di_l.A), copy(di_l.B), copy(di_l.C),
    copy(di_l.u0), [copy(n) for n in di_l.noise], [copy(yi) for yi in di_l.y],
    copy(di_l.H), di_l.prob, di_l.cache,
    di_l.dA, di_l.dB, di_l.dC, di_l.du0, di_l.dnoise, di_l.dy, di_l.dH,
    di_l.dprob, di_l.dcache)

DI_ENZYME["reverse"]["large_mutable"] = @benchmarkable reverse_di_bench!(
    $(copy(di_l.A)), $(copy(di_l.B)), $(copy(di_l.C)),
    $(copy(di_l.u0)), $([copy(n) for n in di_l.noise]), $([copy(yi) for yi in di_l.y]),
    $(copy(di_l.H)), $(di_l.prob), $(di_l.cache),
    $(di_l.dA), $(di_l.dB), $(di_l.dC), $(di_l.du0), $(di_l.dnoise), $(di_l.dy), $(di_l.dH),
    $(di_l.dprob), $(di_l.dcache))

DI_ENZYME
