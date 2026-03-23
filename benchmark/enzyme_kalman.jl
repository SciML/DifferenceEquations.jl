# Enzyme AD benchmarks for Kalman filter
# Returns KALMAN_ENZYME BenchmarkGroup

using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, StateSpaceWorkspace, zero_kalman_cache!!

const KALMAN_ENZYME = BenchmarkGroup()
KALMAN_ENZYME["raw"] = BenchmarkGroup()
KALMAN_ENZYME["forward"] = BenchmarkGroup()
KALMAN_ENZYME["reverse"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

const p_kf_small = (; N = 5, M = 2, K = 2, L = 2, T = 10)
const p_kf_large = (; N = 30, M = 10, K = 10, L = 10, T = 100)

# =============================================================================
# Problem setup
# =============================================================================

function make_kalman_benchmark(p; seed = 42)
    (; N, M, K, L, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    B = 0.1 * randn(N, K)
    C = randn(M, N)
    H = 0.1 * randn(M, L)
    R = H * H'
    mu_0 = zeros(N)
    Sigma_0 = Matrix{Float64}(I, N, N)

    # Generate observations using package's solve
    x0 = randn(N)
    noise = [randn(K) for _ in 1:T]
    sim = solve(LinearStateSpaceProblem(A, B, x0, (0, T); C, noise))
    y = [sim.z[t + 1] + H * randn(L) for t in 1:T]

    # Create problem and workspace
    prob = LinearStateSpaceProblem(A, B, zeros(N), (0, T); C,
        u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y)
    ws = init(prob, KalmanFilter())
    sol_out = ws.output
    cache = ws.cache

    # Shadow copies for AD (all Duplicated)
    dprob = make_zero(prob)
    dsol_out = make_zero(sol_out)
    dcache = make_zero(cache)
    dA = make_zero(A)
    dB = make_zero(B)
    dC = make_zero(C)
    dmu_0 = make_zero(mu_0)
    dSigma_0 = make_zero(Sigma_0)
    dR = make_zero(R)
    dy = [make_zero(y[1]) for _ in 1:T]

    return (; A, B, C, R, mu_0, Sigma_0, y, prob, sol_out, cache,
        dprob, dsol_out, dcache, dA, dB, dC, dmu_0, dSigma_0, dR, dy)
end

# =============================================================================
# Scalar wrapper for reverse mode (returns logpdf)
# =============================================================================

function kalman_loglik_bench!(A, B, C, mu_0, Sigma_0, R, y, prob, sol_out, cache)
    prob_new = remake(prob; A, B, C, u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y)
    ws = StateSpaceWorkspace(prob_new, KalmanFilter(), sol_out, cache)
    return solve!(ws).logpdf
end

# =============================================================================
# Forward wrapper (returns solution output matrices for tangent validation)
# =============================================================================

function kalman_forward_bench!(A, B, C, mu_0, Sigma_0, R, y, prob, sol_out, cache)
    prob_new = remake(prob; A, B, C, u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y)
    ws = StateSpaceWorkspace(prob_new, KalmanFilter(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.P[end])
end

# =============================================================================
# Instantiate problems
# =============================================================================

const kf_s = make_kalman_benchmark(p_kf_small)
const kf_l = make_kalman_benchmark(p_kf_large)

# =============================================================================
# Raw benchmarks (primal solve through public API)
# =============================================================================

function raw_kalman!(prob, sol_out, cache)
    ws = StateSpaceWorkspace(prob, KalmanFilter(), sol_out, cache)
    return solve!(ws).logpdf
end

# Warmup
raw_kalman!(kf_s.prob, kf_s.sol_out, kf_s.cache)
raw_kalman!(kf_l.prob, kf_l.sol_out, kf_l.cache)

KALMAN_ENZYME["raw"]["small_mutable"] = @benchmarkable raw_kalman!(
    $(kf_s.prob), $(kf_s.sol_out), $(kf_s.cache))
KALMAN_ENZYME["raw"]["large_mutable"] = @benchmarkable raw_kalman!(
    $(kf_l.prob), $(kf_l.sol_out), $(kf_l.cache))

# =============================================================================
# Forward mode AD — perturb A[1,1], return computed matrices
# =============================================================================

function forward_kalman_bench!(A, B, C, mu_0, Sigma_0, R, y, prob, sol_out, cache,
        dA, dB, dC, dmu_0, dSigma_0, dR, dy, dprob, dsol_out, dcache)
    # Zero all shadows
    make_zero!(dprob)
    make_zero!(dsol_out)
    make_zero!(dcache)
    make_zero!(dA); make_zero!(dB); make_zero!(dC)
    make_zero!(dmu_0); make_zero!(dSigma_0); make_zero!(dR)
    @inbounds for i in eachindex(dy)
        make_zero!(dy[i])
    end
    # Set perturbation direction
    dA[1, 1] = 1.0

    autodiff(Forward, kalman_forward_bench!,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Duplicated(R, dR), Duplicated(y, dy),
        Duplicated(prob, dprob), Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

# Warmup
forward_kalman_bench!(
    copy(kf_s.A), copy(kf_s.B), copy(kf_s.C),
    copy(kf_s.mu_0), copy(kf_s.Sigma_0), copy(kf_s.R),
    [copy(yi) for yi in kf_s.y], kf_s.prob, kf_s.sol_out, kf_s.cache,
    kf_s.dA, kf_s.dB, kf_s.dC, kf_s.dmu_0, kf_s.dSigma_0, kf_s.dR,
    kf_s.dy, kf_s.dprob, kf_s.dsol_out, kf_s.dcache)

KALMAN_ENZYME["forward"]["small_mutable"] = @benchmarkable forward_kalman_bench!(
    $(copy(kf_s.A)), $(copy(kf_s.B)), $(copy(kf_s.C)),
    $(copy(kf_s.mu_0)), $(copy(kf_s.Sigma_0)), $(copy(kf_s.R)),
    $([copy(yi) for yi in kf_s.y]), $(kf_s.prob), $(kf_s.cache),
    $(kf_s.dA), $(kf_s.dB), $(kf_s.dC), $(kf_s.dmu_0), $(kf_s.dSigma_0), $(kf_s.dR),
    $(kf_s.dy), $(kf_s.dprob), $(kf_s.dcache))

# Warmup large
forward_kalman_bench!(
    copy(kf_l.A), copy(kf_l.B), copy(kf_l.C),
    copy(kf_l.mu_0), copy(kf_l.Sigma_0), copy(kf_l.R),
    [copy(yi) for yi in kf_l.y], kf_l.prob, kf_l.sol_out, kf_l.cache,
    kf_l.dA, kf_l.dB, kf_l.dC, kf_l.dmu_0, kf_l.dSigma_0, kf_l.dR,
    kf_l.dy, kf_l.dprob, kf_l.dsol_out, kf_l.dcache)

KALMAN_ENZYME["forward"]["large_mutable"] = @benchmarkable forward_kalman_bench!(
    $(copy(kf_l.A)), $(copy(kf_l.B)), $(copy(kf_l.C)),
    $(copy(kf_l.mu_0)), $(copy(kf_l.Sigma_0)), $(copy(kf_l.R)),
    $([copy(yi) for yi in kf_l.y]), $(kf_l.prob), $(kf_l.cache),
    $(kf_l.dA), $(kf_l.dB), $(kf_l.dC), $(kf_l.dmu_0), $(kf_l.dSigma_0), $(kf_l.dR),
    $(kf_l.dy), $(kf_l.dprob), $(kf_l.dcache))

# =============================================================================
# Reverse mode AD — all Duplicated, scalar logpdf output
# =============================================================================

function reverse_kalman_bench!(A, B, C, mu_0, Sigma_0, R, y, prob, sol_out, cache,
        dA, dB, dC, dmu_0, dSigma_0, dR, dy, dprob, dsol_out, dcache)
    # Zero all shadows
    make_zero!(dprob)
    make_zero!(dsol_out)
    make_zero!(dcache)
    make_zero!(dA); make_zero!(dB); make_zero!(dC)
    make_zero!(dmu_0); make_zero!(dSigma_0); make_zero!(dR)
    @inbounds for i in eachindex(dy)
        make_zero!(dy[i])
    end

    autodiff(Reverse, kalman_loglik_bench!, Active,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Duplicated(R, dR), Duplicated(y, dy),
        Duplicated(prob, dprob), Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

# Warmup
reverse_kalman_bench!(
    copy(kf_s.A), copy(kf_s.B), copy(kf_s.C),
    copy(kf_s.mu_0), copy(kf_s.Sigma_0), copy(kf_s.R),
    [copy(yi) for yi in kf_s.y], kf_s.prob, kf_s.sol_out, kf_s.cache,
    kf_s.dA, kf_s.dB, kf_s.dC, kf_s.dmu_0, kf_s.dSigma_0, kf_s.dR,
    kf_s.dy, kf_s.dprob, kf_s.dsol_out, kf_s.dcache)

KALMAN_ENZYME["reverse"]["small_mutable"] = @benchmarkable reverse_kalman_bench!(
    $(copy(kf_s.A)), $(copy(kf_s.B)), $(copy(kf_s.C)),
    $(copy(kf_s.mu_0)), $(copy(kf_s.Sigma_0)), $(copy(kf_s.R)),
    $([copy(yi) for yi in kf_s.y]), $(kf_s.prob), $(kf_s.cache),
    $(kf_s.dA), $(kf_s.dB), $(kf_s.dC), $(kf_s.dmu_0), $(kf_s.dSigma_0), $(kf_s.dR),
    $(kf_s.dy), $(kf_s.dprob), $(kf_s.dcache))

# Warmup large
reverse_kalman_bench!(
    copy(kf_l.A), copy(kf_l.B), copy(kf_l.C),
    copy(kf_l.mu_0), copy(kf_l.Sigma_0), copy(kf_l.R),
    [copy(yi) for yi in kf_l.y], kf_l.prob, kf_l.sol_out, kf_l.cache,
    kf_l.dA, kf_l.dB, kf_l.dC, kf_l.dmu_0, kf_l.dSigma_0, kf_l.dR,
    kf_l.dy, kf_l.dprob, kf_l.dsol_out, kf_l.dcache)

KALMAN_ENZYME["reverse"]["large_mutable"] = @benchmarkable reverse_kalman_bench!(
    $(copy(kf_l.A)), $(copy(kf_l.B)), $(copy(kf_l.C)),
    $(copy(kf_l.mu_0)), $(copy(kf_l.Sigma_0)), $(copy(kf_l.R)),
    $([copy(yi) for yi in kf_l.y]), $(kf_l.prob), $(kf_l.cache),
    $(kf_l.dA), $(kf_l.dB), $(kf_l.dC), $(kf_l.dmu_0), $(kf_l.dSigma_0), $(kf_l.dR),
    $(kf_l.dy), $(kf_l.dprob), $(kf_l.dcache))

KALMAN_ENZYME
