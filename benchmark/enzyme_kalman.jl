# Enzyme AD benchmarks for Kalman filter
# Returns KALMAN_ENZYME BenchmarkGroup

using Enzyme: make_zero
using DifferenceEquations: _kalman_loglik!, alloc_kalman_cache, zero_kalman_cache!!

const KALMAN_ENZYME = BenchmarkGroup()
KALMAN_ENZYME["raw"] = BenchmarkGroup()
KALMAN_ENZYME["forward"] = BenchmarkGroup()
KALMAN_ENZYME["reverse"] = BenchmarkGroup()
KALMAN_ENZYME["reverse_model_params"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

const p_kf_small = (; N = 5, M = 2, K = 2, L = 2, T = 10)
const p_kf_large = (; N = 30, M = 10, K = 10, L = 10, T = 100)

# =============================================================================
# Problem setup — mutable arrays
# =============================================================================

function make_kalman_problem(p; seed = 42)
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

    # Generate observations
    x = [zeros(N) for _ in 1:(T + 1)]
    y = [zeros(M) for _ in 1:T]
    x[1] = randn(N)
    for t in 1:T
        x[t + 1] = A * x[t] + B * randn(K)
        y[t] = C * x[t] + H * randn(L)
    end

    # Allocate cache via LinearStateSpaceProblem
    prob = LinearStateSpaceProblem(A, B, zeros(N), (0, T); C,
        u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y, noise = nothing)
    cache = alloc_kalman_cache(prob, T + 1)

    # Shadow copies for AD
    dcache = make_zero(cache)
    dmu_0 = make_zero(mu_0)
    dSigma_0 = make_zero(Sigma_0)
    dy = [make_zero(y[1]) for _ in 1:T]
    dA = make_zero(A)
    dB = make_zero(B)
    dC = make_zero(C)

    return (; A, B, C, R, mu_0, Sigma_0, y, cache,
        dcache, dmu_0, dSigma_0, dy, dA, dB, dC)
end

# =============================================================================
# Problem setup — static arrays
# =============================================================================

function make_kalman_problem_static(p; seed = 42)
    (; N, M, K, L, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = SMatrix{N, N}(0.5 * A_raw / maximum(abs.(eigvals(A_raw))))
    B = SMatrix{N, K}(0.1 * randn(N, K))
    C = SMatrix{M, N}(randn(M, N))
    H = SMatrix{M, L}(0.1 * randn(M, L))
    R = SMatrix{M, M}(Matrix(H) * Matrix(H)')
    mu_0 = SVector{N}(zeros(N))
    Sigma_0 = SMatrix{N, N}(Matrix{Float64}(I, N, N))

    # Generate observations
    Random.seed!(seed)
    x_mut = randn(N)
    y = [SVector{M}(zeros(M)) for _ in 1:T]
    for t in 1:T
        y[t] = SVector{M}(Matrix(C) * x_mut + Matrix(H) * randn(L))
        x_mut = Matrix(A) * x_mut + Matrix(B) * randn(K)
    end

    # Allocate cache via LinearStateSpaceProblem
    prob = LinearStateSpaceProblem(A, B, SVector{N}(zeros(N)), (0, T); C,
        u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = Matrix(R), observables = y,
        noise = nothing)
    cache = alloc_kalman_cache(prob, T + 1)

    # Shadow copies for AD
    dcache = make_zero(cache)
    dmu_0 = make_zero(mu_0)
    dSigma_0 = make_zero(Sigma_0)
    dy = make_zero(y)
    dA = make_zero(A)
    dB = make_zero(B)
    dC = make_zero(C)

    return (; A, B, C, R, mu_0, Sigma_0, y, cache,
        dcache, dmu_0, dSigma_0, dy, dA, dB, dC)
end

# =============================================================================
# Instantiate problems
# =============================================================================

const kf_s = make_kalman_problem(p_kf_small)
const kf_ss = make_kalman_problem_static(p_kf_small)
const kf_l = make_kalman_problem(p_kf_large)

# =============================================================================
# Scalar wrapper (no zeroing, just calls underlying function)
# =============================================================================

scalar_kalman_loglik!(A, B, C, mu_0, Sigma_0, R, y, cache) =
    _kalman_loglik!(A, B, C, mu_0, Sigma_0, R, y, cache; perturb_diagonal = 1e-8)

# =============================================================================
# Helper: zero shadow cache (element-wise for static caches with immutable fields)
# =============================================================================

function zero_kalman_shadow_cache!(dcache)
    Enzyme.make_zero!(dcache)
    return nothing
end

function zero_kalman_shadow_cache_static!(dcache)
    # Static caches have vectors of immutable SArrays — zero the mutable vectors
    T_obs = length(dcache.mu_pred)
    @inbounds for t in 1:T_obs
        dcache.mu_pred[t] = zero(dcache.mu_pred[t])
        dcache.sigma_pred[t] = zero(dcache.sigma_pred[t])
        dcache.A_sigma[t] = zero(dcache.A_sigma[t])
        dcache.sigma_Gt[t] = zero(dcache.sigma_Gt[t])
        dcache.innovation[t] = zero(dcache.innovation[t])
        dcache.innovation_cov[t] = zero(dcache.innovation_cov[t])
        dcache.S_chol[t] = zero(dcache.S_chol[t])
        dcache.innovation_solved[t] = zero(dcache.innovation_solved[t])
        dcache.gain_rhs[t] = zero(dcache.gain_rhs[t])
        dcache.gain[t] = zero(dcache.gain[t])
        dcache.gainG[t] = zero(dcache.gainG[t])
        dcache.KgSigma[t] = zero(dcache.KgSigma[t])
        dcache.mu_update[t] = zero(dcache.mu_update[t])
    end
    T = length(dcache.u)
    @inbounds for t in 1:T
        dcache.u[t] = zero(dcache.u[t])
        dcache.P[t] = zero(dcache.P[t])
        dcache.z[t] = zero(dcache.z[t])
    end
    return nothing
end

# =============================================================================
# Raw benchmarks (include cache zeroing in the call)
# =============================================================================

function raw_kalman!(A, B, C, mu_0, Sigma_0, R, y, cache)
    zero_kalman_cache!!(cache)
    return _kalman_loglik!(A, B, C, mu_0, Sigma_0, R, y, cache; perturb_diagonal = 1e-8)
end

# Warmup
raw_kalman!(kf_s.A, kf_s.B, kf_s.C, kf_s.mu_0, kf_s.Sigma_0, kf_s.R, kf_s.y, kf_s.cache)
raw_kalman!(kf_ss.A, kf_ss.B, kf_ss.C, kf_ss.mu_0, kf_ss.Sigma_0, kf_ss.R, kf_ss.y, kf_ss.cache)
raw_kalman!(kf_l.A, kf_l.B, kf_l.C, kf_l.mu_0, kf_l.Sigma_0, kf_l.R, kf_l.y, kf_l.cache)

KALMAN_ENZYME["raw"]["small_mutable"] = @benchmarkable raw_kalman!(
    $(kf_s.A), $(kf_s.B), $(kf_s.C), $(kf_s.mu_0), $(kf_s.Sigma_0),
    $(kf_s.R), $(kf_s.y), $(kf_s.cache))

KALMAN_ENZYME["raw"]["small_static"] = @benchmarkable raw_kalman!(
    $(kf_ss.A), $(kf_ss.B), $(kf_ss.C), $(kf_ss.mu_0), $(kf_ss.Sigma_0),
    $(kf_ss.R), $(kf_ss.y), $(kf_ss.cache))

KALMAN_ENZYME["raw"]["large_mutable"] = @benchmarkable raw_kalman!(
    $(kf_l.A), $(kf_l.B), $(kf_l.C), $(kf_l.mu_0), $(kf_l.Sigma_0),
    $(kf_l.R), $(kf_l.y), $(kf_l.cache))

# =============================================================================
# Forward mode AD wrappers (mutable)
# =============================================================================

function forward_kalman_mutable!(A, B, C, mu_0, Sigma_0, R, y, cache,
        dmu_0, dSigma_0, dy, dcache)
    # Zero primal cache
    zero_kalman_cache!!(cache)
    # Zero shadows
    zero_kalman_shadow_cache!(dcache)
    Enzyme.make_zero!(dmu_0)
    Enzyme.make_zero!(dSigma_0)
    @inbounds for i in eachindex(dy)
        Enzyme.make_zero!(dy[i])
    end
    # Set perturbation direction
    dmu_0[1] = 1.0

    autodiff(Forward, scalar_kalman_loglik!, Const(A), Const(B), Const(C),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Const(R), Duplicated(y, dy), Duplicated(cache, dcache))
    return nothing
end

# Warmup
forward_kalman_mutable!(kf_s.A, kf_s.B, kf_s.C,
    copy(kf_s.mu_0), copy(kf_s.Sigma_0), kf_s.R, [copy(yi) for yi in kf_s.y], kf_s.cache,
    kf_s.dmu_0, kf_s.dSigma_0, kf_s.dy, kf_s.dcache)

KALMAN_ENZYME["forward"]["small_mutable"] = @benchmarkable forward_kalman_mutable!(
    $(kf_s.A), $(kf_s.B), $(kf_s.C),
    $(copy(kf_s.mu_0)), $(copy(kf_s.Sigma_0)), $(kf_s.R),
    $([copy(yi) for yi in kf_s.y]), $(kf_s.cache),
    $(kf_s.dmu_0), $(kf_s.dSigma_0), $(kf_s.dy), $(kf_s.dcache))

# Warmup large
forward_kalman_mutable!(kf_l.A, kf_l.B, kf_l.C,
    copy(kf_l.mu_0), copy(kf_l.Sigma_0), kf_l.R, [copy(yi) for yi in kf_l.y], kf_l.cache,
    kf_l.dmu_0, kf_l.dSigma_0, kf_l.dy, kf_l.dcache)

KALMAN_ENZYME["forward"]["large_mutable"] = @benchmarkable forward_kalman_mutable!(
    $(kf_l.A), $(kf_l.B), $(kf_l.C),
    $(copy(kf_l.mu_0)), $(copy(kf_l.Sigma_0)), $(kf_l.R),
    $([copy(yi) for yi in kf_l.y]), $(kf_l.cache),
    $(kf_l.dmu_0), $(kf_l.dSigma_0), $(kf_l.dy), $(kf_l.dcache))

# =============================================================================
# Forward mode AD wrappers (static)
# =============================================================================

function forward_kalman_static!(A, B, C, mu_0, Sigma_0, R, y, cache,
        dSigma_0, dy, dcache)
    # Zero primal cache
    zero_kalman_cache!!(cache)
    # Zero shadows (element-wise for immutable)
    zero_kalman_shadow_cache_static!(dcache)
    @inbounds for i in eachindex(dy)
        dy[i] = zero(dy[i])
    end
    # Set perturbation direction (immutable SVector)
    N = length(mu_0)
    dmu_0 = typeof(mu_0)(vcat(1.0, zeros(N - 1)))

    autodiff(Forward, scalar_kalman_loglik!, Const(A), Const(B), Const(C),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Const(R), Duplicated(y, dy), Duplicated(cache, dcache))
    return nothing
end

# Warmup
forward_kalman_static!(kf_ss.A, kf_ss.B, kf_ss.C,
    kf_ss.mu_0, kf_ss.Sigma_0, kf_ss.R, kf_ss.y, kf_ss.cache,
    kf_ss.dSigma_0, kf_ss.dy, kf_ss.dcache)

KALMAN_ENZYME["forward"]["small_static"] = @benchmarkable forward_kalman_static!(
    $(kf_ss.A), $(kf_ss.B), $(kf_ss.C),
    $(kf_ss.mu_0), $(kf_ss.Sigma_0), $(kf_ss.R), $(kf_ss.y), $(kf_ss.cache),
    $(kf_ss.dSigma_0), $(kf_ss.dy), $(kf_ss.dcache))

# =============================================================================
# Reverse mode AD wrappers (mutable)
# =============================================================================

function reverse_kalman_mutable!(A, B, C, mu_0, Sigma_0, R, y, cache,
        dmu_0, dSigma_0, dy, dcache)
    # Zero primal cache
    zero_kalman_cache!!(cache)
    # Zero shadows
    zero_kalman_shadow_cache!(dcache)
    Enzyme.make_zero!(dmu_0)
    Enzyme.make_zero!(dSigma_0)
    @inbounds for i in eachindex(dy)
        Enzyme.make_zero!(dy[i])
    end

    autodiff(Reverse, scalar_kalman_loglik!, Const(A), Const(B), Const(C),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Const(R), Duplicated(y, dy), Duplicated(cache, dcache))
    return nothing
end

# Warmup
reverse_kalman_mutable!(kf_s.A, kf_s.B, kf_s.C,
    copy(kf_s.mu_0), copy(kf_s.Sigma_0), kf_s.R, [copy(yi) for yi in kf_s.y], kf_s.cache,
    kf_s.dmu_0, kf_s.dSigma_0, kf_s.dy, kf_s.dcache)

KALMAN_ENZYME["reverse"]["small_mutable"] = @benchmarkable reverse_kalman_mutable!(
    $(kf_s.A), $(kf_s.B), $(kf_s.C),
    $(copy(kf_s.mu_0)), $(copy(kf_s.Sigma_0)), $(kf_s.R),
    $([copy(yi) for yi in kf_s.y]), $(kf_s.cache),
    $(kf_s.dmu_0), $(kf_s.dSigma_0), $(kf_s.dy), $(kf_s.dcache))

# Warmup large
reverse_kalman_mutable!(kf_l.A, kf_l.B, kf_l.C,
    copy(kf_l.mu_0), copy(kf_l.Sigma_0), kf_l.R, [copy(yi) for yi in kf_l.y], kf_l.cache,
    kf_l.dmu_0, kf_l.dSigma_0, kf_l.dy, kf_l.dcache)

KALMAN_ENZYME["reverse"]["large_mutable"] = @benchmarkable reverse_kalman_mutable!(
    $(kf_l.A), $(kf_l.B), $(kf_l.C),
    $(copy(kf_l.mu_0)), $(copy(kf_l.Sigma_0)), $(kf_l.R),
    $([copy(yi) for yi in kf_l.y]), $(kf_l.cache),
    $(kf_l.dmu_0), $(kf_l.dSigma_0), $(kf_l.dy), $(kf_l.dcache))

# =============================================================================
# Reverse mode AD wrappers (static)
# =============================================================================

function reverse_kalman_static!(A, B, C, mu_0, Sigma_0, R, y, cache,
        dmu_0, dSigma_0, dy, dcache)
    # Zero primal cache
    zero_kalman_cache!!(cache)
    # Zero shadows (element-wise for immutable)
    zero_kalman_shadow_cache_static!(dcache)
    @inbounds for i in eachindex(dy)
        dy[i] = zero(dy[i])
    end

    autodiff(Reverse, scalar_kalman_loglik!, Const(A), Const(B), Const(C),
        DuplicatedNoNeed(mu_0, dmu_0), DuplicatedNoNeed(Sigma_0, dSigma_0),
        Const(R), Duplicated(y, dy), Duplicated(cache, dcache))
    return nothing
end

# Warmup
reverse_kalman_static!(kf_ss.A, kf_ss.B, kf_ss.C,
    kf_ss.mu_0, kf_ss.Sigma_0, kf_ss.R, kf_ss.y, kf_ss.cache,
    kf_ss.dmu_0, kf_ss.dSigma_0, kf_ss.dy, kf_ss.dcache)

KALMAN_ENZYME["reverse"]["small_static"] = @benchmarkable reverse_kalman_static!(
    $(kf_ss.A), $(kf_ss.B), $(kf_ss.C),
    $(kf_ss.mu_0), $(kf_ss.Sigma_0), $(kf_ss.R), $(kf_ss.y), $(kf_ss.cache),
    $(kf_ss.dmu_0), $(kf_ss.dSigma_0), $(kf_ss.dy), $(kf_ss.dcache))

# =============================================================================
# Reverse mode AD w.r.t. model parameters (A, B, C) — mutable
# =============================================================================

function reverse_kalman_model_params_mutable!(A, B, C, mu_0, Sigma_0, R, y, cache,
        dA, dB, dC, dmu_0, dSigma_0, dy, dcache)
    # Zero primal cache
    zero_kalman_cache!!(cache)
    # Zero all shadows
    zero_kalman_shadow_cache!(dcache)
    Enzyme.make_zero!(dA)
    Enzyme.make_zero!(dB)
    Enzyme.make_zero!(dC)
    Enzyme.make_zero!(dmu_0)
    Enzyme.make_zero!(dSigma_0)
    @inbounds for i in eachindex(dy)
        Enzyme.make_zero!(dy[i])
    end

    autodiff(Reverse, scalar_kalman_loglik!,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Const(R), Duplicated(y, dy), Duplicated(cache, dcache))
    return nothing
end

# Warmup small
reverse_kalman_model_params_mutable!(
    copy(kf_s.A), copy(kf_s.B), copy(kf_s.C),
    copy(kf_s.mu_0), copy(kf_s.Sigma_0), kf_s.R, [copy(yi) for yi in kf_s.y], kf_s.cache,
    kf_s.dA, kf_s.dB, kf_s.dC, kf_s.dmu_0, kf_s.dSigma_0, kf_s.dy, kf_s.dcache)

KALMAN_ENZYME["reverse_model_params"]["small_mutable"] = @benchmarkable reverse_kalman_model_params_mutable!(
    $(copy(kf_s.A)), $(copy(kf_s.B)), $(copy(kf_s.C)),
    $(copy(kf_s.mu_0)), $(copy(kf_s.Sigma_0)), $(kf_s.R),
    $([copy(yi) for yi in kf_s.y]), $(kf_s.cache),
    $(kf_s.dA), $(kf_s.dB), $(kf_s.dC),
    $(kf_s.dmu_0), $(kf_s.dSigma_0), $(kf_s.dy), $(kf_s.dcache))

# Warmup large
reverse_kalman_model_params_mutable!(
    copy(kf_l.A), copy(kf_l.B), copy(kf_l.C),
    copy(kf_l.mu_0), copy(kf_l.Sigma_0), kf_l.R, [copy(yi) for yi in kf_l.y], kf_l.cache,
    kf_l.dA, kf_l.dB, kf_l.dC, kf_l.dmu_0, kf_l.dSigma_0, kf_l.dy, kf_l.dcache)

KALMAN_ENZYME["reverse_model_params"]["large_mutable"] = @benchmarkable reverse_kalman_model_params_mutable!(
    $(copy(kf_l.A)), $(copy(kf_l.B)), $(copy(kf_l.C)),
    $(copy(kf_l.mu_0)), $(copy(kf_l.Sigma_0)), $(kf_l.R),
    $([copy(yi) for yi in kf_l.y]), $(kf_l.cache),
    $(kf_l.dA), $(kf_l.dB), $(kf_l.dC),
    $(kf_l.dmu_0), $(kf_l.dSigma_0), $(kf_l.dy), $(kf_l.dcache))

# =============================================================================
# Reverse mode AD w.r.t. model parameters — static
# =============================================================================

function reverse_kalman_model_params_static!(A, B, C, mu_0, Sigma_0, R, y, cache,
        dA, dB, dC, dmu_0, dSigma_0, dy, dcache)
    # Zero primal cache
    zero_kalman_cache!!(cache)
    # Zero shadows (element-wise for immutable)
    zero_kalman_shadow_cache_static!(dcache)
    @inbounds for i in eachindex(dy)
        dy[i] = zero(dy[i])
    end

    autodiff(Reverse, scalar_kalman_loglik!,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Const(R), Duplicated(y, dy), Duplicated(cache, dcache))
    return nothing
end

# Warmup
reverse_kalman_model_params_static!(kf_ss.A, kf_ss.B, kf_ss.C,
    kf_ss.mu_0, kf_ss.Sigma_0, kf_ss.R, kf_ss.y, kf_ss.cache,
    kf_ss.dA, kf_ss.dB, kf_ss.dC, kf_ss.dmu_0, kf_ss.dSigma_0, kf_ss.dy, kf_ss.dcache)

KALMAN_ENZYME["reverse_model_params"]["small_static"] = @benchmarkable reverse_kalman_model_params_static!(
    $(kf_ss.A), $(kf_ss.B), $(kf_ss.C),
    $(kf_ss.mu_0), $(kf_ss.Sigma_0), $(kf_ss.R), $(kf_ss.y), $(kf_ss.cache),
    $(kf_ss.dA), $(kf_ss.dB), $(kf_ss.dC),
    $(kf_ss.dmu_0), $(kf_ss.dSigma_0), $(kf_ss.dy), $(kf_ss.dcache))

KALMAN_ENZYME
