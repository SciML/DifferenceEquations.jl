# ForwardDiff AD benchmarks for Kalman filter
# Returns KALMAN_FD BenchmarkGroup

using ForwardDiff
using DifferenceEquations: init, solve!, StateSpaceWorkspace

const KALMAN_FD = BenchmarkGroup()
KALMAN_FD["gradient"] = BenchmarkGroup()

# =============================================================================
# Type promotion helper
# =============================================================================

_fd_promote(::Type{T}, x::AbstractArray{T}) where {T} = x
_fd_promote(::Type{T}, x::AbstractArray) where {T} = T.(x)

# =============================================================================
# Problem sizes (same as enzyme_kalman.jl)
# =============================================================================

const p_kf_fd_small = (; N = 5, M = 2, K = 2, L = 2, T = 10)
const p_kf_fd_large = (; N = 30, M = 10, K = 10, L = 10, T = 100)

# =============================================================================
# Problem setup
# =============================================================================

function make_kalman_fd_benchmark(p; seed = 42)
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

    x0 = randn(N)
    noise = [randn(K) for _ in 1:T]
    sim = solve(LinearStateSpaceProblem(A, B, x0, (0, T); C, noise))
    y = [sim.z[t + 1] + H * randn(L) for t in 1:T]

    return (; A, B, C, R, mu_0, Sigma_0, y)
end

# =============================================================================
# ForwardDiff wrapper — gradient of loglik w.r.t. vec(A)
# =============================================================================

function kalman_loglik_fd_bench(A_vec, B, C, mu_0, Sigma_0, R, y, N)
    T_el = eltype(A_vec)
    A = reshape(A_vec, N, N)
    prob = LinearStateSpaceProblem(
        A, _fd_promote(T_el, B),
        zeros(T_el, N), (0, length(y));
        C = _fd_promote(T_el, C),
        u0_prior_mean = _fd_promote(T_el, mu_0),
        u0_prior_var = _fd_promote(T_el, Sigma_0),
        observables_noise = _fd_promote(T_el, R),
        observables = y
    )
    sol = solve(prob, KalmanFilter())
    return sol.logpdf
end

function fd_gradient_kalman!(A_vec, B, C, mu_0, Sigma_0, R, y, N)
    return ForwardDiff.gradient(
        a -> kalman_loglik_fd_bench(a, B, C, mu_0, Sigma_0, R, y, N), A_vec
    )
end

# =============================================================================
# Instantiate problems
# =============================================================================

const kf_fd_s = make_kalman_fd_benchmark(p_kf_fd_small)
const kf_fd_l = make_kalman_fd_benchmark(p_kf_fd_large)

# =============================================================================
# Warmup and benchmarks
# =============================================================================

# Warmup
fd_gradient_kalman!(
    vec(copy(kf_fd_s.A)), kf_fd_s.B, kf_fd_s.C,
    kf_fd_s.mu_0, kf_fd_s.Sigma_0, kf_fd_s.R, kf_fd_s.y, p_kf_fd_small.N
)
fd_gradient_kalman!(
    vec(copy(kf_fd_l.A)), kf_fd_l.B, kf_fd_l.C,
    kf_fd_l.mu_0, kf_fd_l.Sigma_0, kf_fd_l.R, kf_fd_l.y, p_kf_fd_large.N
)

KALMAN_FD["gradient"]["small_mutable"] = @benchmarkable fd_gradient_kalman!(
    $(vec(copy(kf_fd_s.A))), $(kf_fd_s.B), $(kf_fd_s.C),
    $(kf_fd_s.mu_0), $(kf_fd_s.Sigma_0), $(kf_fd_s.R), $(kf_fd_s.y),
    $(p_kf_fd_small.N)
)

KALMAN_FD["gradient"]["large_mutable"] = @benchmarkable fd_gradient_kalman!(
    $(vec(copy(kf_fd_l.A))), $(kf_fd_l.B), $(kf_fd_l.C),
    $(kf_fd_l.mu_0), $(kf_fd_l.Sigma_0), $(kf_fd_l.R), $(kf_fd_l.y),
    $(p_kf_fd_large.N)
)

# =============================================================================
# StaticArrays variant (small only — static types impractical for N=30)
# =============================================================================

KALMAN_FD["gradient"]["small_static"] = let
    (; A, B, C, R, mu_0, Sigma_0, y) = kf_fd_s
    N = p_kf_fd_small.N; M = p_kf_fd_small.M; K = p_kf_fd_small.K

    A_s = SMatrix{N, N}(A); B_s = SMatrix{N, K}(B); C_s = SMatrix{M, N}(C)
    R_s = SMatrix{M, M}(R); mu_s = SVector{N}(mu_0); Sig_s = SMatrix{N, N}(Sigma_0)
    y_s = [SVector{M}(yi) for yi in y]

    function _kf_loglik_static(
            A_vec, B_s, C_s, mu_s, Sig_s, R_s, y_s,
            ::Val{N_}, ::Val{M_}, ::Val{K_}
        ) where {N_, M_, K_}
        T_el = eltype(A_vec)
        A_d = SMatrix{N_, N_}(reshape(A_vec, N_, N_))
        prob = LinearStateSpaceProblem(
            A_d, SMatrix{N_, K_}(T_el.(B_s)),
            SVector{N_}(zeros(T_el, N_)), (0, length(y_s));
            C = SMatrix{M_, N_}(T_el.(C_s)),
            u0_prior_mean = SVector{N_}(T_el.(mu_s)),
            u0_prior_var = SMatrix{N_, N_}(T_el.(Sig_s)),
            observables_noise = SMatrix{M_, M_}(T_el.(R_s)),
            observables = y_s
        )
        sol = solve(prob, KalmanFilter())
        return sol.logpdf
    end

    A_vec = collect(vec(Matrix(A)))
    f = a -> _kf_loglik_static(
        a, B_s, C_s, mu_s, Sig_s, R_s, y_s,
        Val(N), Val(M), Val(K)
    )
    # Warmup
    ForwardDiff.gradient(f, A_vec)

    @benchmarkable ForwardDiff.gradient($f, $(copy(A_vec)))
end

KALMAN_FD
