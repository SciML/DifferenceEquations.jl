# ForwardDiff AD benchmarks for DirectIteration (joint likelihood)
# Returns DI_FD BenchmarkGroup

using ForwardDiff
using DifferenceEquations: init, solve!, StateSpaceWorkspace

const DI_FD = BenchmarkGroup()
DI_FD["gradient"] = BenchmarkGroup()

# =============================================================================
# Type promotion helper
# =============================================================================

_fd_promote_di(::Type{T}, x::AbstractArray{T}) where {T} = x
_fd_promote_di(::Type{T}, x::AbstractArray) where {T} = T.(x)

# =============================================================================
# Problem sizes (same as enzyme_linear_likelihood.jl)
# =============================================================================

const p_di_fd_small = (; N = 5, M = 2, K = 2, L = 2, T = 10)
const p_di_fd_large = (; N = 30, M = 10, K = 10, L = 10, T = 100)

# =============================================================================
# Problem setup
# =============================================================================

function make_di_fd_benchmark(p; seed = 42)
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

    sim = solve(LinearStateSpaceProblem(A, B, u0, (0, T); C, noise))
    y = [sim.z[t + 1] + H * randn(L) for t in 1:T]

    return (; A, B, C, H, R, u0, noise, y)
end

# =============================================================================
# ForwardDiff wrapper — gradient of loglik w.r.t. vec(A)
# =============================================================================

function di_loglik_fd_bench(A_vec, B, C, u0, noise, y, H, N)
    T_el = eltype(A_vec)
    A = reshape(A_vec, N, N)
    R = _fd_promote_di(T_el, H) * _fd_promote_di(T_el, H)'
    prob = LinearStateSpaceProblem(
        A, _fd_promote_di(T_el, B),
        _fd_promote_di(T_el, u0), (0, length(y));
        C = _fd_promote_di(T_el, C),
        observables_noise = R,
        observables = y, noise = noise
    )
    sol = solve(prob, DirectIteration())
    return sol.logpdf
end

function fd_gradient_di!(A_vec, B, C, u0, noise, y, H, N)
    return ForwardDiff.gradient(
        a -> di_loglik_fd_bench(a, B, C, u0, noise, y, H, N), A_vec
    )
end

# =============================================================================
# Instantiate problems
# =============================================================================

const di_fd_s = make_di_fd_benchmark(p_di_fd_small)
const di_fd_l = make_di_fd_benchmark(p_di_fd_large)

# =============================================================================
# Warmup and benchmarks
# =============================================================================

fd_gradient_di!(
    vec(copy(di_fd_s.A)), di_fd_s.B, di_fd_s.C,
    di_fd_s.u0, di_fd_s.noise, di_fd_s.y, di_fd_s.H, p_di_fd_small.N
)
fd_gradient_di!(
    vec(copy(di_fd_l.A)), di_fd_l.B, di_fd_l.C,
    di_fd_l.u0, di_fd_l.noise, di_fd_l.y, di_fd_l.H, p_di_fd_large.N
)

DI_FD["gradient"]["small_mutable"] = @benchmarkable fd_gradient_di!(
    $(vec(copy(di_fd_s.A))), $(di_fd_s.B), $(di_fd_s.C),
    $(di_fd_s.u0), $(di_fd_s.noise), $(di_fd_s.y), $(di_fd_s.H),
    $(p_di_fd_small.N)
)

DI_FD["gradient"]["large_mutable"] = @benchmarkable fd_gradient_di!(
    $(vec(copy(di_fd_l.A))), $(di_fd_l.B), $(di_fd_l.C),
    $(di_fd_l.u0), $(di_fd_l.noise), $(di_fd_l.y), $(di_fd_l.H),
    $(p_di_fd_large.N)
)

# =============================================================================
# StaticArrays variant (small only)
# =============================================================================

DI_FD["gradient"]["small_static"] = let
    (; A, B, C, H, u0, noise, y) = di_fd_s
    N = p_di_fd_small.N; M = p_di_fd_small.M; K = p_di_fd_small.K; L = p_di_fd_small.L

    B_s = SMatrix{N, K}(B); C_s = SMatrix{M, N}(C); H_s = SMatrix{M, L}(H)
    noise_s = [SVector{K}(n) for n in noise]
    y_s = [SVector{M}(yi) for yi in y]

    function _di_loglik_static(
            A_vec, B_s, C_s, H_s, noise_s, y_s,
            ::Val{N_}, ::Val{M_}, ::Val{K_}, ::Val{L_}
        ) where {N_, M_, K_, L_}
        T_el = eltype(A_vec)
        A_d = SMatrix{N_, N_}(reshape(A_vec, N_, N_))
        B_d = SMatrix{N_, K_}(T_el.(B_s))
        C_d = SMatrix{M_, N_}(T_el.(C_s))
        H_d = SMatrix{M_, L_}(T_el.(H_s))
        R_d = H_d * H_d'
        u0_d = SVector{N_}(zeros(T_el, N_))
        prob = LinearStateSpaceProblem(
            A_d, B_d, u0_d, (0, length(y_s));
            C = C_d, observables_noise = R_d,
            observables = y_s, noise = noise_s
        )
        sol = solve(prob, DirectIteration())
        return sol.logpdf
    end

    A_vec = collect(vec(Matrix(A)))
    f = a -> _di_loglik_static(
        a, B_s, C_s, H_s, noise_s, y_s,
        Val(N), Val(M), Val(K), Val(L)
    )
    ForwardDiff.gradient(f, A_vec)

    @benchmarkable ForwardDiff.gradient($f, $(copy(A_vec)))
end

DI_FD
