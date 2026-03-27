# Apples-to-apples gradient benchmark: ForwardDiff vs Enzyme BatchDuplicated vs Enzyme Reverse
# All methods compute the SAME quantity: full gradient of loglik w.r.t. vec(A) (N² components).
#
# Returns GRAD_CMP BenchmarkGroup

using ForwardDiff
using Enzyme: make_zero, make_zero!, BatchDuplicated
using DifferenceEquations: init, solve!, StateSpaceWorkspace, fill_zero!!

const GRAD_CMP = BenchmarkGroup()
GRAD_CMP["kalman"] = BenchmarkGroup()
GRAD_CMP["di_likelihood"] = BenchmarkGroup()

# =============================================================================
# Type promotion helper (ForwardDiff path)
# =============================================================================

_gc_promote_bench(::Type{T}, x::AbstractArray{T}) where {T} = x
_gc_promote_bench(::Type{T}, x::AbstractArray) where {T} = T.(x)

# =============================================================================
# Problem sizes
# =============================================================================

const p_gc_small = (; N = 5, M = 2, K = 2, L = 2, T = 10)
const p_gc_large = (; N = 30, M = 10, K = 10, L = 10, T = 100)
const BATCH_SIZE = 10  # chunk size for both ForwardDiff and Enzyme BatchDuplicated

# =============================================================================
# Kalman setup
# =============================================================================

function make_gc_kalman(p; seed = 42)
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

    # Enzyme workspace
    prob = LinearStateSpaceProblem(
        A, B, zeros(N), (0, T); C,
        u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y
    )
    ws = init(prob, KalmanFilter())

    # Reverse shadows (single copy)
    rv_dA = make_zero(A); rv_dB = make_zero(B); rv_dC = make_zero(C)
    rv_dmu0 = make_zero(mu_0); rv_dSig0 = make_zero(Sigma_0); rv_dR = make_zero(R)
    rv_dy = [make_zero(y[1]) for _ in 1:T]
    rv_dsol = make_zero(ws.output); rv_dcache = make_zero(ws.cache)

    return (;
        A, B, C, R, mu_0, Sigma_0, y,
        sol_out = ws.output, cache = ws.cache,
        rv_dA, rv_dB, rv_dC, rv_dmu0, rv_dSig0, rv_dR, rv_dy, rv_dsol, rv_dcache,
    )
end

# =============================================================================
# Kalman wrapper functions
# =============================================================================

# Enzyme inner function (shared by forward & reverse)
function _kf_loglik_gc!(A, B, C, mu_0, Sigma_0, R, y, sol_out, cache)
    prob = LinearStateSpaceProblem(
        A, B, zeros(eltype(A), size(A, 1)), (0, length(y)); C,
        u0_prior_mean = mu_0, u0_prior_var = Sigma_0,
        observables_noise = R, observables = y
    )
    ws = StateSpaceWorkspace(prob, KalmanFilter(), sol_out, cache)
    return solve!(ws).logpdf
end

# ForwardDiff wrapper
function _kf_loglik_fd_gc(A_vec, B, C, mu_0, Sigma_0, R, y, N)
    T_el = eltype(A_vec)
    A = reshape(A_vec, N, N)
    prob = LinearStateSpaceProblem(
        A, _gc_promote_bench(T_el, B),
        zeros(T_el, N), (0, length(y));
        C = _gc_promote_bench(T_el, C),
        u0_prior_mean = _gc_promote_bench(T_el, mu_0),
        u0_prior_var = _gc_promote_bench(T_el, Sigma_0),
        observables_noise = _gc_promote_bench(T_el, R),
        observables = y
    )
    sol = solve(prob, KalmanFilter())
    return sol.logpdf
end

function bench_forwarddiff_kf!(A_vec, B, C, mu_0, Sigma_0, R, y, N)
    return ForwardDiff.gradient(
        a -> _kf_loglik_fd_gc(a, B, C, mu_0, Sigma_0, R, y, N), A_vec
    )
end

# Enzyme BatchDuplicated forward — full gradient
function bench_enzyme_batched_fwd_kf!(
        grad_out, A, B, C, mu_0, Sigma_0, R, y,
        sol_out, cache,
        dAs, dBs, dCs, dmu0s, dSig0s, dRs, dys, dsols, dcaches
    )
    chunk_size = length(dAs)
    N_params = length(vec(A))
    for chunk_start in 1:chunk_size:N_params
        chunk_end = min(chunk_start + chunk_size - 1, N_params)
        actual = chunk_end - chunk_start + 1

        for k in 1:chunk_size
            fill_zero!!(dAs[k]); fill_zero!!(dBs[k]); fill_zero!!(dCs[k])
            fill_zero!!(dmu0s[k]); fill_zero!!(dSig0s[k]); fill_zero!!(dRs[k])
            for t in eachindex(dys[k])
                dys[k][t] = fill_zero!!(dys[k][t])
            end
            make_zero!(dsols[k]); make_zero!(dcaches[k])
        end
        for k in 1:actual
            dAs[k][chunk_start + k - 1] = 1.0
        end

        result = autodiff(
            Forward, _kf_loglik_gc!,
            BatchDuplicated(A, dAs),
            BatchDuplicated(B, dBs),
            BatchDuplicated(C, dCs),
            BatchDuplicated(mu_0, dmu0s),
            BatchDuplicated(Sigma_0, dSig0s),
            BatchDuplicated(R, dRs),
            BatchDuplicated(y, dys),
            BatchDuplicated(sol_out, dsols),
            BatchDuplicated(cache, dcaches)
        )

        derivs = values(result[1])
        for k in 1:actual
            grad_out[chunk_start + k - 1] = derivs[k]
        end
    end
    return grad_out
end

# Enzyme Reverse — full gradient, extract dA
function bench_enzyme_reverse_kf!(
        A, B, C, mu_0, Sigma_0, R, y,
        sol_out, cache, dA, dB, dC, dmu_0, dSigma_0, dR, dy, dsol_out, dcache
    )
    make_zero!(dsol_out); make_zero!(dcache)
    fill_zero!!(dA); fill_zero!!(dB); fill_zero!!(dC)
    fill_zero!!(dmu_0); fill_zero!!(dSigma_0); fill_zero!!(dR)
    @inbounds for i in eachindex(dy)
        dy[i] = fill_zero!!(dy[i])
    end

    autodiff(
        Reverse, _kf_loglik_gc!, Active,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(mu_0, dmu_0), Duplicated(Sigma_0, dSigma_0),
        Duplicated(R, dR), Duplicated(y, dy),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache)
    )
    return vec(dA)
end

# =============================================================================
# Kalman benchmarks
# =============================================================================

const gc_kf_s = make_gc_kalman(p_gc_small)
const gc_kf_l = make_gc_kalman(p_gc_large)

# Warmup
bench_forwarddiff_kf!(
    vec(copy(gc_kf_s.A)), gc_kf_s.B, gc_kf_s.C,
    gc_kf_s.mu_0, gc_kf_s.Sigma_0, gc_kf_s.R, gc_kf_s.y, p_gc_small.N
)
# BatchDuplicated forward is always slower than ForwardDiff for this codebase:
# the shadow-copy overhead for all arguments (sol, cache, etc.) dominates.
# Kept for reference but not benchmarked.
# bench_enzyme_batched_fwd_kf!(zeros(p_gc_small.N^2),
#     gc_kf_s.A, gc_kf_s.B, gc_kf_s.C, gc_kf_s.mu_0, gc_kf_s.Sigma_0, gc_kf_s.R, gc_kf_s.y,
#     gc_kf_s.sol_out, gc_kf_s.cache,
#     gc_kf_s.bd_dAs, gc_kf_s.bd_dBs, gc_kf_s.bd_dCs, gc_kf_s.bd_dmu0s, gc_kf_s.bd_dSig0s,
#     gc_kf_s.bd_dRs, gc_kf_s.bd_dys, gc_kf_s.bd_dsols, gc_kf_s.bd_dcaches)
bench_enzyme_reverse_kf!(
    gc_kf_s.A, gc_kf_s.B, gc_kf_s.C,
    gc_kf_s.mu_0, gc_kf_s.Sigma_0, gc_kf_s.R, gc_kf_s.y,
    gc_kf_s.sol_out, gc_kf_s.cache,
    gc_kf_s.rv_dA, gc_kf_s.rv_dB, gc_kf_s.rv_dC, gc_kf_s.rv_dmu0, gc_kf_s.rv_dSig0,
    gc_kf_s.rv_dR, gc_kf_s.rv_dy, gc_kf_s.rv_dsol, gc_kf_s.rv_dcache
)

bench_forwarddiff_kf!(
    vec(copy(gc_kf_l.A)), gc_kf_l.B, gc_kf_l.C,
    gc_kf_l.mu_0, gc_kf_l.Sigma_0, gc_kf_l.R, gc_kf_l.y, p_gc_large.N
)
# bench_enzyme_batched_fwd_kf!(zeros(p_gc_large.N^2),
#     gc_kf_l.A, gc_kf_l.B, gc_kf_l.C, gc_kf_l.mu_0, gc_kf_l.Sigma_0, gc_kf_l.R, gc_kf_l.y,
#     gc_kf_l.sol_out, gc_kf_l.cache,
#     gc_kf_l.bd_dAs, gc_kf_l.bd_dBs, gc_kf_l.bd_dCs, gc_kf_l.bd_dmu0s, gc_kf_l.bd_dSig0s,
#     gc_kf_l.bd_dRs, gc_kf_l.bd_dys, gc_kf_l.bd_dsols, gc_kf_l.bd_dcaches)
bench_enzyme_reverse_kf!(
    gc_kf_l.A, gc_kf_l.B, gc_kf_l.C,
    gc_kf_l.mu_0, gc_kf_l.Sigma_0, gc_kf_l.R, gc_kf_l.y,
    gc_kf_l.sol_out, gc_kf_l.cache,
    gc_kf_l.rv_dA, gc_kf_l.rv_dB, gc_kf_l.rv_dC, gc_kf_l.rv_dmu0, gc_kf_l.rv_dSig0,
    gc_kf_l.rv_dR, gc_kf_l.rv_dy, gc_kf_l.rv_dsol, gc_kf_l.rv_dcache
)

# --- Kalman ForwardDiff ---
GRAD_CMP["kalman"]["forwarddiff_small"] = @benchmarkable bench_forwarddiff_kf!(
    $(vec(copy(gc_kf_s.A))), $(gc_kf_s.B), $(gc_kf_s.C),
    $(gc_kf_s.mu_0), $(gc_kf_s.Sigma_0), $(gc_kf_s.R), $(gc_kf_s.y), $(p_gc_small.N)
)

GRAD_CMP["kalman"]["forwarddiff_large"] = @benchmarkable bench_forwarddiff_kf!(
    $(vec(copy(gc_kf_l.A))), $(gc_kf_l.B), $(gc_kf_l.C),
    $(gc_kf_l.mu_0), $(gc_kf_l.Sigma_0), $(gc_kf_l.R), $(gc_kf_l.y), $(p_gc_large.N)
)

# --- Kalman Enzyme BatchDuplicated Forward (commented out — always slower than ForwardDiff) ---
# GRAD_CMP["kalman"]["enzyme_batched_fwd_small"] = @benchmarkable bench_enzyme_batched_fwd_kf!(
#     $(zeros(p_gc_small.N^2)),
#     $(gc_kf_s.A), $(gc_kf_s.B), $(gc_kf_s.C), $(gc_kf_s.mu_0), $(gc_kf_s.Sigma_0),
#     $(gc_kf_s.R), $(gc_kf_s.y), $(gc_kf_s.sol_out), $(gc_kf_s.cache),
#     $(gc_kf_s.bd_dAs), $(gc_kf_s.bd_dBs), $(gc_kf_s.bd_dCs), $(gc_kf_s.bd_dmu0s),
#     $(gc_kf_s.bd_dSig0s), $(gc_kf_s.bd_dRs), $(gc_kf_s.bd_dys),
#     $(gc_kf_s.bd_dsols), $(gc_kf_s.bd_dcaches))
#
# GRAD_CMP["kalman"]["enzyme_batched_fwd_large"] = @benchmarkable bench_enzyme_batched_fwd_kf!(
#     $(zeros(p_gc_large.N^2)),
#     $(gc_kf_l.A), $(gc_kf_l.B), $(gc_kf_l.C), $(gc_kf_l.mu_0), $(gc_kf_l.Sigma_0),
#     $(gc_kf_l.R), $(gc_kf_l.y), $(gc_kf_l.sol_out), $(gc_kf_l.cache),
#     $(gc_kf_l.bd_dAs), $(gc_kf_l.bd_dBs), $(gc_kf_l.bd_dCs), $(gc_kf_l.bd_dmu0s),
#     $(gc_kf_l.bd_dSig0s), $(gc_kf_l.bd_dRs), $(gc_kf_l.bd_dys),
#     $(gc_kf_l.bd_dsols), $(gc_kf_l.bd_dcaches))

# --- Kalman Enzyme Reverse ---
GRAD_CMP["kalman"]["enzyme_reverse_small"] = @benchmarkable bench_enzyme_reverse_kf!(
    $(gc_kf_s.A), $(gc_kf_s.B), $(gc_kf_s.C),
    $(gc_kf_s.mu_0), $(gc_kf_s.Sigma_0), $(gc_kf_s.R), $(gc_kf_s.y),
    $(gc_kf_s.sol_out), $(gc_kf_s.cache),
    $(gc_kf_s.rv_dA), $(gc_kf_s.rv_dB), $(gc_kf_s.rv_dC), $(gc_kf_s.rv_dmu0),
    $(gc_kf_s.rv_dSig0), $(gc_kf_s.rv_dR), $(gc_kf_s.rv_dy),
    $(gc_kf_s.rv_dsol), $(gc_kf_s.rv_dcache)
) teardown = (GC.enable(true); GC.gc(); GC.enable(false))

GRAD_CMP["kalman"]["enzyme_reverse_large"] = @benchmarkable bench_enzyme_reverse_kf!(
    $(gc_kf_l.A), $(gc_kf_l.B), $(gc_kf_l.C),
    $(gc_kf_l.mu_0), $(gc_kf_l.Sigma_0), $(gc_kf_l.R), $(gc_kf_l.y),
    $(gc_kf_l.sol_out), $(gc_kf_l.cache),
    $(gc_kf_l.rv_dA), $(gc_kf_l.rv_dB), $(gc_kf_l.rv_dC), $(gc_kf_l.rv_dmu0),
    $(gc_kf_l.rv_dSig0), $(gc_kf_l.rv_dR), $(gc_kf_l.rv_dy),
    $(gc_kf_l.rv_dsol), $(gc_kf_l.rv_dcache)
) teardown = (GC.enable(true); GC.gc(); GC.enable(false))

# =============================================================================
# DirectIteration likelihood setup
# =============================================================================

function make_gc_di(p; seed = 42)
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

    prob = LinearStateSpaceProblem(
        A, B, u0, (0, T); C,
        observables_noise = R, observables = y, noise
    )
    ws = init(prob, DirectIteration())

    rv_dA = make_zero(A); rv_dB = make_zero(B); rv_dC = make_zero(C)
    rv_du0 = make_zero(u0); rv_dH = make_zero(H)
    rv_dnoise = [make_zero(noise[1]) for _ in 1:T]
    rv_dy = [make_zero(y[1]) for _ in 1:T]
    rv_dsol = make_zero(ws.output); rv_dcache = make_zero(ws.cache)

    return (;
        A, B, C, H, R, u0, noise, y,
        sol_out = ws.output, cache = ws.cache,
        rv_dA, rv_dB, rv_dC, rv_du0, rv_dH, rv_dnoise, rv_dy, rv_dsol, rv_dcache,
    )
end

# =============================================================================
# DI wrapper functions
# =============================================================================

function _di_loglik_gc!(A, B, C, u0, noise, y, H, sol_out, cache)
    R = H * H'
    prob = LinearStateSpaceProblem(
        A, B, u0, (0, length(y));
        C, observables_noise = R, observables = y, noise
    )
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return solve!(ws).logpdf
end

function _di_loglik_fd_gc(A_vec, B, C, u0, noise, y, H, N)
    T_el = eltype(A_vec)
    A = reshape(A_vec, N, N)
    H_d = _gc_promote_bench(T_el, H)
    R = H_d * H_d'
    prob = LinearStateSpaceProblem(
        A, _gc_promote_bench(T_el, B),
        _gc_promote_bench(T_el, u0), (0, length(y));
        C = _gc_promote_bench(T_el, C),
        observables_noise = R,
        observables = y, noise = noise
    )
    sol = solve(prob, DirectIteration())
    return sol.logpdf
end

function bench_forwarddiff_di!(A_vec, B, C, u0, noise, y, H, N)
    return ForwardDiff.gradient(
        a -> _di_loglik_fd_gc(a, B, C, u0, noise, y, H, N), A_vec
    )
end

function bench_enzyme_batched_fwd_di!(
        grad_out, A, B, C, u0, noise, y, H,
        sol_out, cache,
        dAs, dBs, dCs, du0s, dnoises, dys, dHs, dsols, dcaches
    )
    chunk_size = length(dAs)
    N_params = length(vec(A))
    for chunk_start in 1:chunk_size:N_params
        chunk_end = min(chunk_start + chunk_size - 1, N_params)
        actual = chunk_end - chunk_start + 1

        for k in 1:chunk_size
            fill_zero!!(dAs[k]); fill_zero!!(dBs[k]); fill_zero!!(dCs[k])
            fill_zero!!(du0s[k]); fill_zero!!(dHs[k])
            for t in eachindex(dnoises[k])
                dnoises[k][t] = fill_zero!!(dnoises[k][t])
            end
            for t in eachindex(dys[k])
                dys[k][t] = fill_zero!!(dys[k][t])
            end
            make_zero!(dsols[k]); make_zero!(dcaches[k])
        end
        for k in 1:actual
            dAs[k][chunk_start + k - 1] = 1.0
        end

        result = autodiff(
            Forward, _di_loglik_gc!,
            BatchDuplicated(A, dAs),
            BatchDuplicated(B, dBs),
            BatchDuplicated(C, dCs),
            BatchDuplicated(u0, du0s),
            BatchDuplicated(noise, dnoises),
            BatchDuplicated(y, dys),
            BatchDuplicated(H, dHs),
            BatchDuplicated(sol_out, dsols),
            BatchDuplicated(cache, dcaches)
        )

        derivs = values(result[1])
        for k in 1:actual
            grad_out[chunk_start + k - 1] = derivs[k]
        end
    end
    return grad_out
end

function bench_enzyme_reverse_di!(
        A, B, C, u0, noise, y, H,
        sol_out, cache, dA, dB, dC, du0, dnoise, dy, dH, dsol_out, dcache
    )
    make_zero!(dsol_out); make_zero!(dcache)
    fill_zero!!(dA); fill_zero!!(dB); fill_zero!!(dC)
    fill_zero!!(du0); fill_zero!!(dH)
    @inbounds for i in eachindex(dnoise)
        dnoise[i] = fill_zero!!(dnoise[i])
    end
    @inbounds for i in eachindex(dy)
        dy[i] = fill_zero!!(dy[i])
    end

    autodiff(
        Reverse, _di_loglik_gc!, Active,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(y, dy), Duplicated(H, dH),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache)
    )
    return vec(dA)
end

# =============================================================================
# DI benchmarks
# =============================================================================

const gc_di_s = make_gc_di(p_gc_small)
const gc_di_l = make_gc_di(p_gc_large)

# Warmup
bench_forwarddiff_di!(
    vec(copy(gc_di_s.A)), gc_di_s.B, gc_di_s.C,
    gc_di_s.u0, gc_di_s.noise, gc_di_s.y, gc_di_s.H, p_gc_small.N
)
# bench_enzyme_batched_fwd_di!(zeros(p_gc_small.N^2),
#     gc_di_s.A, gc_di_s.B, gc_di_s.C, gc_di_s.u0, gc_di_s.noise, gc_di_s.y, gc_di_s.H,
#     gc_di_s.sol_out, gc_di_s.cache,
#     gc_di_s.bd_dAs, gc_di_s.bd_dBs, gc_di_s.bd_dCs, gc_di_s.bd_du0s,
#     gc_di_s.bd_dnoises, gc_di_s.bd_dys, gc_di_s.bd_dHs,
#     gc_di_s.bd_dsols, gc_di_s.bd_dcaches)
bench_enzyme_reverse_di!(
    gc_di_s.A, gc_di_s.B, gc_di_s.C,
    gc_di_s.u0, gc_di_s.noise, gc_di_s.y, gc_di_s.H,
    gc_di_s.sol_out, gc_di_s.cache,
    gc_di_s.rv_dA, gc_di_s.rv_dB, gc_di_s.rv_dC, gc_di_s.rv_du0,
    gc_di_s.rv_dnoise, gc_di_s.rv_dy, gc_di_s.rv_dH,
    gc_di_s.rv_dsol, gc_di_s.rv_dcache
)

bench_forwarddiff_di!(
    vec(copy(gc_di_l.A)), gc_di_l.B, gc_di_l.C,
    gc_di_l.u0, gc_di_l.noise, gc_di_l.y, gc_di_l.H, p_gc_large.N
)
# bench_enzyme_batched_fwd_di!(zeros(p_gc_large.N^2),
#     gc_di_l.A, gc_di_l.B, gc_di_l.C, gc_di_l.u0, gc_di_l.noise, gc_di_l.y, gc_di_l.H,
#     gc_di_l.sol_out, gc_di_l.cache,
#     gc_di_l.bd_dAs, gc_di_l.bd_dBs, gc_di_l.bd_dCs, gc_di_l.bd_du0s,
#     gc_di_l.bd_dnoises, gc_di_l.bd_dys, gc_di_l.bd_dHs,
#     gc_di_l.bd_dsols, gc_di_l.bd_dcaches)
bench_enzyme_reverse_di!(
    gc_di_l.A, gc_di_l.B, gc_di_l.C,
    gc_di_l.u0, gc_di_l.noise, gc_di_l.y, gc_di_l.H,
    gc_di_l.sol_out, gc_di_l.cache,
    gc_di_l.rv_dA, gc_di_l.rv_dB, gc_di_l.rv_dC, gc_di_l.rv_du0,
    gc_di_l.rv_dnoise, gc_di_l.rv_dy, gc_di_l.rv_dH,
    gc_di_l.rv_dsol, gc_di_l.rv_dcache
)

# --- DI ForwardDiff ---
GRAD_CMP["di_likelihood"]["forwarddiff_small"] = @benchmarkable bench_forwarddiff_di!(
    $(vec(copy(gc_di_s.A))), $(gc_di_s.B), $(gc_di_s.C),
    $(gc_di_s.u0), $(gc_di_s.noise), $(gc_di_s.y), $(gc_di_s.H), $(p_gc_small.N)
)

GRAD_CMP["di_likelihood"]["forwarddiff_large"] = @benchmarkable bench_forwarddiff_di!(
    $(vec(copy(gc_di_l.A))), $(gc_di_l.B), $(gc_di_l.C),
    $(gc_di_l.u0), $(gc_di_l.noise), $(gc_di_l.y), $(gc_di_l.H), $(p_gc_large.N)
)

# --- DI Enzyme BatchDuplicated Forward (commented out — always slower than ForwardDiff) ---
# GRAD_CMP["di_likelihood"]["enzyme_batched_fwd_small"] = @benchmarkable bench_enzyme_batched_fwd_di!(
#     $(zeros(p_gc_small.N^2)),
#     $(gc_di_s.A), $(gc_di_s.B), $(gc_di_s.C), $(gc_di_s.u0),
#     $(gc_di_s.noise), $(gc_di_s.y), $(gc_di_s.H),
#     $(gc_di_s.sol_out), $(gc_di_s.cache),
#     $(gc_di_s.bd_dAs), $(gc_di_s.bd_dBs), $(gc_di_s.bd_dCs), $(gc_di_s.bd_du0s),
#     $(gc_di_s.bd_dnoises), $(gc_di_s.bd_dys), $(gc_di_s.bd_dHs),
#     $(gc_di_s.bd_dsols), $(gc_di_s.bd_dcaches))
#
# GRAD_CMP["di_likelihood"]["enzyme_batched_fwd_large"] = @benchmarkable bench_enzyme_batched_fwd_di!(
#     $(zeros(p_gc_large.N^2)),
#     $(gc_di_l.A), $(gc_di_l.B), $(gc_di_l.C), $(gc_di_l.u0),
#     $(gc_di_l.noise), $(gc_di_l.y), $(gc_di_l.H),
#     $(gc_di_l.sol_out), $(gc_di_l.cache),
#     $(gc_di_l.bd_dAs), $(gc_di_l.bd_dBs), $(gc_di_l.bd_dCs), $(gc_di_l.bd_du0s),
#     $(gc_di_l.bd_dnoises), $(gc_di_l.bd_dys), $(gc_di_l.bd_dHs),
#     $(gc_di_l.bd_dsols), $(gc_di_l.bd_dcaches))

# --- DI Enzyme Reverse ---
GRAD_CMP["di_likelihood"]["enzyme_reverse_small"] = @benchmarkable bench_enzyme_reverse_di!(
    $(gc_di_s.A), $(gc_di_s.B), $(gc_di_s.C),
    $(gc_di_s.u0), $(gc_di_s.noise), $(gc_di_s.y), $(gc_di_s.H),
    $(gc_di_s.sol_out), $(gc_di_s.cache),
    $(gc_di_s.rv_dA), $(gc_di_s.rv_dB), $(gc_di_s.rv_dC), $(gc_di_s.rv_du0),
    $(gc_di_s.rv_dnoise), $(gc_di_s.rv_dy), $(gc_di_s.rv_dH),
    $(gc_di_s.rv_dsol), $(gc_di_s.rv_dcache)
) teardown = (GC.enable(true); GC.gc(); GC.enable(false))

GRAD_CMP["di_likelihood"]["enzyme_reverse_large"] = @benchmarkable bench_enzyme_reverse_di!(
    $(gc_di_l.A), $(gc_di_l.B), $(gc_di_l.C),
    $(gc_di_l.u0), $(gc_di_l.noise), $(gc_di_l.y), $(gc_di_l.H),
    $(gc_di_l.sol_out), $(gc_di_l.cache),
    $(gc_di_l.rv_dA), $(gc_di_l.rv_dB), $(gc_di_l.rv_dC), $(gc_di_l.rv_du0),
    $(gc_di_l.rv_dnoise), $(gc_di_l.rv_dy), $(gc_di_l.rv_dH),
    $(gc_di_l.rv_dsol), $(gc_di_l.rv_dcache)
) teardown = (GC.enable(true); GC.gc(); GC.enable(false))

GRAD_CMP
