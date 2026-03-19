# Primal-only benchmarks for DifferenceEquations.jl
# All benchmarks use pre-allocated caches (init/solve!) to measure only
# the solver loop — zero allocation overhead from problem/cache construction.
#
# Usage: julia --project=benchmark benchmark/primal_benchmarks.jl

using DifferenceEquations, BenchmarkTools
using DelimitedFiles, Distributions, LinearAlgebra, Random, StaticArrays
using DifferenceEquations: mul!!, muladd!!, init, solve!

# Check if MKL or not
julia_mkl = any(contains("mkl"), getfield.(LinearAlgebra.BLAS.get_config().loaded_libs, :libname))

if !julia_mkl
    openblas_threads = min(4, Int64(round(Sys.CPU_THREADS / 2)))
    BLAS.set_num_threads(openblas_threads)
end

println("Julia $(VERSION)")
println("Threads.nthreads = $(Threads.nthreads()), MKL = $julia_mkl, BLAS.num_threads = $(BLAS.get_num_threads())")
println()

# =============================================================================
# Callbacks
# =============================================================================

# !! pattern: works for both mutable (Vector/Matrix) and static (SVector/SMatrix)
@inline function f_lss!!(x_p, x, w, p, t)
    x_p = mul!!(x_p, p.A, x)
    return muladd!!(x_p, p.B, w)
end

@inline function g_lss!!(y, x, p, t)
    return mul!!(y, p.C, x)
end

# Quadratic callbacks with captured mutable state
function make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
    n_x = length(u0)
    n_obs = length(C_0)
    u_f = copy(u0)
    u_f_new = similar(u0)

    function f!!(x_next, x, w, p, t)
        mul!(u_f_new, A_1, u_f)
        mul!(u_f_new, B, w, 1.0, 1.0)
        copyto!(x_next, A_0)
        mul!(x_next, A_1, x, 1.0, 1.0)
        @inbounds for i in 1:n_x
            x_next[i] += dot(u_f, view(A_2, i, :, :), u_f)
        end
        mul!(x_next, B, w, 1.0, 1.0)
        copyto!(u_f, u_f_new)
        return x_next
    end

    function g!!(y, x, p, t)
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
# Linear model data
# =============================================================================

const A_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
const B_rbc = reshape([0.0; -0.01], 2, 1)
const C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
const D_rbc = abs2.([0.1, 0.1])
const u0_rbc = zeros(2)
const u0_prior_var_rbc = diagm(ones(length(u0_rbc)))

const observables_rbc = [collect(row) for row in eachrow(readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ','))]
const noise_rbc = [collect(row) for row in eachrow(readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ','))]
const T_rbc = length(observables_rbc)

const A_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A.csv"), ',')
const B_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
const C_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C.csv"), ',')
const D_FVGQ = abs2.(ones(6) * 1.0e-3)
const observables_FVGQ = [collect(row) for row in eachrow(readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_observables.csv"), ','))]
const noise_FVGQ = [collect(row) for row in eachrow(readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"), ','))]
const u0_FVGQ = zeros(size(A_FVGQ, 1))
const u0_prior_var_FVGQ = diagm(ones(length(u0_FVGQ)))
const T_FVGQ = length(observables_FVGQ)

# =============================================================================
# Quadratic model data
# =============================================================================

const A_0_rbc = [-7.824904812740593e-5, 0.0]
const A_1_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
const A_2_rbc = cat([-0.00019761505863889124 0.03375055315837927; 0.0 0.0],
    [0.03375055315837913 3.128758481817603; 0.0 0.0]; dims = 3)
const B_2_rbc = reshape([0.0; -0.01], 2, 1)
const C_0_rbc = [7.824904812740593e-5, 0.0]
const C_1_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
const C_2_rbc = cat([-0.00018554166974717046 0.0025652363153049716; 0.0 0.0],
    [0.002565236315304951 0.3132705036896446; 0.0 0.0]; dims = 3)
const D_2_rbc = abs2.([0.1, 0.1])
const u0_2_rbc = zeros(2)
const observables_2_rbc = [collect(row) for row in eachrow(readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ','))]
const noise_2_rbc = [collect(row) for row in eachrow(readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ','))]

A_0_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_0.csv"), ',')
const A_0_FVGQ = vec(A_0_raw)
const A_1_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_1.csv"), ',')
A_2_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_2.csv"), ',')
const A_2_FVGQ = reshape(A_2_raw, length(A_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
const B_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
C_0_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_0.csv"), ',')
const C_0_FVGQ = vec(C_0_raw)
const C_1_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_1.csv"), ',')
C_2_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_2.csv"), ',')
const C_2_FVGQ = reshape(C_2_raw, length(C_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
const D_2_FVGQ = ones(6) * 1.0e-3
const u0_2_FVGQ = zeros(size(A_1_FVGQ, 1))
const observables_2_FVGQ = [collect(row) for row in eachrow(readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_observables.csv"), ','))]
const noise_2_FVGQ = [collect(row) for row in eachrow(readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"), ','))]

# =============================================================================
# Static data — same values wrapped in SMatrix/SVector
# =============================================================================

const A_s = SMatrix{2, 2}(A_rbc)
const B_s = SMatrix{2, 1}(B_rbc)
const C_s = SMatrix{2, 2}(C_rbc)
const u0_s = SVector{2}(u0_rbc)
const nse_s = [SVector{1}(noise_rbc[t]) for t in 1:length(noise_rbc)]
const p_m = (; A = A_rbc, B = B_rbc, C = C_rbc)
const p_s = (; A = A_s, B = B_s, C = C_s)

# =============================================================================
# Pre-allocate all workspaces (one-time cost, not benchmarked)
# =============================================================================

# Linear RBC — DirectIteration
const ws_lin_rbc_di = init(
    LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, T_rbc);
        C = C_rbc, observables_noise = D_rbc, noise = noise_rbc, observables = observables_rbc),
    DirectIteration())

# Linear RBC — KalmanFilter
const ws_lin_rbc_kf = init(
    LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, T_rbc);
        C = C_rbc, observables_noise = D_rbc, observables = observables_rbc,
        u0_prior_mean = u0_rbc, u0_prior_var = u0_prior_var_rbc),
    KalmanFilter())

# Linear RBC — simulation (no observables)
const ws_lin_rbc_sim = init(
    LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, T_rbc); C = C_rbc, noise = noise_rbc),
    DirectIteration())

# Linear FVGQ — DirectIteration
const ws_lin_fvgq_di = init(
    LinearStateSpaceProblem(A_FVGQ, B_FVGQ, u0_FVGQ, (0, T_FVGQ);
        C = C_FVGQ, observables_noise = D_FVGQ, noise = noise_FVGQ, observables = observables_FVGQ),
    DirectIteration())

# Linear FVGQ — KalmanFilter
const ws_lin_fvgq_kf = init(
    LinearStateSpaceProblem(A_FVGQ, B_FVGQ, u0_FVGQ, (0, T_FVGQ);
        C = C_FVGQ, observables_noise = D_FVGQ, observables = observables_FVGQ,
        u0_prior_mean = u0_FVGQ, u0_prior_var = u0_prior_var_FVGQ),
    KalmanFilter())

# Linear FVGQ — simulation (no observables)
const ws_lin_fvgq_sim = init(
    LinearStateSpaceProblem(A_FVGQ, B_FVGQ, u0_FVGQ, (0, T_FVGQ); C = C_FVGQ, noise = noise_FVGQ),
    DirectIteration())

# Quadratic (Generic) RBC
const f_quad_rbc!!, g_quad_rbc!! = make_quadratic_callbacks(
    A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_2_rbc)
const ws_quad_rbc = init(
    StateSpaceProblem(f_quad_rbc!!, g_quad_rbc!!, u0_2_rbc, (0, length(observables_2_rbc));
        n_shocks = 1, n_obs = 2, observables_noise = D_2_rbc,
        noise = noise_2_rbc, observables = observables_2_rbc),
    DirectIteration())

# Quadratic (Generic) FVGQ
const f_quad_fvgq!!, g_quad_fvgq!! = make_quadratic_callbacks(
    A_0_FVGQ, A_1_FVGQ, A_2_FVGQ, B_2_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ, u0_2_FVGQ)
const ws_quad_fvgq = init(
    StateSpaceProblem(f_quad_fvgq!!, g_quad_fvgq!!, u0_2_FVGQ, (0, length(observables_2_FVGQ));
        n_shocks = size(B_2_FVGQ, 2), n_obs = length(C_0_FVGQ),
        observables_noise = D_2_FVGQ, noise = noise_2_FVGQ, observables = observables_2_FVGQ),
    DirectIteration())

# Generic Linear — mutable (!! callbacks)
const ws_gen_mut = init(
    StateSpaceProblem(f_lss!!, g_lss!!, u0_rbc, (0, T_rbc), p_m;
        n_shocks = 1, n_obs = 2, observables_noise = D_rbc,
        noise = noise_rbc, observables = observables_rbc),
    DirectIteration())

# Generic Linear — static (!! callbacks, same functions)
const ws_gen_static = init(
    StateSpaceProblem(f_lss!!, g_lss!!, u0_s, (0, T_rbc), p_s;
        n_shocks = 1, n_obs = 2, observables_noise = D_rbc,
        noise = nse_s, observables = observables_rbc),
    DirectIteration())

# Linear — static (for comparison)
const ws_lin_static = init(
    LinearStateSpaceProblem(A_s, B_s, u0_s, (0, T_rbc);
        C = C_s, observables_noise = D_rbc, noise = nse_s, observables = observables_rbc),
    DirectIteration())

# =============================================================================
# Scalable benchmark problems at various dimensions
# =============================================================================

function make_bench_data(; N, M, K, T, seed = 42)
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    C = 0.1 * randn(N, K)  # noise matrix (C in diff_econ notation)
    G = randn(M, N)         # observation matrix (G in diff_econ notation)
    x_0 = randn(N)
    w = [rand(K) for _ in 1:T]
    return (; A, C, G, x_0, w)
end

function make_bench_data_static(; N, M, K, T, seed = 42)
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = SMatrix{N, N}(0.5 * A_raw / maximum(abs.(eigvals(A_raw))))
    C = SMatrix{N, K}(0.1 * randn(N, K))
    G = SMatrix{M, N}(randn(M, N))
    x_0 = SVector{N}(randn(N))
    w = [SVector{K}(rand(K)) for _ in 1:T]
    return (; A, C, G, x_0, w)
end

# Callbacks using (A, C, G) field names for generic SSM
@inline function f_de!!(x_p, x, w, p, t)
    x_p = mul!!(x_p, p.A, x)
    return muladd!!(x_p, p.C, w)
end
@inline function g_de!!(y, x, p, t)
    return mul!!(y, p.G, x)
end

# Small (N=5, M=2, K=2, T=10) — mutable
const de_sm = make_bench_data(N = 5, M = 2, K = 2, T = 10)
const ws_de_sm = init(
    StateSpaceProblem(f_de!!, g_de!!, de_sm.x_0, (0, 10),
        (; A = de_sm.A, C = de_sm.C, G = de_sm.G);
        n_shocks = 2, n_obs = 2, noise = de_sm.w),
    DirectIteration())

# Small (N=5, M=2, K=2, T=10) — static
const de_sm_s = make_bench_data_static(N = 5, M = 2, K = 2, T = 10)
const ws_de_sm_s = init(
    StateSpaceProblem(f_de!!, g_de!!, de_sm_s.x_0, (0, 10),
        (; A = de_sm_s.A, C = de_sm_s.C, G = de_sm_s.G);
        n_shocks = 2, n_obs = 2, noise = de_sm_s.w),
    DirectIteration())

# Large (N=30, M=10, K=10, T=100) — mutable
const de_lg = make_bench_data(N = 30, M = 10, K = 10, T = 100)
const ws_de_lg = init(
    StateSpaceProblem(f_de!!, g_de!!, de_lg.x_0, (0, 100),
        (; A = de_lg.A, C = de_lg.C, G = de_lg.G);
        n_shocks = 10, n_obs = 10, noise = de_lg.w),
    DirectIteration())

# =============================================================================
# Warmup all solve! paths
# =============================================================================

for ws in (ws_lin_rbc_di, ws_lin_rbc_kf, ws_lin_rbc_sim,
        ws_lin_fvgq_di, ws_lin_fvgq_kf, ws_lin_fvgq_sim,
        ws_quad_rbc, ws_quad_fvgq,
        ws_gen_mut, ws_gen_static, ws_lin_static,
        ws_de_sm, ws_de_sm_s, ws_de_lg)
    solve!(ws)
    solve!(ws)
end

# =============================================================================
# Run benchmarks — solve! only (pre-allocated cache, no allocation overhead)
# =============================================================================

println("=" ^ 70)
println("LINEAR MODEL — RBC (2×2, T=$(T_rbc))")
println("=" ^ 70)

print("  DirectIteration:  ")
display(@benchmark solve!($ws_lin_rbc_di))

print("  KalmanFilter:     ")
display(@benchmark solve!($ws_lin_rbc_kf))

print("  simulate:         ")
display(@benchmark solve!($ws_lin_rbc_sim))

println()
println("=" ^ 70)
println("LINEAR MODEL — FVGQ ($(size(A_FVGQ,1))×$(size(A_FVGQ,2)), T=$(T_FVGQ))")
println("=" ^ 70)

print("  DirectIteration:  ")
display(@benchmark solve!($ws_lin_fvgq_di))

print("  KalmanFilter:     ")
display(@benchmark solve!($ws_lin_fvgq_kf))

print("  simulate:         ")
display(@benchmark solve!($ws_lin_fvgq_sim))

println()
println("=" ^ 70)
println("QUADRATIC (Generic) MODEL — RBC (2×2, T=$(length(observables_2_rbc)))")
println("=" ^ 70)

print("  DirectIteration:  ")
display(@benchmark solve!($ws_quad_rbc))

println()
println("=" ^ 70)
println("QUADRATIC (Generic) MODEL — FVGQ ($(size(A_1_FVGQ,1))×$(size(A_1_FVGQ,2)), T=$(length(observables_2_FVGQ)))")
println("=" ^ 70)

print("  DirectIteration:  ")
display(@benchmark solve!($ws_quad_fvgq))

println()
println("=" ^ 70)
println("GENERIC LINEAR — !! callbacks, RBC (2×2, T=$(T_rbc))")
println("=" ^ 70)

print("  LinearSSP mutable:   ")
display(@benchmark solve!($ws_lin_rbc_di))

print("  LinearSSP static:    ")
display(@benchmark solve!($ws_lin_static))

print("  GenericSSP mutable:  ")
display(@benchmark solve!($ws_gen_mut))

print("  GenericSSP static:   ")
display(@benchmark solve!($ws_gen_static))

println()
println("=" ^ 70)
println("SCALABLE — generic SSM at various dimensions")
println("=" ^ 70)

print("  small mutable (N=5, M=2, T=10):    ")
display(@benchmark solve!($ws_de_sm))

print("  small static  (N=5, M=2, T=10):    ")
display(@benchmark solve!($ws_de_sm_s))

print("  large mutable (N=30, M=10, T=100):  ")
display(@benchmark solve!($ws_de_lg))

println()
println("Done.")
