# StaticArrays primal performance via workspace pattern (init/solve!)
# Vector{SVector} workspace works because the solver uses reassignment:
#   u[t] = _transition!!(u[t], ...) — bang-bang returns new SVector, outer = replaces element
#
# Returns SA_BENCH BenchmarkGroup

using StaticArrays
using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, mul!!, muladd!!, fill_zero!!, StateSpaceWorkspace

const SA_BENCH = BenchmarkGroup()
SA_BENCH["linear"] = BenchmarkGroup()
SA_BENCH["generic"] = BenchmarkGroup()
SA_BENCH["kalman"] = BenchmarkGroup()

function bench_solve!(ws)
    solve!(ws)
    return nothing
end

# --- Linear DirectIteration N=2, T=20 ---

const A_sa_2 = @SMatrix [0.9 0.1; 0.0 0.8]
const B_sa_2 = @SMatrix [0.0; 0.1;;]
const C_sa_2 = @SMatrix [1.0 0.0; 0.0 1.0]
const u0_sa_2 = @SVector [0.5, 0.3]
const noise_sa_2 = [SVector{1}(randn()) for _ in 1:20]

const ws_ls2 = init(LinearStateSpaceProblem(A_sa_2, B_sa_2, u0_sa_2, (0, 20);
    C = C_sa_2, noise = noise_sa_2), DirectIteration())
SA_BENCH["linear"]["static_2x2"] = @benchmarkable bench_solve!($ws_ls2)

const ws_lm2 = init(LinearStateSpaceProblem(Matrix(A_sa_2), Matrix(B_sa_2), Vector(u0_sa_2), (0, 20);
    C = Matrix(C_sa_2), noise = [Vector(n) for n in noise_sa_2]), DirectIteration())
SA_BENCH["linear"]["mutable_2x2"] = @benchmarkable bench_solve!($ws_lm2)

# --- Linear DirectIteration N=5, T=50 ---

Random.seed!(123)
const A_sa_5_raw = randn(5, 5)
const A_sa_5 = SMatrix{5, 5}(0.5 * A_sa_5_raw / maximum(abs.(eigvals(A_sa_5_raw))))
const B_sa_5 = SMatrix{5, 2}(0.1 * randn(5, 2))
const C_sa_5 = SMatrix{3, 5}(randn(3, 5))
const u0_sa_5 = SVector{5}(zeros(5))
const noise_sa_5 = [SVector{2}(randn(2)) for _ in 1:50]

const ws_ls5 = init(LinearStateSpaceProblem(A_sa_5, B_sa_5, u0_sa_5, (0, 50);
    C = C_sa_5, noise = noise_sa_5), DirectIteration())
SA_BENCH["linear"]["static_5x5"] = @benchmarkable bench_solve!($ws_ls5)

const ws_lm5 = init(LinearStateSpaceProblem(Matrix(A_sa_5), Matrix(B_sa_5), Vector(u0_sa_5), (0, 50);
    C = Matrix(C_sa_5), noise = [Vector(n) for n in noise_sa_5]), DirectIteration())
SA_BENCH["linear"]["mutable_5x5"] = @benchmarkable bench_solve!($ws_lm5)

# --- Generic !! callbacks ---

@inline function f_lss_sa!!(x_p, x, w, p, t)
    x_p = mul!!(x_p, p.A, x)
    return muladd!!(x_p, p.B, w)
end

@inline function g_lss_sa!!(y, x, p, t)
    return mul!!(y, p.C, x)
end

# --- Generic N=2, T=20 ---

const p_gen_s2 = (; A = A_sa_2, B = B_sa_2, C = C_sa_2)
const ws_gs2 = init(StateSpaceProblem(f_lss_sa!!, g_lss_sa!!, u0_sa_2, (0, 20), p_gen_s2;
    n_shocks = 1, n_obs = 2, noise = noise_sa_2), DirectIteration())
SA_BENCH["generic"]["static_2x2"] = @benchmarkable bench_solve!($ws_gs2)

const p_gen_m2 = (; A = Matrix(A_sa_2), B = Matrix(B_sa_2), C = Matrix(C_sa_2))
const ws_gm2 = init(StateSpaceProblem(f_lss_sa!!, g_lss_sa!!, Vector(u0_sa_2), (0, 20), p_gen_m2;
    n_shocks = 1, n_obs = 2, noise = [Vector(n) for n in noise_sa_2]), DirectIteration())
SA_BENCH["generic"]["mutable_2x2"] = @benchmarkable bench_solve!($ws_gm2)

# --- Generic N=5, T=50 ---

const p_gen_s5 = (; A = A_sa_5, B = B_sa_5, C = C_sa_5)
const ws_gs5 = init(StateSpaceProblem(f_lss_sa!!, g_lss_sa!!, u0_sa_5, (0, 50), p_gen_s5;
    n_shocks = 2, n_obs = 3, noise = noise_sa_5), DirectIteration())
SA_BENCH["generic"]["static_5x5"] = @benchmarkable bench_solve!($ws_gs5)

const p_gen_m5 = (; A = Matrix(A_sa_5), B = Matrix(B_sa_5), C = Matrix(C_sa_5))
const ws_gm5 = init(StateSpaceProblem(f_lss_sa!!, g_lss_sa!!, Vector(u0_sa_5), (0, 50), p_gen_m5;
    n_shocks = 2, n_obs = 3, noise = [Vector(n) for n in noise_sa_5]), DirectIteration())
SA_BENCH["generic"]["mutable_5x5"] = @benchmarkable bench_solve!($ws_gm5)

# --- Kalman filter N=3, M=2, T=10 ---

Random.seed!(789)
const A_kf_3_raw = randn(3, 3)
const A_kf_3 = SMatrix{3, 3}(0.5 * A_kf_3_raw / maximum(abs.(eigvals(A_kf_3_raw))))
const B_kf_3 = SMatrix{3, 2}(0.1 * randn(3, 2))
const C_kf_3 = SMatrix{2, 3}(randn(2, 3))
const R_kf_3 = SMatrix{2, 2}(0.01 * I(2))
const mu0_kf_3 = SVector{3}(zeros(3))
const Sig0_kf_3 = SMatrix{3, 3}(1.0 * I(3))

# Generate observations for Kalman
const noise_kf_3 = [SVector{2}(randn(2)) for _ in 1:10]
const sim_kf_3 = solve(LinearStateSpaceProblem(A_kf_3, B_kf_3, mu0_kf_3, (0, 10);
    C = C_kf_3, noise = noise_kf_3))
const y_kf_3 = [sim_kf_3.z[t + 1] + SVector{2}(0.1 * randn(2)) for t in 1:10]

const ws_ks3 = init(LinearStateSpaceProblem(A_kf_3, B_kf_3, mu0_kf_3, (0, 10);
    C = C_kf_3, u0_prior_mean = mu0_kf_3, u0_prior_var = Sig0_kf_3,
    observables_noise = R_kf_3, observables = y_kf_3), KalmanFilter())
SA_BENCH["kalman"]["static_3x3"] = @benchmarkable bench_solve!($ws_ks3)

const ws_km3 = init(LinearStateSpaceProblem(Matrix(A_kf_3), Matrix(B_kf_3), Vector(mu0_kf_3), (0, 10);
    C = Matrix(C_kf_3), u0_prior_mean = Vector(mu0_kf_3), u0_prior_var = Matrix(Sig0_kf_3),
    observables_noise = Matrix(R_kf_3), observables = [Vector(y) for y in y_kf_3]), KalmanFilter())
SA_BENCH["kalman"]["mutable_3x3"] = @benchmarkable bench_solve!($ws_km3)

# --- Kalman filter N=5, M=3, T=20 ---

Random.seed!(101)
const A_kf_5_raw = randn(5, 5)
const A_kf_5 = SMatrix{5, 5}(0.5 * A_kf_5_raw / maximum(abs.(eigvals(A_kf_5_raw))))
const B_kf_5 = SMatrix{5, 2}(0.1 * randn(5, 2))
const C_kf_5 = SMatrix{3, 5}(randn(3, 5))
const R_kf_5 = SMatrix{3, 3}(0.01 * I(3))
const mu0_kf_5 = SVector{5}(zeros(5))
const Sig0_kf_5 = SMatrix{5, 5}(1.0 * I(5))

const noise_kf_5 = [SVector{2}(randn(2)) for _ in 1:20]
const sim_kf_5 = solve(LinearStateSpaceProblem(A_kf_5, B_kf_5, mu0_kf_5, (0, 20);
    C = C_kf_5, noise = noise_kf_5))
const y_kf_5 = [sim_kf_5.z[t + 1] + SVector{3}(0.1 * randn(3)) for t in 1:20]

const ws_ks5 = init(LinearStateSpaceProblem(A_kf_5, B_kf_5, mu0_kf_5, (0, 20);
    C = C_kf_5, u0_prior_mean = mu0_kf_5, u0_prior_var = Sig0_kf_5,
    observables_noise = R_kf_5, observables = y_kf_5), KalmanFilter())
SA_BENCH["kalman"]["static_5x5"] = @benchmarkable bench_solve!($ws_ks5)

const ws_km5 = init(LinearStateSpaceProblem(Matrix(A_kf_5), Matrix(B_kf_5), Vector(mu0_kf_5), (0, 20);
    C = Matrix(C_kf_5), u0_prior_mean = Vector(mu0_kf_5), u0_prior_var = Matrix(Sig0_kf_5),
    observables_noise = Matrix(R_kf_5), observables = [Vector(y) for y in y_kf_5]), KalmanFilter())
SA_BENCH["kalman"]["mutable_5x5"] = @benchmarkable bench_solve!($ws_km5)

# --- Quadratic PrunedQuadraticStateSpaceProblem (pruned, using new types) ---

SA_BENCH["quadratic"] = BenchmarkGroup()

# N=2, K=1, M=2, T=10
Random.seed!(42)
const A_2_q = 0.01 * randn(2, 2, 2)
const C_2_q = 0.01 * randn(2, 2, 2)
const noise_q = [randn() for _ in 1:10]

const As1 = @SMatrix [0.3 0.1; -0.1 0.3]
const As0 = @SVector [0.001, -0.001]
const Bs = @SMatrix [0.1; 0.0;;]
const Cs0 = @SVector [0.001, -0.001]
const Cs1 = @SMatrix [1.0 0.0; 0.0 1.0]
const u0s = @SVector zeros(2)
const noise_s = [SVector{1}(n) for n in noise_q]

# Static 2x2 (pruned quadratic)
const prob_qs = PrunedQuadraticStateSpaceProblem(As0, As1, A_2_q, Bs, u0s, (0, 10);
    C_0 = Cs0, C_1 = Cs1, C_2 = C_2_q, noise = noise_s)
const ws_qs = init(prob_qs, DirectIteration())
SA_BENCH["quadratic"]["static_2x2"] = @benchmarkable bench_solve!($ws_qs)

# Mutable 2x2 (pruned quadratic)
const prob_qm = PrunedQuadraticStateSpaceProblem(
    Vector(As0), Matrix(As1), copy(A_2_q), Matrix(Bs),
    Vector(u0s), (0, 10);
    C_0 = Vector(Cs0), C_1 = Matrix(Cs1), C_2 = copy(C_2_q),
    noise = [Vector(n) for n in noise_s])
const ws_qm = init(prob_qm, DirectIteration())
SA_BENCH["quadratic"]["mutable_2x2"] = @benchmarkable bench_solve!($ws_qm)

# =============================================================================
# AD benchmarks for Linear 2x2 (static and mutable)
# =============================================================================

SA_BENCH["linear"]["forward"] = BenchmarkGroup()
SA_BENCH["linear"]["reverse"] = BenchmarkGroup()

function sim_fwd_sa!(A, B, C, u0, noise, sol_out, cache)
    prob = LinearStateSpaceProblem(A, B, u0, (0, length(noise)); C, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.z[end])
end

function sim_rev_sa!(A, B, C, u0, noise, sol_out, cache)::Float64
    prob = LinearStateSpaceProblem(A, B, u0, (0, length(noise)); C, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return sum(solve!(ws).u[end])
end

function forward_sa!(A, B, C, u0, noise, sol_out, cache,
        dA, dB, dC, du0, dnoise, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA = fill_zero!!(dA); dB = fill_zero!!(dB); dC = fill_zero!!(dC); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end
    if ismutable(dA); dA[1,1] = 1.0; else; dA = setindex(dA, 1.0, 1, 1); end
    autodiff(Forward, sim_fwd_sa!,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

function reverse_sa!(A, B, C, u0, noise, sol_out, cache,
        dA, dB, dC, du0, dnoise, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA = fill_zero!!(dA); dB = fill_zero!!(dB); dC = fill_zero!!(dC); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end
    autodiff(Reverse, sim_rev_sa!, Active,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

# --- Static 2x2 AD shadows ---

const dA_s2 = make_zero(A_sa_2)
const dB_s2 = make_zero(B_sa_2)
const dC_s2 = make_zero(C_sa_2)
const du0_s2 = make_zero(u0_sa_2)
const dnoise_s2 = [make_zero(noise_sa_2[1]) for _ in 1:20]
const dsol_s2 = make_zero(ws_ls2.output)
const dcache_s2 = make_zero(ws_ls2.cache)

# --- Mutable 2x2 AD shadows ---

const A_m2 = Matrix(A_sa_2)
const B_m2 = Matrix(B_sa_2)
const C_m2 = Matrix(C_sa_2)
const u0_m2 = Vector(u0_sa_2)
const noise_m2 = [Vector(n) for n in noise_sa_2]
const dA_m2 = make_zero(A_m2)
const dB_m2 = make_zero(B_m2)
const dC_m2 = make_zero(C_m2)
const du0_m2 = make_zero(u0_m2)
const dnoise_m2 = [make_zero(noise_m2[1]) for _ in 1:20]
const dsol_m2 = make_zero(ws_lm2.output)
const dcache_m2 = make_zero(ws_lm2.cache)

# --- Warmups ---

forward_sa!(A_sa_2, B_sa_2, C_sa_2, u0_sa_2, noise_sa_2,
    ws_ls2.output, ws_ls2.cache,
    dA_s2, dB_s2, dC_s2, du0_s2, dnoise_s2, dsol_s2, dcache_s2)

forward_sa!(A_m2, B_m2, C_m2, u0_m2, noise_m2,
    ws_lm2.output, ws_lm2.cache,
    dA_m2, dB_m2, dC_m2, du0_m2, dnoise_m2, dsol_m2, dcache_m2)

reverse_sa!(A_sa_2, B_sa_2, C_sa_2, u0_sa_2, noise_sa_2,
    ws_ls2.output, ws_ls2.cache,
    dA_s2, dB_s2, dC_s2, du0_s2, dnoise_s2, dsol_s2, dcache_s2)

reverse_sa!(A_m2, B_m2, C_m2, u0_m2, noise_m2,
    ws_lm2.output, ws_lm2.cache,
    dA_m2, dB_m2, dC_m2, du0_m2, dnoise_m2, dsol_m2, dcache_m2)

# --- Benchmarkables ---

SA_BENCH["linear"]["forward"]["static_2x2"] = @benchmarkable forward_sa!(
    $A_sa_2, $B_sa_2, $C_sa_2, $u0_sa_2, $noise_sa_2,
    $(ws_ls2.output), $(ws_ls2.cache),
    $dA_s2, $dB_s2, $dC_s2, $du0_s2, $dnoise_s2, $dsol_s2, $dcache_s2)

SA_BENCH["linear"]["forward"]["mutable_2x2"] = @benchmarkable forward_sa!(
    $A_m2, $B_m2, $C_m2, $u0_m2, $noise_m2,
    $(ws_lm2.output), $(ws_lm2.cache),
    $dA_m2, $dB_m2, $dC_m2, $du0_m2, $dnoise_m2, $dsol_m2, $dcache_m2)

SA_BENCH["linear"]["reverse"]["static_2x2"] = @benchmarkable reverse_sa!(
    $A_sa_2, $B_sa_2, $C_sa_2, $u0_sa_2, $noise_sa_2,
    $(ws_ls2.output), $(ws_ls2.cache),
    $dA_s2, $dB_s2, $dC_s2, $du0_s2, $dnoise_s2, $dsol_s2, $dcache_s2)

SA_BENCH["linear"]["reverse"]["mutable_2x2"] = @benchmarkable reverse_sa!(
    $A_m2, $B_m2, $C_m2, $u0_m2, $noise_m2,
    $(ws_lm2.output), $(ws_lm2.cache),
    $dA_m2, $dB_m2, $dC_m2, $du0_m2, $dnoise_m2, $dsol_m2, $dcache_m2)

# =============================================================================
# AD benchmarks for Quadratic 2x2 (static and mutable) — PrunedQuadraticStateSpaceProblem
# =============================================================================

SA_BENCH["quadratic"]["forward"] = BenchmarkGroup()
SA_BENCH["quadratic"]["reverse"] = BenchmarkGroup()

# --- Inner wrappers: construct prob inside (correct Enzyme pattern) ---

function quad_fwd_sa!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache)
    prob = PrunedQuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.z[end])
end

function quad_rev_sa!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache)::Float64
    prob = PrunedQuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, (0, length(noise));
        C_0, C_1, C_2, noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return sum(solve!(ws).u[end])
end

# --- Outer bench functions: zero shadows, call autodiff ---

function forward_quad_sa!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); make_zero!(dA_2)
    dB = fill_zero!!(dB); dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1)
    make_zero!(dC_2); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end
    if ismutable(dA_1); dA_1[1,1] = 1.0; else; dA_1 = setindex(dA_1, 1.0, 1, 1); end

    autodiff(Forward, quad_fwd_sa!,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

function reverse_quad_sa!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); make_zero!(dA_2)
    dB = fill_zero!!(dB); dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1)
    make_zero!(dC_2); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end

    autodiff(Reverse, quad_rev_sa!, Active,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

# --- Static quadratic AD shadows ---

const dAs0 = make_zero(As0); const dAs1 = make_zero(As1)
const dA_2_qs = make_zero(A_2_q); const dBs = make_zero(Bs)
const dCs0 = make_zero(Cs0); const dCs1 = make_zero(Cs1)
const dC_2_qs = make_zero(C_2_q); const du0s = make_zero(u0s)
const dnoise_qs = [make_zero(noise_s[1]) for _ in 1:10]
const dsol_qs = make_zero(ws_qs.output); const dcache_qs = make_zero(ws_qs.cache)

# --- Mutable quadratic AD shadows ---

const A_0_qm_ad = Vector(As0); const A_1_qm_ad = Matrix(As1)
const A_2_qm_ad = copy(A_2_q); const B_qm_ad = Matrix(Bs)
const C_0_qm_ad = Vector(Cs0); const C_1_qm_ad = Matrix(Cs1)
const C_2_qm_ad = copy(C_2_q); const u0_qm_ad = Vector(u0s)
const noise_qm_ad = [Vector(n) for n in noise_s]
const dA_0_qm = make_zero(A_0_qm_ad); const dA_1_qm = make_zero(A_1_qm_ad)
const dA_2_qm = make_zero(A_2_qm_ad); const dB_qm_ad = make_zero(B_qm_ad)
const dC_0_qm = make_zero(C_0_qm_ad); const dC_1_qm = make_zero(C_1_qm_ad)
const dC_2_qm = make_zero(C_2_qm_ad); const du0_qm_ad = make_zero(u0_qm_ad)
const dnoise_qm_ad = [make_zero(noise_qm_ad[1]) for _ in 1:10]
const dsol_qm = make_zero(ws_qm.output); const dcache_qm = make_zero(ws_qm.cache)

# --- Quadratic warmups ---

forward_quad_sa!(As0, As1, A_2_q, Bs, Cs0, Cs1, C_2_q,
    u0s, noise_s, ws_qs.output, ws_qs.cache,
    dAs0, dAs1, dA_2_qs, dBs, dCs0, dCs1, dC_2_qs,
    du0s, dnoise_qs, dsol_qs, dcache_qs)

forward_quad_sa!(A_0_qm_ad, A_1_qm_ad, A_2_qm_ad, B_qm_ad, C_0_qm_ad, C_1_qm_ad, C_2_qm_ad,
    u0_qm_ad, noise_qm_ad, ws_qm.output, ws_qm.cache,
    dA_0_qm, dA_1_qm, dA_2_qm, dB_qm_ad, dC_0_qm, dC_1_qm, dC_2_qm,
    du0_qm_ad, dnoise_qm_ad, dsol_qm, dcache_qm)

reverse_quad_sa!(As0, As1, A_2_q, Bs, Cs0, Cs1, C_2_q,
    u0s, noise_s, ws_qs.output, ws_qs.cache,
    dAs0, dAs1, dA_2_qs, dBs, dCs0, dCs1, dC_2_qs,
    du0s, dnoise_qs, dsol_qs, dcache_qs)

reverse_quad_sa!(A_0_qm_ad, A_1_qm_ad, A_2_qm_ad, B_qm_ad, C_0_qm_ad, C_1_qm_ad, C_2_qm_ad,
    u0_qm_ad, noise_qm_ad, ws_qm.output, ws_qm.cache,
    dA_0_qm, dA_1_qm, dA_2_qm, dB_qm_ad, dC_0_qm, dC_1_qm, dC_2_qm,
    du0_qm_ad, dnoise_qm_ad, dsol_qm, dcache_qm)

# --- Quadratic benchmarkables ---

SA_BENCH["quadratic"]["forward"]["static_2x2"] = @benchmarkable forward_quad_sa!(
    $As0, $As1, $A_2_q, $Bs, $Cs0, $Cs1, $C_2_q,
    $u0s, $noise_s, $(ws_qs.output), $(ws_qs.cache),
    $dAs0, $dAs1, $dA_2_qs, $dBs, $dCs0, $dCs1, $dC_2_qs,
    $du0s, $dnoise_qs, $dsol_qs, $dcache_qs)

SA_BENCH["quadratic"]["forward"]["mutable_2x2"] = @benchmarkable forward_quad_sa!(
    $A_0_qm_ad, $A_1_qm_ad, $A_2_qm_ad, $B_qm_ad, $C_0_qm_ad, $C_1_qm_ad, $C_2_qm_ad,
    $u0_qm_ad, $noise_qm_ad, $(ws_qm.output), $(ws_qm.cache),
    $dA_0_qm, $dA_1_qm, $dA_2_qm, $dB_qm_ad, $dC_0_qm, $dC_1_qm, $dC_2_qm,
    $du0_qm_ad, $dnoise_qm_ad, $dsol_qm, $dcache_qm)

SA_BENCH["quadratic"]["reverse"]["static_2x2"] = @benchmarkable reverse_quad_sa!(
    $As0, $As1, $A_2_q, $Bs, $Cs0, $Cs1, $C_2_q,
    $u0s, $noise_s, $(ws_qs.output), $(ws_qs.cache),
    $dAs0, $dAs1, $dA_2_qs, $dBs, $dCs0, $dCs1, $dC_2_qs,
    $du0s, $dnoise_qs, $dsol_qs, $dcache_qs)

SA_BENCH["quadratic"]["reverse"]["mutable_2x2"] = @benchmarkable reverse_quad_sa!(
    $A_0_qm_ad, $A_1_qm_ad, $A_2_qm_ad, $B_qm_ad, $C_0_qm_ad, $C_1_qm_ad, $C_2_qm_ad,
    $u0_qm_ad, $noise_qm_ad, $(ws_qm.output), $(ws_qm.cache),
    $dA_0_qm, $dA_1_qm, $dA_2_qm, $dB_qm_ad, $dC_0_qm, $dC_1_qm, $dC_2_qm,
    $du0_qm_ad, $dnoise_qm_ad, $dsol_qm, $dcache_qm)

SA_BENCH
