# Enzyme AD benchmarks for generic StateSpaceProblem with quadratic callbacks
# Standard formulation: matrices passed through p (no closures), bang-bang operators
# Returns QUAD_ENZYME BenchmarkGroup

using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, StateSpaceWorkspace, mul!!, muladd!!, copyto!!, fill_zero!!

const QUAD_ENZYME = BenchmarkGroup()
QUAD_ENZYME["simulation"] = BenchmarkGroup()
QUAD_ENZYME["simulation"]["raw"] = BenchmarkGroup()
QUAD_ENZYME["simulation"]["forward"] = BenchmarkGroup()
QUAD_ENZYME["simulation"]["reverse"] = BenchmarkGroup()
QUAD_ENZYME["likelihood"] = BenchmarkGroup()
QUAD_ENZYME["likelihood"]["raw"] = BenchmarkGroup()
QUAD_ENZYME["likelihood"]["forward"] = BenchmarkGroup()
QUAD_ENZYME["likelihood"]["reverse"] = BenchmarkGroup()

# --- Problem sizes ---

const p_quad_small = (; N = 2, K = 1, M = 2, T = 10)
const p_quad_large = (; N = 10, K = 3, M = 6, T = 50)

# --- Quadratic callbacks (standard formulation, bang-bang) ---

function quad_f!!(x_next, x, w, p, t)
    (; A_0, A_1, A_2, B, u_f, u_f_new) = p
    n_x = length(x)
    u_f_new = mul!!(u_f_new, A_1, u_f)
    u_f_new = muladd!!(u_f_new, B, w)
    x_next = copyto!!(x_next, A_0)
    x_next = mul!!(x_next, A_1, x, 1.0, 1.0)
    @inbounds for i in 1:n_x
        x_next[i] += dot(u_f, view(A_2, i, :, :), u_f)
    end
    x_next = muladd!!(x_next, B, w)
    copyto!!(u_f, u_f_new)
    return x_next
end

function quad_g!!(y, x, p, t)
    (; C_0, C_1, C_2, u_f) = p
    n_obs = length(C_0)
    y = copyto!!(y, C_0)
    y = mul!!(y, C_1, x, 1.0, 1.0)
    @inbounds for i in 1:n_obs
        y[i] += dot(u_f, view(C_2, i, :, :), u_f)
    end
    return y
end

# Transition-only version for no-observation benchmarks
function quad_f_only!!(x_next, x, w, p, t)
    return quad_f!!(x_next, x, w, p, t)
end

# --- Matrix generation ---

function make_quadratic_matrices(N, K, M; seed = 42)
    Random.seed!(seed)
    A_1_raw = randn(N, N)
    A_1 = 0.5 * A_1_raw / maximum(abs.(eigvals(A_1_raw)))
    A_0 = 0.001 * randn(N)
    A_2 = 0.01 * randn(N, N, N) / N
    B = 0.1 * randn(N, K)
    C_0 = 0.001 * randn(M)
    C_1 = randn(M, N)
    C_2 = 0.01 * randn(M, N, N) / N
    return (; A_0, A_1, A_2, B, C_0, C_1, C_2)
end

function make_quad_params(mats, u0)
    return (; mats..., u_f = copy(u0), u_f_new = similar(u0))
end

# --- Problem setup ---

function make_quad_benchmark(psz; seed = 42)
    (; N, K, M, T) = psz
    mats = make_quadratic_matrices(N, K, M; seed)
    u0 = zeros(N)

    # Simulation problem (no observables)
    p_sim = make_quad_params(mats, u0)
    prob_sim = StateSpaceProblem(quad_f!!, quad_g!!, u0, (0, T), p_sim;
        n_shocks = K, n_obs = M)
    Random.seed!(seed + 1)
    noise = [randn(K) for _ in 1:T]
    prob_sim = remake(prob_sim; noise)
    ws_sim = init(prob_sim, DirectIteration())

    # Likelihood problem (observables + obs_noise)
    sim_sol = solve(prob_sim)
    Random.seed!(seed + 2)
    H = 0.1 * randn(M, M)
    R = H * H'
    y = [sim_sol.z[t + 1] + H * randn(M) for t in 1:T]
    p_lik = make_quad_params(mats, u0)
    prob_lik = StateSpaceProblem(quad_f!!, quad_g!!, u0, (0, T), p_lik;
        n_shocks = K, n_obs = M, noise, observables = y, observables_noise = R)
    ws_lik = init(prob_lik, DirectIteration())

    # Shadow copies for AD
    dA_0 = make_zero(mats.A_0); dA_1 = make_zero(mats.A_1)
    dA_2 = make_zero(mats.A_2); dB = make_zero(mats.B)
    dC_0 = make_zero(mats.C_0); dC_1 = make_zero(mats.C_1)
    dC_2 = make_zero(mats.C_2); du0 = make_zero(u0)
    du_f = make_zero(u0); du_f_new = make_zero(u0)
    dnoise = [make_zero(noise[1]) for _ in 1:T]
    dH = make_zero(H); dy = [make_zero(y[1]) for _ in 1:T]
    dsol_sim = make_zero(ws_sim.output)
    dcache_sim = make_zero(ws_sim.cache)
    dsol_lik = make_zero(ws_lik.output)
    dcache_lik = make_zero(ws_lik.cache)

    return (; mats, u0, noise, y, H, R, T,
        prob_sim, sol_sim = ws_sim.output, cache_sim = ws_sim.cache,
        prob_lik, sol_lik = ws_lik.output, cache_lik = ws_lik.cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, du_f, du_f_new, dnoise,
        dH, dy, dsol_sim, dcache_sim, dsol_lik, dcache_lik)
end

const quad_s = make_quad_benchmark(p_quad_small)
const quad_l = make_quad_benchmark(p_quad_large)

# --- Raw benchmarks ---

function raw_quad!(prob, sol_out, cache)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return nothing
end

raw_quad!(quad_s.prob_sim, quad_s.sol_sim, quad_s.cache_sim)
raw_quad!(quad_l.prob_sim, quad_l.sol_sim, quad_l.cache_sim)
raw_quad!(quad_s.prob_lik, quad_s.sol_lik, quad_s.cache_lik)
raw_quad!(quad_l.prob_lik, quad_l.sol_lik, quad_l.cache_lik)

QUAD_ENZYME["simulation"]["raw"]["small_mutable"] = @benchmarkable raw_quad!(
    $(quad_s.prob_sim), $(quad_s.sol_sim), $(quad_s.cache_sim))
QUAD_ENZYME["simulation"]["raw"]["large_mutable"] = @benchmarkable raw_quad!(
    $(quad_l.prob_sim), $(quad_l.sol_sim), $(quad_l.cache_sim))
QUAD_ENZYME["likelihood"]["raw"]["small_mutable"] = @benchmarkable raw_quad!(
    $(quad_s.prob_lik), $(quad_s.sol_lik), $(quad_s.cache_lik))
QUAD_ENZYME["likelihood"]["raw"]["large_mutable"] = @benchmarkable raw_quad!(
    $(quad_l.prob_lik), $(quad_l.sol_lik), $(quad_l.cache_lik))

# --- Edge cases (raw only, small) ---

QUAD_ENZYME["simulation"]["raw"]["no_obs_small"] = let
    p_no = make_quad_params(quad_s.mats, quad_s.u0)
    prob = StateSpaceProblem(quad_f_only!!, nothing, quad_s.u0, (0, p_quad_small.T), p_no;
        n_shocks = p_quad_small.K, n_obs = 0, noise = quad_s.noise)
    ws = init(prob, DirectIteration())
    @benchmarkable raw_quad!($prob, $(ws.output), $(ws.cache))
end

QUAD_ENZYME["simulation"]["raw"]["no_noise_small"] = let
    p_nn = make_quad_params(quad_s.mats, quad_s.u0)
    prob = StateSpaceProblem(quad_f!!, quad_g!!, quad_s.u0, (0, p_quad_small.T), p_nn;
        n_shocks = 0, n_obs = p_quad_small.M)
    ws = init(prob, DirectIteration())
    @benchmarkable raw_quad!($prob, $(ws.output), $(ws.cache))
end

# --- AD wrappers (standard formulation: all matrices as separate Duplicated args) ---

function quad_sim_fwd_inner!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise,
        u_f, u_f_new, sol_out, cache)
    p = (; A_0, A_1, A_2, B, C_0, C_1, C_2, u_f, u_f_new)
    prob = StateSpaceProblem(quad_f!!, quad_g!!, u0, (0, length(noise)), p;
        n_shocks = size(B, 2), n_obs = length(C_0), noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.z[end])
end

function quad_sim_rev_inner!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise,
        u_f, u_f_new, sol_out, cache)::Float64
    p = (; A_0, A_1, A_2, B, C_0, C_1, C_2, u_f, u_f_new)
    prob = StateSpaceProblem(quad_f!!, quad_g!!, u0, (0, length(noise)), p;
        n_shocks = size(B, 2), n_obs = length(C_0), noise)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return sum(sol_out.u[end])
end

function quad_lik_fwd_inner!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, y, H,
        u_f, u_f_new, sol_out, cache)
    R = H * H'
    p = (; A_0, A_1, A_2, B, C_0, C_1, C_2, u_f, u_f_new)
    prob = StateSpaceProblem(quad_f!!, quad_g!!, u0, (0, length(noise)), p;
        n_shocks = size(B, 2), n_obs = length(C_0),
        noise, observables = y, observables_noise = R)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.z[end])
end

function quad_lik_rev_inner!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, y, H,
        u_f, u_f_new, sol_out, cache)::Float64
    R = H * H'
    p = (; A_0, A_1, A_2, B, C_0, C_1, C_2, u_f, u_f_new)
    prob = StateSpaceProblem(quad_f!!, quad_g!!, u0, (0, length(noise)), p;
        n_shocks = size(B, 2), n_obs = length(C_0),
        noise, observables = y, observables_noise = R)
    ws = StateSpaceWorkspace(prob, DirectIteration(), sol_out, cache)
    return solve!(ws).logpdf
end

# --- Forward mode: simulation ---

function forward_quad_sim!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise,
        u_f, u_f_new, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise,
        du_f, du_f_new, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); dA_2 = fill_zero!!(dA_2); dB = fill_zero!!(dB)
    dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1); dC_2 = fill_zero!!(dC_2); du0 = fill_zero!!(du0)
    du_f = fill_zero!!(du_f); du_f_new = fill_zero!!(du_f_new)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end
    dA_1[1, 1] = 1.0

    autodiff(Forward, quad_sim_fwd_inner!,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(u_f, du_f), Duplicated(u_f_new, du_f_new),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

# Warmup + benchmarkable
forward_quad_sim!(
    copy(quad_s.mats.A_0), copy(quad_s.mats.A_1), copy(quad_s.mats.A_2),
    copy(quad_s.mats.B), copy(quad_s.mats.C_0), copy(quad_s.mats.C_1),
    copy(quad_s.mats.C_2), copy(quad_s.u0), [copy(n) for n in quad_s.noise],
    copy(quad_s.u0), similar(quad_s.u0),
    quad_s.sol_sim, quad_s.cache_sim,
    quad_s.dA_0, quad_s.dA_1, quad_s.dA_2, quad_s.dB,
    quad_s.dC_0, quad_s.dC_1, quad_s.dC_2, quad_s.du0, quad_s.dnoise,
    quad_s.du_f, quad_s.du_f_new, quad_s.dsol_sim, quad_s.dcache_sim)

QUAD_ENZYME["simulation"]["forward"]["small_mutable"] = @benchmarkable forward_quad_sim!(
    $(copy(quad_s.mats.A_0)), $(copy(quad_s.mats.A_1)), $(copy(quad_s.mats.A_2)),
    $(copy(quad_s.mats.B)), $(copy(quad_s.mats.C_0)), $(copy(quad_s.mats.C_1)),
    $(copy(quad_s.mats.C_2)), $(copy(quad_s.u0)), $([copy(n) for n in quad_s.noise]),
    $(copy(quad_s.u0)), $(similar(quad_s.u0)),
    $(quad_s.sol_sim), $(quad_s.cache_sim),
    $(quad_s.dA_0), $(quad_s.dA_1), $(quad_s.dA_2), $(quad_s.dB),
    $(quad_s.dC_0), $(quad_s.dC_1), $(quad_s.dC_2), $(quad_s.du0), $(quad_s.dnoise),
    $(quad_s.du_f), $(quad_s.du_f_new), $(quad_s.dsol_sim), $(quad_s.dcache_sim))

forward_quad_sim!(
    copy(quad_l.mats.A_0), copy(quad_l.mats.A_1), copy(quad_l.mats.A_2),
    copy(quad_l.mats.B), copy(quad_l.mats.C_0), copy(quad_l.mats.C_1),
    copy(quad_l.mats.C_2), copy(quad_l.u0), [copy(n) for n in quad_l.noise],
    copy(quad_l.u0), similar(quad_l.u0),
    quad_l.sol_sim, quad_l.cache_sim,
    quad_l.dA_0, quad_l.dA_1, quad_l.dA_2, quad_l.dB,
    quad_l.dC_0, quad_l.dC_1, quad_l.dC_2, quad_l.du0, quad_l.dnoise,
    quad_l.du_f, quad_l.du_f_new, quad_l.dsol_sim, quad_l.dcache_sim)

QUAD_ENZYME["simulation"]["forward"]["large_mutable"] = @benchmarkable forward_quad_sim!(
    $(copy(quad_l.mats.A_0)), $(copy(quad_l.mats.A_1)), $(copy(quad_l.mats.A_2)),
    $(copy(quad_l.mats.B)), $(copy(quad_l.mats.C_0)), $(copy(quad_l.mats.C_1)),
    $(copy(quad_l.mats.C_2)), $(copy(quad_l.u0)), $([copy(n) for n in quad_l.noise]),
    $(copy(quad_l.u0)), $(similar(quad_l.u0)),
    $(quad_l.sol_sim), $(quad_l.cache_sim),
    $(quad_l.dA_0), $(quad_l.dA_1), $(quad_l.dA_2), $(quad_l.dB),
    $(quad_l.dC_0), $(quad_l.dC_1), $(quad_l.dC_2), $(quad_l.du0), $(quad_l.dnoise),
    $(quad_l.du_f), $(quad_l.du_f_new), $(quad_l.dsol_sim), $(quad_l.dcache_sim))

# --- Reverse mode: simulation ---

function reverse_quad_sim!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise,
        u_f, u_f_new, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise,
        du_f, du_f_new, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); dA_2 = fill_zero!!(dA_2); dB = fill_zero!!(dB)
    dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1); dC_2 = fill_zero!!(dC_2); du0 = fill_zero!!(du0)
    du_f = fill_zero!!(du_f); du_f_new = fill_zero!!(du_f_new)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end

    autodiff(Reverse, quad_sim_rev_inner!, Active,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(u_f, du_f), Duplicated(u_f_new, du_f_new),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

reverse_quad_sim!(
    copy(quad_s.mats.A_0), copy(quad_s.mats.A_1), copy(quad_s.mats.A_2),
    copy(quad_s.mats.B), copy(quad_s.mats.C_0), copy(quad_s.mats.C_1),
    copy(quad_s.mats.C_2), copy(quad_s.u0), [copy(n) for n in quad_s.noise],
    copy(quad_s.u0), similar(quad_s.u0),
    quad_s.sol_sim, quad_s.cache_sim,
    quad_s.dA_0, quad_s.dA_1, quad_s.dA_2, quad_s.dB,
    quad_s.dC_0, quad_s.dC_1, quad_s.dC_2, quad_s.du0, quad_s.dnoise,
    quad_s.du_f, quad_s.du_f_new, quad_s.dsol_sim, quad_s.dcache_sim)

QUAD_ENZYME["simulation"]["reverse"]["small_mutable"] = @benchmarkable reverse_quad_sim!(
    $(copy(quad_s.mats.A_0)), $(copy(quad_s.mats.A_1)), $(copy(quad_s.mats.A_2)),
    $(copy(quad_s.mats.B)), $(copy(quad_s.mats.C_0)), $(copy(quad_s.mats.C_1)),
    $(copy(quad_s.mats.C_2)), $(copy(quad_s.u0)), $([copy(n) for n in quad_s.noise]),
    $(copy(quad_s.u0)), $(similar(quad_s.u0)),
    $(quad_s.sol_sim), $(quad_s.cache_sim),
    $(quad_s.dA_0), $(quad_s.dA_1), $(quad_s.dA_2), $(quad_s.dB),
    $(quad_s.dC_0), $(quad_s.dC_1), $(quad_s.dC_2), $(quad_s.du0), $(quad_s.dnoise),
    $(quad_s.du_f), $(quad_s.du_f_new), $(quad_s.dsol_sim), $(quad_s.dcache_sim))

reverse_quad_sim!(
    copy(quad_l.mats.A_0), copy(quad_l.mats.A_1), copy(quad_l.mats.A_2),
    copy(quad_l.mats.B), copy(quad_l.mats.C_0), copy(quad_l.mats.C_1),
    copy(quad_l.mats.C_2), copy(quad_l.u0), [copy(n) for n in quad_l.noise],
    copy(quad_l.u0), similar(quad_l.u0),
    quad_l.sol_sim, quad_l.cache_sim,
    quad_l.dA_0, quad_l.dA_1, quad_l.dA_2, quad_l.dB,
    quad_l.dC_0, quad_l.dC_1, quad_l.dC_2, quad_l.du0, quad_l.dnoise,
    quad_l.du_f, quad_l.du_f_new, quad_l.dsol_sim, quad_l.dcache_sim)

QUAD_ENZYME["simulation"]["reverse"]["large_mutable"] = @benchmarkable reverse_quad_sim!(
    $(copy(quad_l.mats.A_0)), $(copy(quad_l.mats.A_1)), $(copy(quad_l.mats.A_2)),
    $(copy(quad_l.mats.B)), $(copy(quad_l.mats.C_0)), $(copy(quad_l.mats.C_1)),
    $(copy(quad_l.mats.C_2)), $(copy(quad_l.u0)), $([copy(n) for n in quad_l.noise]),
    $(copy(quad_l.u0)), $(similar(quad_l.u0)),
    $(quad_l.sol_sim), $(quad_l.cache_sim),
    $(quad_l.dA_0), $(quad_l.dA_1), $(quad_l.dA_2), $(quad_l.dB),
    $(quad_l.dC_0), $(quad_l.dC_1), $(quad_l.dC_2), $(quad_l.du0), $(quad_l.dnoise),
    $(quad_l.du_f), $(quad_l.du_f_new), $(quad_l.dsol_sim), $(quad_l.dcache_sim))

# --- Forward mode: likelihood ---

function forward_quad_lik!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, y, H,
        u_f, u_f_new, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dy, dH,
        du_f, du_f_new, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); dA_2 = fill_zero!!(dA_2); dB = fill_zero!!(dB)
    dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1); dC_2 = fill_zero!!(dC_2); du0 = fill_zero!!(du0)
    du_f = fill_zero!!(du_f); du_f_new = fill_zero!!(du_f_new); dH = fill_zero!!(dH)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end
    @inbounds for i in eachindex(dy); dy[i] = fill_zero!!(dy[i]); end
    dA_1[1, 1] = 1.0

    autodiff(Forward, quad_lik_fwd_inner!,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(y, dy), Duplicated(H, dH),
        Duplicated(u_f, du_f), Duplicated(u_f_new, du_f_new),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

forward_quad_lik!(
    copy(quad_s.mats.A_0), copy(quad_s.mats.A_1), copy(quad_s.mats.A_2),
    copy(quad_s.mats.B), copy(quad_s.mats.C_0), copy(quad_s.mats.C_1),
    copy(quad_s.mats.C_2), copy(quad_s.u0), [copy(n) for n in quad_s.noise],
    [copy(y) for y in quad_s.y], copy(quad_s.H),
    copy(quad_s.u0), similar(quad_s.u0),
    quad_s.sol_lik, quad_s.cache_lik,
    quad_s.dA_0, quad_s.dA_1, quad_s.dA_2, quad_s.dB,
    quad_s.dC_0, quad_s.dC_1, quad_s.dC_2, quad_s.du0, quad_s.dnoise,
    quad_s.dy, quad_s.dH,
    quad_s.du_f, quad_s.du_f_new, quad_s.dsol_lik, quad_s.dcache_lik)

QUAD_ENZYME["likelihood"]["forward"]["small_mutable"] = @benchmarkable forward_quad_lik!(
    $(copy(quad_s.mats.A_0)), $(copy(quad_s.mats.A_1)), $(copy(quad_s.mats.A_2)),
    $(copy(quad_s.mats.B)), $(copy(quad_s.mats.C_0)), $(copy(quad_s.mats.C_1)),
    $(copy(quad_s.mats.C_2)), $(copy(quad_s.u0)), $([copy(n) for n in quad_s.noise]),
    $([copy(y) for y in quad_s.y]), $(copy(quad_s.H)),
    $(copy(quad_s.u0)), $(similar(quad_s.u0)),
    $(quad_s.sol_lik), $(quad_s.cache_lik),
    $(quad_s.dA_0), $(quad_s.dA_1), $(quad_s.dA_2), $(quad_s.dB),
    $(quad_s.dC_0), $(quad_s.dC_1), $(quad_s.dC_2), $(quad_s.du0), $(quad_s.dnoise),
    $(quad_s.dy), $(quad_s.dH),
    $(quad_s.du_f), $(quad_s.du_f_new), $(quad_s.dsol_lik), $(quad_s.dcache_lik))

forward_quad_lik!(
    copy(quad_l.mats.A_0), copy(quad_l.mats.A_1), copy(quad_l.mats.A_2),
    copy(quad_l.mats.B), copy(quad_l.mats.C_0), copy(quad_l.mats.C_1),
    copy(quad_l.mats.C_2), copy(quad_l.u0), [copy(n) for n in quad_l.noise],
    [copy(y) for y in quad_l.y], copy(quad_l.H),
    copy(quad_l.u0), similar(quad_l.u0),
    quad_l.sol_lik, quad_l.cache_lik,
    quad_l.dA_0, quad_l.dA_1, quad_l.dA_2, quad_l.dB,
    quad_l.dC_0, quad_l.dC_1, quad_l.dC_2, quad_l.du0, quad_l.dnoise,
    quad_l.dy, quad_l.dH,
    quad_l.du_f, quad_l.du_f_new, quad_l.dsol_lik, quad_l.dcache_lik)

QUAD_ENZYME["likelihood"]["forward"]["large_mutable"] = @benchmarkable forward_quad_lik!(
    $(copy(quad_l.mats.A_0)), $(copy(quad_l.mats.A_1)), $(copy(quad_l.mats.A_2)),
    $(copy(quad_l.mats.B)), $(copy(quad_l.mats.C_0)), $(copy(quad_l.mats.C_1)),
    $(copy(quad_l.mats.C_2)), $(copy(quad_l.u0)), $([copy(n) for n in quad_l.noise]),
    $([copy(y) for y in quad_l.y]), $(copy(quad_l.H)),
    $(copy(quad_l.u0)), $(similar(quad_l.u0)),
    $(quad_l.sol_lik), $(quad_l.cache_lik),
    $(quad_l.dA_0), $(quad_l.dA_1), $(quad_l.dA_2), $(quad_l.dB),
    $(quad_l.dC_0), $(quad_l.dC_1), $(quad_l.dC_2), $(quad_l.du0), $(quad_l.dnoise),
    $(quad_l.dy), $(quad_l.dH),
    $(quad_l.du_f), $(quad_l.du_f_new), $(quad_l.dsol_lik), $(quad_l.dcache_lik))

# --- Reverse mode: likelihood ---

function reverse_quad_lik!(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, y, H,
        u_f, u_f_new, sol_out, cache,
        dA_0, dA_1, dA_2, dB, dC_0, dC_1, dC_2, du0, dnoise, dy, dH,
        du_f, du_f_new, dsol_out, dcache)
    make_zero!(dsol_out); make_zero!(dcache)
    dA_0 = fill_zero!!(dA_0); dA_1 = fill_zero!!(dA_1); dA_2 = fill_zero!!(dA_2); dB = fill_zero!!(dB)
    dC_0 = fill_zero!!(dC_0); dC_1 = fill_zero!!(dC_1); dC_2 = fill_zero!!(dC_2); du0 = fill_zero!!(du0)
    du_f = fill_zero!!(du_f); du_f_new = fill_zero!!(du_f_new); dH = fill_zero!!(dH)
    @inbounds for i in eachindex(dnoise); dnoise[i] = fill_zero!!(dnoise[i]); end
    @inbounds for i in eachindex(dy); dy[i] = fill_zero!!(dy[i]); end

    autodiff(Reverse, quad_lik_rev_inner!, Active,
        Duplicated(A_0, dA_0), Duplicated(A_1, dA_1), Duplicated(A_2, dA_2),
        Duplicated(B, dB), Duplicated(C_0, dC_0), Duplicated(C_1, dC_1),
        Duplicated(C_2, dC_2), Duplicated(u0, du0), Duplicated(noise, dnoise),
        Duplicated(y, dy), Duplicated(H, dH),
        Duplicated(u_f, du_f), Duplicated(u_f_new, du_f_new),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache))
    return nothing
end

reverse_quad_lik!(
    copy(quad_s.mats.A_0), copy(quad_s.mats.A_1), copy(quad_s.mats.A_2),
    copy(quad_s.mats.B), copy(quad_s.mats.C_0), copy(quad_s.mats.C_1),
    copy(quad_s.mats.C_2), copy(quad_s.u0), [copy(n) for n in quad_s.noise],
    [copy(y) for y in quad_s.y], copy(quad_s.H),
    copy(quad_s.u0), similar(quad_s.u0),
    quad_s.sol_lik, quad_s.cache_lik,
    quad_s.dA_0, quad_s.dA_1, quad_s.dA_2, quad_s.dB,
    quad_s.dC_0, quad_s.dC_1, quad_s.dC_2, quad_s.du0, quad_s.dnoise,
    quad_s.dy, quad_s.dH,
    quad_s.du_f, quad_s.du_f_new, quad_s.dsol_lik, quad_s.dcache_lik)

QUAD_ENZYME["likelihood"]["reverse"]["small_mutable"] = @benchmarkable reverse_quad_lik!(
    $(copy(quad_s.mats.A_0)), $(copy(quad_s.mats.A_1)), $(copy(quad_s.mats.A_2)),
    $(copy(quad_s.mats.B)), $(copy(quad_s.mats.C_0)), $(copy(quad_s.mats.C_1)),
    $(copy(quad_s.mats.C_2)), $(copy(quad_s.u0)), $([copy(n) for n in quad_s.noise]),
    $([copy(y) for y in quad_s.y]), $(copy(quad_s.H)),
    $(copy(quad_s.u0)), $(similar(quad_s.u0)),
    $(quad_s.sol_lik), $(quad_s.cache_lik),
    $(quad_s.dA_0), $(quad_s.dA_1), $(quad_s.dA_2), $(quad_s.dB),
    $(quad_s.dC_0), $(quad_s.dC_1), $(quad_s.dC_2), $(quad_s.du0), $(quad_s.dnoise),
    $(quad_s.dy), $(quad_s.dH),
    $(quad_s.du_f), $(quad_s.du_f_new), $(quad_s.dsol_lik), $(quad_s.dcache_lik))

reverse_quad_lik!(
    copy(quad_l.mats.A_0), copy(quad_l.mats.A_1), copy(quad_l.mats.A_2),
    copy(quad_l.mats.B), copy(quad_l.mats.C_0), copy(quad_l.mats.C_1),
    copy(quad_l.mats.C_2), copy(quad_l.u0), [copy(n) for n in quad_l.noise],
    [copy(y) for y in quad_l.y], copy(quad_l.H),
    copy(quad_l.u0), similar(quad_l.u0),
    quad_l.sol_lik, quad_l.cache_lik,
    quad_l.dA_0, quad_l.dA_1, quad_l.dA_2, quad_l.dB,
    quad_l.dC_0, quad_l.dC_1, quad_l.dC_2, quad_l.du0, quad_l.dnoise,
    quad_l.dy, quad_l.dH,
    quad_l.du_f, quad_l.du_f_new, quad_l.dsol_lik, quad_l.dcache_lik)

QUAD_ENZYME["likelihood"]["reverse"]["large_mutable"] = @benchmarkable reverse_quad_lik!(
    $(copy(quad_l.mats.A_0)), $(copy(quad_l.mats.A_1)), $(copy(quad_l.mats.A_2)),
    $(copy(quad_l.mats.B)), $(copy(quad_l.mats.C_0)), $(copy(quad_l.mats.C_1)),
    $(copy(quad_l.mats.C_2)), $(copy(quad_l.u0)), $([copy(n) for n in quad_l.noise]),
    $([copy(y) for y in quad_l.y]), $(copy(quad_l.H)),
    $(copy(quad_l.u0)), $(similar(quad_l.u0)),
    $(quad_l.sol_lik), $(quad_l.cache_lik),
    $(quad_l.dA_0), $(quad_l.dA_1), $(quad_l.dA_2), $(quad_l.dB),
    $(quad_l.dC_0), $(quad_l.dC_1), $(quad_l.dC_2), $(quad_l.du0), $(quad_l.dnoise),
    $(quad_l.dy), $(quad_l.dH),
    $(quad_l.du_f), $(quad_l.du_f_new), $(quad_l.dsol_lik), $(quad_l.dcache_lik))

QUAD_ENZYME
