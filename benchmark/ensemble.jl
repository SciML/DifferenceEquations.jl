# Manual ensemble loop with Enzyme AD
# NOT using EnsembleProblem — Enzyme cannot differentiate through DiffEqBase dispatch.
# Uses construct-inside + solve! in a tight loop over trajectories.
#
# Returns ENS_BENCH BenchmarkGroup

using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, StateSpaceWorkspace, fill_zero!!

const ENS_BENCH = BenchmarkGroup()
ENS_BENCH["raw"] = BenchmarkGroup()
ENS_BENCH["forward"] = BenchmarkGroup()
ENS_BENCH["reverse"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

const p_ens_small = (; N = 2, K = 1, M = 2, T = 10, N_traj = 20)
const p_ens_large = (; N = 5, K = 2, M = 3, T = 50, N_traj = 50)

# =============================================================================
# Problem setup
# =============================================================================

function make_ensemble_benchmark(; N, K, M, T, N_traj, seed = 42)
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    B = 0.1 * randn(N, K)
    C = randn(M, N)
    u0 = zeros(N)

    # Pre-generate noise for each trajectory
    all_noise = [[randn(K) for _ in 1:T] for _ in 1:N_traj]

    # Pre-allocate sol/cache for each trajectory (simulation only, no obs)
    prob_template = LinearStateSpaceProblem(A, B, u0, (0, T); C, noise = all_noise[1])
    all_sol = [deepcopy(init(prob_template, DirectIteration()).output) for _ in 1:N_traj]
    all_cache = [deepcopy(init(prob_template, DirectIteration()).cache) for _ in 1:N_traj]

    # Shadows for AD
    dA = make_zero(A)
    dB = make_zero(B)
    dC = make_zero(C)
    du0 = make_zero(u0)
    dall_noise = [[make_zero(all_noise[1][1]) for _ in 1:T] for _ in 1:N_traj]
    dall_sol = [make_zero(s) for s in all_sol]
    dall_cache = [make_zero(c) for c in all_cache]

    return (; A, B, C, u0, all_noise, all_sol, all_cache,
        dA, dB, dC, du0, dall_noise, dall_sol, dall_cache)
end

# =============================================================================
# Wrapper functions
# =============================================================================

function ensemble_raw!(A, B, C, u0, all_noise, all_sol, all_cache)
    total = 0.0
    for i in eachindex(all_noise)
        prob = LinearStateSpaceProblem(A, B, u0, (0, length(all_noise[i])); C, noise = all_noise[i])
        ws = StateSpaceWorkspace(prob, DirectIteration(), all_sol[i], all_cache[i])
        solve!(ws)
        total += sum(all_sol[i].u[end])
    end
    return total / length(all_noise)
end

function ensemble_forward_bench!(A, B, C, u0, all_noise, all_sol, all_cache)
    # Same as raw — Enzyme differentiates through this
    return ensemble_raw!(A, B, C, u0, all_noise, all_sol, all_cache)
end

function ensemble_scalar!(A, B, C, u0, all_noise, all_sol, all_cache)::Float64
    return ensemble_raw!(A, B, C, u0, all_noise, all_sol, all_cache)
end

# =============================================================================
# Instantiate problems
# =============================================================================

const ens_s = make_ensemble_benchmark(; p_ens_small...)
const ens_l = make_ensemble_benchmark(; p_ens_large...)

# =============================================================================
# Raw benchmarks (primal solve through public API)
# =============================================================================

function raw_ens!(A, B, C, u0, all_noise, all_sol, all_cache)
    return ensemble_raw!(A, B, C, u0, all_noise, all_sol, all_cache)
end

# Warmup
raw_ens!(ens_s.A, ens_s.B, ens_s.C, ens_s.u0, ens_s.all_noise,
    ens_s.all_sol, ens_s.all_cache)
raw_ens!(ens_l.A, ens_l.B, ens_l.C, ens_l.u0, ens_l.all_noise,
    ens_l.all_sol, ens_l.all_cache)

ENS_BENCH["raw"]["small"] = @benchmarkable raw_ens!(
    $(ens_s.A), $(ens_s.B), $(ens_s.C), $(ens_s.u0), $(ens_s.all_noise),
    $(ens_s.all_sol), $(ens_s.all_cache))
ENS_BENCH["raw"]["large"] = @benchmarkable raw_ens!(
    $(ens_l.A), $(ens_l.B), $(ens_l.C), $(ens_l.u0), $(ens_l.all_noise),
    $(ens_l.all_sol), $(ens_l.all_cache))

# =============================================================================
# Forward mode AD — perturb A[1,1], return computed arrays
# =============================================================================

function forward_ensemble_bench!(A, B, C, u0, all_noise, all_sol, all_cache,
        dA, dB, dC, du0, dall_noise, dall_sol, dall_cache)
    # Zero all shadows
    dA = fill_zero!!(dA); dB = fill_zero!!(dB); dC = fill_zero!!(dC); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dall_noise)
        for j in eachindex(dall_noise[i])
            dall_noise[i][j] = fill_zero!!(dall_noise[i][j])
        end
    end
    @inbounds for i in eachindex(dall_sol)
        make_zero!(dall_sol[i])
    end
    @inbounds for i in eachindex(dall_cache)
        make_zero!(dall_cache[i])
    end
    # Set perturbation direction
    dA[1, 1] = 1.0

    autodiff(Forward, ensemble_forward_bench!,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(all_noise, dall_noise),
        Duplicated(all_sol, dall_sol),
        Duplicated(all_cache, dall_cache))
    return nothing
end

# Warmup
forward_ensemble_bench!(
    copy(ens_s.A), copy(ens_s.B), copy(ens_s.C), copy(ens_s.u0),
    [[copy(n) for n in traj] for traj in ens_s.all_noise],
    ens_s.all_sol, ens_s.all_cache,
    ens_s.dA, ens_s.dB, ens_s.dC, ens_s.du0,
    ens_s.dall_noise, ens_s.dall_sol, ens_s.dall_cache)

ENS_BENCH["forward"]["small"] = @benchmarkable forward_ensemble_bench!(
    $(copy(ens_s.A)), $(copy(ens_s.B)), $(copy(ens_s.C)), $(copy(ens_s.u0)),
    $([[copy(n) for n in traj] for traj in ens_s.all_noise]),
    $(ens_s.all_sol), $(ens_s.all_cache),
    $(ens_s.dA), $(ens_s.dB), $(ens_s.dC), $(ens_s.du0),
    $(ens_s.dall_noise), $(ens_s.dall_sol), $(ens_s.dall_cache))

# Warmup large
forward_ensemble_bench!(
    copy(ens_l.A), copy(ens_l.B), copy(ens_l.C), copy(ens_l.u0),
    [[copy(n) for n in traj] for traj in ens_l.all_noise],
    ens_l.all_sol, ens_l.all_cache,
    ens_l.dA, ens_l.dB, ens_l.dC, ens_l.du0,
    ens_l.dall_noise, ens_l.dall_sol, ens_l.dall_cache)

ENS_BENCH["forward"]["large"] = @benchmarkable forward_ensemble_bench!(
    $(copy(ens_l.A)), $(copy(ens_l.B)), $(copy(ens_l.C)), $(copy(ens_l.u0)),
    $([[copy(n) for n in traj] for traj in ens_l.all_noise]),
    $(ens_l.all_sol), $(ens_l.all_cache),
    $(ens_l.dA), $(ens_l.dB), $(ens_l.dC), $(ens_l.du0),
    $(ens_l.dall_noise), $(ens_l.dall_sol), $(ens_l.dall_cache))

# =============================================================================
# Reverse mode AD — all Duplicated, scalar return with Active
# =============================================================================

function reverse_ensemble_bench!(A, B, C, u0, all_noise, all_sol, all_cache,
        dA, dB, dC, du0, dall_noise, dall_sol, dall_cache)
    # Zero all shadows
    dA = fill_zero!!(dA); dB = fill_zero!!(dB); dC = fill_zero!!(dC); du0 = fill_zero!!(du0)
    @inbounds for i in eachindex(dall_noise)
        for j in eachindex(dall_noise[i])
            dall_noise[i][j] = fill_zero!!(dall_noise[i][j])
        end
    end
    @inbounds for i in eachindex(dall_sol)
        make_zero!(dall_sol[i])
    end
    @inbounds for i in eachindex(dall_cache)
        make_zero!(dall_cache[i])
    end

    autodiff(Reverse, ensemble_scalar!, Active,
        Duplicated(A, dA), Duplicated(B, dB), Duplicated(C, dC),
        Duplicated(u0, du0), Duplicated(all_noise, dall_noise),
        Duplicated(all_sol, dall_sol),
        Duplicated(all_cache, dall_cache))
    return nothing
end

# Warmup
reverse_ensemble_bench!(
    copy(ens_s.A), copy(ens_s.B), copy(ens_s.C), copy(ens_s.u0),
    [[copy(n) for n in traj] for traj in ens_s.all_noise],
    ens_s.all_sol, ens_s.all_cache,
    ens_s.dA, ens_s.dB, ens_s.dC, ens_s.du0,
    ens_s.dall_noise, ens_s.dall_sol, ens_s.dall_cache)

ENS_BENCH["reverse"]["small"] = @benchmarkable reverse_ensemble_bench!(
    $(copy(ens_s.A)), $(copy(ens_s.B)), $(copy(ens_s.C)), $(copy(ens_s.u0)),
    $([[copy(n) for n in traj] for traj in ens_s.all_noise]),
    $(ens_s.all_sol), $(ens_s.all_cache),
    $(ens_s.dA), $(ens_s.dB), $(ens_s.dC), $(ens_s.du0),
    $(ens_s.dall_noise), $(ens_s.dall_sol), $(ens_s.dall_cache))

# Warmup large
reverse_ensemble_bench!(
    copy(ens_l.A), copy(ens_l.B), copy(ens_l.C), copy(ens_l.u0),
    [[copy(n) for n in traj] for traj in ens_l.all_noise],
    ens_l.all_sol, ens_l.all_cache,
    ens_l.dA, ens_l.dB, ens_l.dC, ens_l.du0,
    ens_l.dall_noise, ens_l.dall_sol, ens_l.dall_cache)

ENS_BENCH["reverse"]["large"] = @benchmarkable reverse_ensemble_bench!(
    $(copy(ens_l.A)), $(copy(ens_l.B)), $(copy(ens_l.C)), $(copy(ens_l.u0)),
    $([[copy(n) for n in traj] for traj in ens_l.all_noise]),
    $(ens_l.all_sol), $(ens_l.all_cache),
    $(ens_l.dA), $(ens_l.dB), $(ens_l.dC), $(ens_l.du0),
    $(ens_l.dall_noise), $(ens_l.dall_sol), $(ens_l.dall_cache))

ENS_BENCH
