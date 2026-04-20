# Enzyme AD benchmarks for ConditionalLikelihood
# Returns CL_ENZYME BenchmarkGroup

using Enzyme: make_zero, make_zero!
using DifferenceEquations: init, solve!, StateSpaceWorkspace, fill_zero!!

const CL_ENZYME = BenchmarkGroup()
CL_ENZYME["raw"] = BenchmarkGroup()
CL_ENZYME["forward"] = BenchmarkGroup()
CL_ENZYME["reverse"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

# CL requires fully-observed state: M = N (observations are state-dimensional)
const p_cl_small = (; N = 5, M = 5, T = 10)
const p_cl_large = (; N = 30, M = 30, T = 100)

# =============================================================================
# Problem setup
# =============================================================================

function make_cl_benchmark(p; seed = 42)
    (; N, M, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    C = randn(M, N)
    H = 0.1 * randn(M, M)
    R = H * H'

    # Generate observations: state evolves via A, observed via C + noise
    x = zeros(N)
    y = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        x = A * x + 0.1 * randn(N)
        y[t] = C * x + H * randn(M)
    end

    # Create problem and workspace (B=nothing, no process noise in prediction)
    prob = LinearStateSpaceProblem(
        A, nothing, zeros(N), (0, T); C,
        observables_noise = R, observables = y
    )
    ws = init(prob, ConditionalLikelihood())
    sol_out = ws.output
    cache = ws.cache

    # Shadow copies for AD
    dsol_out = make_zero(sol_out)
    dcache = make_zero(cache)
    dA = make_zero(A)
    dC = make_zero(C)
    dH = make_zero(H)
    dy = [make_zero(y[1]) for _ in 1:T]

    return (;
        A, C, H, R, y, prob, sol_out, cache,
        dsol_out, dcache, dA, dC, dH, dy,
    )
end

# =============================================================================
# Scalar wrapper for reverse mode (returns logpdf)
# =============================================================================

function cl_loglik_bench!(A, C, H, y, sol_out, cache)
    R = H * H'
    prob = LinearStateSpaceProblem(
        A, nothing, zeros(eltype(A), size(A, 1)), (0, length(y)); C,
        observables_noise = R, observables = y
    )
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol_out, cache)
    return solve!(ws).logpdf
end

# =============================================================================
# Forward wrapper (returns state and obs from cache)
# =============================================================================

function cl_forward_bench!(A, C, H, y, sol_out, cache)
    R = H * H'
    prob = LinearStateSpaceProblem(
        A, nothing, zeros(eltype(A), size(A, 1)), (0, length(y)); C,
        observables_noise = R, observables = y
    )
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol_out, cache)
    solve!(ws)
    return (sol_out.u[end], sol_out.z[end])
end

# =============================================================================
# Instantiate problems
# =============================================================================

const cl_s = make_cl_benchmark(p_cl_small)
const cl_l = make_cl_benchmark(p_cl_large)

# =============================================================================
# Raw benchmarks (primal solve through public API)
# =============================================================================

function raw_cl!(prob, sol_out, cache)
    ws = StateSpaceWorkspace(prob, ConditionalLikelihood(), sol_out, cache)
    return solve!(ws).logpdf
end

# Warmup
raw_cl!(cl_s.prob, cl_s.sol_out, cl_s.cache)
raw_cl!(cl_l.prob, cl_l.sol_out, cl_l.cache)

CL_ENZYME["raw"]["small_mutable"] = @benchmarkable raw_cl!($(cl_s.prob), $(cl_s.sol_out), $(cl_s.cache))
CL_ENZYME["raw"]["large_mutable"] = @benchmarkable raw_cl!($(cl_l.prob), $(cl_l.sol_out), $(cl_l.cache))

# =============================================================================
# Forward mode AD — perturb A[1,1], return computed matrices
# =============================================================================

function forward_cl_bench!(
        A, C, H, y, sol_out, cache,
        dA, dC, dH, dy, dsol_out, dcache
    )
    # Zero all shadows
    make_zero!(dsol_out)
    make_zero!(dcache)
    dA = fill_zero!!(dA); dC = fill_zero!!(dC); dH = fill_zero!!(dH)
    @inbounds for i in eachindex(dy)
        dy[i] = fill_zero!!(dy[i])
    end
    # Set perturbation direction
    dA[1, 1] = 1.0

    autodiff(
        Forward, cl_forward_bench!,
        Duplicated(A, dA), Duplicated(C, dC),
        Duplicated(H, dH), Duplicated(y, dy),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache)
    )
    return nothing
end

# Warmup
forward_cl_bench!(
    copy(cl_s.A), copy(cl_s.C), copy(cl_s.H),
    [copy(yi) for yi in cl_s.y], cl_s.sol_out, cl_s.cache,
    cl_s.dA, cl_s.dC, cl_s.dH, cl_s.dy, cl_s.dsol_out, cl_s.dcache
)

CL_ENZYME["forward"]["small_mutable"] = @benchmarkable forward_cl_bench!(
    $(copy(cl_s.A)), $(copy(cl_s.C)), $(copy(cl_s.H)),
    $([copy(yi) for yi in cl_s.y]), $(cl_s.sol_out), $(cl_s.cache),
    $(cl_s.dA), $(cl_s.dC), $(cl_s.dH), $(cl_s.dy), $(cl_s.dsol_out), $(cl_s.dcache)
) teardown = (GC.enable(true); GC.gc(); GC.enable(false))

# Warmup large
forward_cl_bench!(
    copy(cl_l.A), copy(cl_l.C), copy(cl_l.H),
    [copy(yi) for yi in cl_l.y], cl_l.sol_out, cl_l.cache,
    cl_l.dA, cl_l.dC, cl_l.dH, cl_l.dy, cl_l.dsol_out, cl_l.dcache
)

CL_ENZYME["forward"]["large_mutable"] = @benchmarkable forward_cl_bench!(
    $(copy(cl_l.A)), $(copy(cl_l.C)), $(copy(cl_l.H)),
    $([copy(yi) for yi in cl_l.y]), $(cl_l.sol_out), $(cl_l.cache),
    $(cl_l.dA), $(cl_l.dC), $(cl_l.dH), $(cl_l.dy), $(cl_l.dsol_out), $(cl_l.dcache)
) teardown = (GC.enable(true); GC.gc(); GC.enable(false))

# =============================================================================
# Reverse mode AD — all Duplicated, scalar logpdf output
# =============================================================================

function reverse_cl_bench!(
        A, C, H, y, sol_out, cache,
        dA, dC, dH, dy, dsol_out, dcache
    )
    # Zero all shadows
    make_zero!(dsol_out)
    make_zero!(dcache)
    dA = fill_zero!!(dA); dC = fill_zero!!(dC); dH = fill_zero!!(dH)
    @inbounds for i in eachindex(dy)
        dy[i] = fill_zero!!(dy[i])
    end

    autodiff(
        Reverse, cl_loglik_bench!, Active,
        Duplicated(A, dA), Duplicated(C, dC),
        Duplicated(H, dH), Duplicated(y, dy),
        Duplicated(sol_out, dsol_out), Duplicated(cache, dcache)
    )
    return nothing
end

# Warmup
reverse_cl_bench!(
    copy(cl_s.A), copy(cl_s.C), copy(cl_s.H),
    [copy(yi) for yi in cl_s.y], cl_s.sol_out, cl_s.cache,
    cl_s.dA, cl_s.dC, cl_s.dH, cl_s.dy, cl_s.dsol_out, cl_s.dcache
)

CL_ENZYME["reverse"]["small_mutable"] = @benchmarkable reverse_cl_bench!(
    $(copy(cl_s.A)), $(copy(cl_s.C)), $(copy(cl_s.H)),
    $([copy(yi) for yi in cl_s.y]), $(cl_s.sol_out), $(cl_s.cache),
    $(cl_s.dA), $(cl_s.dC), $(cl_s.dH), $(cl_s.dy), $(cl_s.dsol_out), $(cl_s.dcache)
) teardown = (GC.enable(true); GC.gc(); GC.enable(false))

# Warmup large
reverse_cl_bench!(
    copy(cl_l.A), copy(cl_l.C), copy(cl_l.H),
    [copy(yi) for yi in cl_l.y], cl_l.sol_out, cl_l.cache,
    cl_l.dA, cl_l.dC, cl_l.dH, cl_l.dy, cl_l.dsol_out, cl_l.dcache
)

CL_ENZYME["reverse"]["large_mutable"] = @benchmarkable reverse_cl_bench!(
    $(copy(cl_l.A)), $(copy(cl_l.C)), $(copy(cl_l.H)),
    $([copy(yi) for yi in cl_l.y]), $(cl_l.sol_out), $(cl_l.cache),
    $(cl_l.dA), $(cl_l.dC), $(cl_l.dH), $(cl_l.dy), $(cl_l.dsol_out), $(cl_l.dcache)
) teardown = (GC.enable(true); GC.gc(); GC.enable(false))

CL_ENZYME
