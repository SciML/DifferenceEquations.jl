# Enzyme AD benchmarks for DirectIteration (joint likelihood)
# Returns DI_ENZYME BenchmarkGroup

using Enzyme: make_zero
using DifferenceEquations: _direct_iteration_loglik!, alloc_direct_loglik_cache,
    zero_direct_loglik_cache!!

const DI_ENZYME = BenchmarkGroup()
DI_ENZYME["raw"] = BenchmarkGroup()
DI_ENZYME["forward"] = BenchmarkGroup()
DI_ENZYME["reverse"] = BenchmarkGroup()

# =============================================================================
# Problem sizes
# =============================================================================

const p_di_small = (; N = 5, M = 2, K = 2, L = 2, T = 10)
const p_di_large = (; N = 30, M = 10, K = 10, L = 10, T = 100)

# =============================================================================
# Problem setup — mutable arrays
# =============================================================================

function make_di_problem(p; seed = 42)
    (; N, M, K, L, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    B = 0.1 * randn(N, K)
    C = randn(M, N)
    H = 0.1 * randn(M, L)
    u0 = zeros(N)
    noise = [randn(K) for _ in 1:T]

    # Generate observations
    x = [zeros(N) for _ in 1:(T + 1)]
    x[1] = randn(N)
    for t in 1:T
        x[t + 1] = A * x[t] + B * noise[t]
    end
    y = [C * x[t + 1] + H * randn(L) for t in 1:T]

    # Allocate cache
    cache = alloc_direct_loglik_cache(u0, A, B, C, H, T + 1)

    # Shadow copies for AD
    dcache = make_zero(cache)
    du0 = make_zero(u0)
    dnoise = [make_zero(noise[1]) for _ in 1:T]
    dy = [make_zero(y[1]) for _ in 1:T]
    dA = make_zero(A)
    dB = make_zero(B)
    dC = make_zero(C)

    return (; A, B, C, H, u0, noise, y, cache,
        dcache, du0, dnoise, dy, dA, dB, dC)
end

# =============================================================================
# Problem setup — static arrays
# =============================================================================

function make_di_problem_static(p; seed = 42)
    (; N, M, K, L, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = SMatrix{N, N}(0.5 * A_raw / maximum(abs.(eigvals(A_raw))))
    B = SMatrix{N, K}(0.1 * randn(N, K))
    C = SMatrix{M, N}(randn(M, N))
    H = SMatrix{M, L}(0.1 * randn(M, L))
    u0 = SVector{N}(zeros(N))

    # Generate observations
    Random.seed!(seed)
    x_mut = randn(N)
    noise = [SVector{K}(randn(K)) for _ in 1:T]
    y = [SVector{M}(zeros(M)) for _ in 1:T]
    for t in 1:T
        x_next = Matrix(A) * x_mut + Matrix(B) * Vector(noise[t])
        y[t] = SVector{M}(Matrix(C) * x_next + Matrix(H) * randn(L))
        x_mut = x_next
    end

    # Allocate cache
    cache = alloc_direct_loglik_cache(u0, A, B, C, H, T + 1)

    # Shadow copies for AD
    dcache = make_zero(cache)
    du0 = make_zero(u0)
    dnoise = make_zero(noise)
    dy = make_zero(y)
    dA = make_zero(A)
    dB = make_zero(B)
    dC = make_zero(C)

    return (; A, B, C, H, u0, noise, y, cache,
        dcache, du0, dnoise, dy, dA, dB, dC)
end

# =============================================================================
# Instantiate problems
# =============================================================================

const di_s = make_di_problem(p_di_small)
const di_ss = make_di_problem_static(p_di_small)
const di_l = make_di_problem(p_di_large)

# =============================================================================
# Scalar wrapper (no zeroing, just calls underlying function)
# =============================================================================

scalar_di_loglik!(A, B, C, u0, noise, y, H, cache) =
    _direct_iteration_loglik!(A, B, C, u0, noise, y, H, cache)

# =============================================================================
# Helper: zero shadow cache (element-wise for static caches with immutable fields)
# =============================================================================

function zero_di_shadow_cache!(dcache)
    Enzyme.make_zero!(dcache)
    return nothing
end

function zero_di_shadow_cache_static!(dcache)
    # Static caches have vectors of immutable SArrays — zero by reassignment
    @inbounds for t in eachindex(dcache.u)
        dcache.u[t] = zero(dcache.u[t])
    end
    @inbounds for t in eachindex(dcache.z)
        dcache.z[t] = zero(dcache.z[t])
    end
    @inbounds for t in eachindex(dcache.innovation)
        dcache.innovation[t] = zero(dcache.innovation[t])
        dcache.innovation_solved[t] = zero(dcache.innovation_solved[t])
    end
    # R and R_chol are immutable SMatrix — already zero from make_zero, no-op
    return nothing
end

# =============================================================================
# Raw benchmarks (include cache zeroing in the call)
# =============================================================================

function raw_di!(A, B, C, u0, noise, y, H, cache)
    zero_direct_loglik_cache!!(cache)
    return _direct_iteration_loglik!(A, B, C, u0, noise, y, H, cache)
end

# Warmup
raw_di!(di_s.A, di_s.B, di_s.C, di_s.u0, di_s.noise, di_s.y, di_s.H, di_s.cache)
raw_di!(di_ss.A, di_ss.B, di_ss.C, di_ss.u0, di_ss.noise, di_ss.y, di_ss.H, di_ss.cache)
raw_di!(di_l.A, di_l.B, di_l.C, di_l.u0, di_l.noise, di_l.y, di_l.H, di_l.cache)

DI_ENZYME["raw"]["small_mutable"] = @benchmarkable raw_di!(
    $(di_s.A), $(di_s.B), $(di_s.C), $(di_s.u0),
    $(di_s.noise), $(di_s.y), $(di_s.H), $(di_s.cache))

DI_ENZYME["raw"]["small_static"] = @benchmarkable raw_di!(
    $(di_ss.A), $(di_ss.B), $(di_ss.C), $(di_ss.u0),
    $(di_ss.noise), $(di_ss.y), $(di_ss.H), $(di_ss.cache))

DI_ENZYME["raw"]["large_mutable"] = @benchmarkable raw_di!(
    $(di_l.A), $(di_l.B), $(di_l.C), $(di_l.u0),
    $(di_l.noise), $(di_l.y), $(di_l.H), $(di_l.cache))

# =============================================================================
# Forward mode AD wrappers (mutable)
# =============================================================================

function forward_di_mutable!(A, B, C, u0, noise, y, H, cache,
        dnoise, dy, dcache)
    # Zero primal cache
    zero_direct_loglik_cache!!(cache)
    # Zero shadows
    zero_di_shadow_cache!(dcache)
    @inbounds for i in eachindex(dnoise)
        Enzyme.make_zero!(dnoise[i])
    end
    @inbounds for i in eachindex(dy)
        Enzyme.make_zero!(dy[i])
    end
    # Set perturbation direction
    dnoise[1][1] = 1.0

    autodiff(Forward, scalar_di_loglik!, Const(A), Const(B), Const(C), Const(u0),
        Duplicated(noise, dnoise), Duplicated(y, dy),
        Const(H), Duplicated(cache, dcache))
    return nothing
end

# Warmup small
forward_di_mutable!(di_s.A, di_s.B, di_s.C, di_s.u0,
    [copy(ni) for ni in di_s.noise], [copy(yi) for yi in di_s.y], di_s.H, di_s.cache,
    di_s.dnoise, di_s.dy, di_s.dcache)

DI_ENZYME["forward"]["small_mutable"] = @benchmarkable forward_di_mutable!(
    $(di_s.A), $(di_s.B), $(di_s.C), $(di_s.u0),
    $([copy(ni) for ni in di_s.noise]), $([copy(yi) for yi in di_s.y]),
    $(di_s.H), $(di_s.cache),
    $(di_s.dnoise), $(di_s.dy), $(di_s.dcache))

# Warmup large
forward_di_mutable!(di_l.A, di_l.B, di_l.C, di_l.u0,
    [copy(ni) for ni in di_l.noise], [copy(yi) for yi in di_l.y], di_l.H, di_l.cache,
    di_l.dnoise, di_l.dy, di_l.dcache)

DI_ENZYME["forward"]["large_mutable"] = @benchmarkable forward_di_mutable!(
    $(di_l.A), $(di_l.B), $(di_l.C), $(di_l.u0),
    $([copy(ni) for ni in di_l.noise]), $([copy(yi) for yi in di_l.y]),
    $(di_l.H), $(di_l.cache),
    $(di_l.dnoise), $(di_l.dy), $(di_l.dcache))

# =============================================================================
# Forward mode AD wrappers (static)
# =============================================================================

function forward_di_static!(A, B, C, u0, noise, y, H, cache,
        dnoise, dy, dcache)
    # Zero primal cache
    zero_direct_loglik_cache!!(cache)
    # Zero shadows (element-wise for immutable)
    zero_di_shadow_cache_static!(dcache)
    @inbounds for i in eachindex(dnoise)
        dnoise[i] = zero(dnoise[i])
    end
    @inbounds for i in eachindex(dy)
        dy[i] = zero(dy[i])
    end
    # Set perturbation direction (immutable SVector)
    K = length(noise[1])
    dnoise[1] = typeof(noise[1])(vcat(1.0, zeros(K - 1)))

    autodiff(Forward, scalar_di_loglik!, Const(A), Const(B), Const(C), Const(u0),
        Duplicated(noise, dnoise), Duplicated(y, dy),
        Const(H), Duplicated(cache, dcache))
    return nothing
end

# Warmup
forward_di_static!(di_ss.A, di_ss.B, di_ss.C, di_ss.u0,
    di_ss.noise, di_ss.y, di_ss.H, di_ss.cache,
    di_ss.dnoise, di_ss.dy, di_ss.dcache)

DI_ENZYME["forward"]["small_static"] = @benchmarkable forward_di_static!(
    $(di_ss.A), $(di_ss.B), $(di_ss.C), $(di_ss.u0),
    $(di_ss.noise), $(di_ss.y), $(di_ss.H), $(di_ss.cache),
    $(di_ss.dnoise), $(di_ss.dy), $(di_ss.dcache))

# =============================================================================
# Reverse mode AD wrappers (mutable)
# =============================================================================

function reverse_di_mutable!(A, B, C, u0, noise, y, H, cache,
        du0, dnoise, dy, dcache)
    # Zero primal cache
    zero_direct_loglik_cache!!(cache)
    # Zero shadows
    zero_di_shadow_cache!(dcache)
    Enzyme.make_zero!(du0)
    @inbounds for i in eachindex(dnoise)
        Enzyme.make_zero!(dnoise[i])
    end
    @inbounds for i in eachindex(dy)
        Enzyme.make_zero!(dy[i])
    end

    autodiff(Reverse, scalar_di_loglik!, Const(A), Const(B), Const(C),
        Duplicated(u0, du0), Duplicated(noise, dnoise), Duplicated(y, dy),
        Const(H), Duplicated(cache, dcache))
    return nothing
end

# Warmup small
reverse_di_mutable!(di_s.A, di_s.B, di_s.C,
    copy(di_s.u0), [copy(ni) for ni in di_s.noise], [copy(yi) for yi in di_s.y],
    di_s.H, di_s.cache,
    di_s.du0, di_s.dnoise, di_s.dy, di_s.dcache)

DI_ENZYME["reverse"]["small_mutable"] = @benchmarkable reverse_di_mutable!(
    $(di_s.A), $(di_s.B), $(di_s.C),
    $(copy(di_s.u0)), $([copy(ni) for ni in di_s.noise]), $([copy(yi) for yi in di_s.y]),
    $(di_s.H), $(di_s.cache),
    $(di_s.du0), $(di_s.dnoise), $(di_s.dy), $(di_s.dcache))

# Warmup large
reverse_di_mutable!(di_l.A, di_l.B, di_l.C,
    copy(di_l.u0), [copy(ni) for ni in di_l.noise], [copy(yi) for yi in di_l.y],
    di_l.H, di_l.cache,
    di_l.du0, di_l.dnoise, di_l.dy, di_l.dcache)

DI_ENZYME["reverse"]["large_mutable"] = @benchmarkable reverse_di_mutable!(
    $(di_l.A), $(di_l.B), $(di_l.C),
    $(copy(di_l.u0)), $([copy(ni) for ni in di_l.noise]), $([copy(yi) for yi in di_l.y]),
    $(di_l.H), $(di_l.cache),
    $(di_l.du0), $(di_l.dnoise), $(di_l.dy), $(di_l.dcache))

# =============================================================================
# Reverse mode AD wrappers (static)
# =============================================================================

function reverse_di_static!(A, B, C, u0, noise, y, H, cache,
        du0, dnoise, dy, dcache)
    # Zero primal cache
    zero_direct_loglik_cache!!(cache)
    # Zero shadows (element-wise for immutable)
    zero_di_shadow_cache_static!(dcache)
    @inbounds for i in eachindex(dnoise)
        dnoise[i] = zero(dnoise[i])
    end
    @inbounds for i in eachindex(dy)
        dy[i] = zero(dy[i])
    end

    autodiff(Reverse, scalar_di_loglik!, Const(A), Const(B), Const(C),
        DuplicatedNoNeed(u0, du0), Duplicated(noise, dnoise), Duplicated(y, dy),
        Const(H), Duplicated(cache, dcache))
    return nothing
end

# Warmup
reverse_di_static!(di_ss.A, di_ss.B, di_ss.C,
    di_ss.u0, di_ss.noise, di_ss.y, di_ss.H, di_ss.cache,
    di_ss.du0, di_ss.dnoise, di_ss.dy, di_ss.dcache)

DI_ENZYME["reverse"]["small_static"] = @benchmarkable reverse_di_static!(
    $(di_ss.A), $(di_ss.B), $(di_ss.C),
    $(di_ss.u0), $(di_ss.noise), $(di_ss.y), $(di_ss.H), $(di_ss.cache),
    $(di_ss.du0), $(di_ss.dnoise), $(di_ss.dy), $(di_ss.dcache))

DI_ENZYME
