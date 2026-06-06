# ForwardDiff AD benchmarks for Linear DirectIteration simulation (no observations/likelihood)
# Returns SIM_FD BenchmarkGroup

using ForwardDiff
using DifferenceEquations: init, solve!, StateSpaceWorkspace

const SIM_FD = BenchmarkGroup()
SIM_FD["gradient"] = BenchmarkGroup()

# =============================================================================
# Type promotion helper
# =============================================================================

_fd_promote_sim(::Type{T}, x::AbstractArray{T}) where {T} = x
_fd_promote_sim(::Type{T}, x::AbstractArray) where {T} = T.(x)

# =============================================================================
# Problem sizes (same as enzyme_linear_simulation.jl)
# =============================================================================

const p_sim_fd_small = (; N = 5, M = 3, K = 2, T = 10)
const p_sim_fd_large = (; N = 30, M = 10, K = 10, T = 100)

# =============================================================================
# Problem setup
# =============================================================================

function make_sim_fd_benchmark(p; seed = 42)
    (; N, M, K, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    B = 0.1 * randn(N, K)
    C = randn(M, N)
    u0 = zeros(N)
    noise = [randn(K) for _ in 1:T]

    return (; A, B, C, u0, noise)
end

# =============================================================================
# ForwardDiff wrapper — gradient of sum(u[end]) w.r.t. vec(A)
# =============================================================================

function sim_scalar_fd_bench(A_vec, B, C, u0, noise, N)
    T_el = eltype(A_vec)
    A = reshape(A_vec, N, N)
    prob = LinearStateSpaceProblem(
        A, _fd_promote_sim(T_el, B),
        _fd_promote_sim(T_el, u0), (0, length(noise));
        C = _fd_promote_sim(T_el, C), noise = noise
    )
    sol = solve(prob, DirectIteration())
    return sum(sol.u[end])
end

function fd_gradient_sim!(A_vec, B, C, u0, noise, N)
    return ForwardDiff.gradient(
        a -> sim_scalar_fd_bench(a, B, C, u0, noise, N), A_vec
    )
end

# =============================================================================
# Instantiate problems
# =============================================================================

const sim_fd_s = make_sim_fd_benchmark(p_sim_fd_small)
const sim_fd_l = make_sim_fd_benchmark(p_sim_fd_large)

# =============================================================================
# Warmup and benchmarks
# =============================================================================

fd_gradient_sim!(
    vec(copy(sim_fd_s.A)), sim_fd_s.B, sim_fd_s.C,
    sim_fd_s.u0, sim_fd_s.noise, p_sim_fd_small.N
)
fd_gradient_sim!(
    vec(copy(sim_fd_l.A)), sim_fd_l.B, sim_fd_l.C,
    sim_fd_l.u0, sim_fd_l.noise, p_sim_fd_large.N
)

SIM_FD["gradient"]["small_mutable"] = @benchmarkable fd_gradient_sim!(
    $(vec(copy(sim_fd_s.A))), $(sim_fd_s.B), $(sim_fd_s.C),
    $(sim_fd_s.u0), $(sim_fd_s.noise), $(p_sim_fd_small.N)
)

SIM_FD["gradient"]["large_mutable"] = @benchmarkable fd_gradient_sim!(
    $(vec(copy(sim_fd_l.A))), $(sim_fd_l.B), $(sim_fd_l.C),
    $(sim_fd_l.u0), $(sim_fd_l.noise), $(p_sim_fd_large.N)
)

# =============================================================================
# StaticArrays variant (small only)
# =============================================================================

SIM_FD["gradient"]["small_static"] = let
    (; A, B, C, u0, noise) = sim_fd_s
    N = p_sim_fd_small.N; M = p_sim_fd_small.M; K = p_sim_fd_small.K

    B_s = SMatrix{N, K}(B); C_s = SMatrix{M, N}(C)
    noise_s = [SVector{K}(n) for n in noise]

    function _sim_scalar_static(
            A_vec, B_s, C_s, noise_s,
            ::Val{N_}, ::Val{M_}, ::Val{K_}
        ) where {N_, M_, K_}
        T_el = eltype(A_vec)
        A_d = SMatrix{N_, N_}(reshape(A_vec, N_, N_))
        B_d = SMatrix{N_, K_}(T_el.(B_s))
        C_d = SMatrix{M_, N_}(T_el.(C_s))
        u0_d = SVector{N_}(zeros(T_el, N_))
        prob = LinearStateSpaceProblem(
            A_d, B_d, u0_d, (0, length(noise_s));
            C = C_d, noise = noise_s
        )
        sol = solve(prob, DirectIteration())
        return sum(sol.u[end])
    end

    A_vec = collect(vec(Matrix(A)))
    f = a -> _sim_scalar_static(a, B_s, C_s, noise_s, Val(N), Val(M), Val(K))
    ForwardDiff.gradient(f, A_vec)

    @benchmarkable ForwardDiff.gradient($f, $(copy(A_vec)))
end

SIM_FD
