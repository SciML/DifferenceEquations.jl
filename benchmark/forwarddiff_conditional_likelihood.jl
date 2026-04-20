# ForwardDiff AD benchmarks for ConditionalLikelihood
# Returns CL_FD BenchmarkGroup

using ForwardDiff
using DifferenceEquations: init, solve!, StateSpaceWorkspace

const CL_FD = BenchmarkGroup()
CL_FD["gradient"] = BenchmarkGroup()

# =============================================================================
# Type promotion helper
# =============================================================================

_fd_promote_cl(::Type{T}, x::AbstractArray{T}) where {T} = x
_fd_promote_cl(::Type{T}, x::AbstractArray) where {T} = T.(x)

# =============================================================================
# Problem sizes (same as enzyme_conditional_likelihood.jl)
# =============================================================================

# CL requires fully-observed state: M = N
const p_cl_fd_small = (; N = 5, M = 5, T = 10)
const p_cl_fd_large = (; N = 30, M = 30, T = 100)

# =============================================================================
# Problem setup
# =============================================================================

function make_cl_fd_benchmark(p; seed = 42)
    (; N, M, T) = p
    Random.seed!(seed)
    A_raw = randn(N, N)
    A = 0.5 * A_raw / maximum(abs.(eigvals(A_raw)))
    C = randn(M, N)
    H = 0.1 * randn(M, M)
    R = H * H'

    x = zeros(N)
    y = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        x = A * x + 0.1 * randn(N)
        y[t] = C * x + H * randn(M)
    end

    return (; A, C, H, R, y)
end

# =============================================================================
# ForwardDiff wrapper — gradient of loglik w.r.t. vec(A)
# =============================================================================

function cl_loglik_fd_bench(A_vec, C, H, y, N)
    T_el = eltype(A_vec)
    A = reshape(A_vec, N, N)
    R = _fd_promote_cl(T_el, H) * _fd_promote_cl(T_el, H)'
    prob = LinearStateSpaceProblem(
        A, nothing,
        zeros(T_el, N), (0, length(y));
        C = _fd_promote_cl(T_el, C),
        observables_noise = R,
        observables = y,
    )
    sol = solve(prob, ConditionalLikelihood())
    return sol.logpdf
end

function fd_gradient_cl!(A_vec, C, H, y, N)
    return ForwardDiff.gradient(
        a -> cl_loglik_fd_bench(a, C, H, y, N), A_vec
    )
end

# =============================================================================
# Instantiate problems
# =============================================================================

const cl_fd_s = make_cl_fd_benchmark(p_cl_fd_small)
const cl_fd_l = make_cl_fd_benchmark(p_cl_fd_large)

# =============================================================================
# Warmup and benchmarks
# =============================================================================

fd_gradient_cl!(
    vec(copy(cl_fd_s.A)), cl_fd_s.C, cl_fd_s.H, cl_fd_s.y, p_cl_fd_small.N
)
fd_gradient_cl!(
    vec(copy(cl_fd_l.A)), cl_fd_l.C, cl_fd_l.H, cl_fd_l.y, p_cl_fd_large.N
)

CL_FD["gradient"]["small_mutable"] = @benchmarkable fd_gradient_cl!(
    $(vec(copy(cl_fd_s.A))), $(cl_fd_s.C), $(cl_fd_s.H),
    $(cl_fd_s.y), $(p_cl_fd_small.N)
)

CL_FD["gradient"]["large_mutable"] = @benchmarkable fd_gradient_cl!(
    $(vec(copy(cl_fd_l.A))), $(cl_fd_l.C), $(cl_fd_l.H),
    $(cl_fd_l.y), $(p_cl_fd_large.N)
)

# =============================================================================
# StaticArrays variant (small only)
# =============================================================================

# StaticArrays CL: no C matrix (identity observation, state = obs)
CL_FD["gradient"]["small_static"] = let
    (; A, H, y) = cl_fd_s
    N = p_cl_fd_small.N

    # For CL without C, observables must be state-dimensional
    H_s = SMatrix{N, N}(0.1 * I(N))
    y_s = [SVector{N}(yi) for yi in y]

    function _cl_loglik_static(
            A_vec, H_s, y_s,
            ::Val{N_}
        ) where {N_}
        T_el = eltype(A_vec)
        A_d = SMatrix{N_, N_}(reshape(A_vec, N_, N_))
        H_d = SMatrix{N_, N_}(T_el.(H_s))
        R_d = H_d * H_d'
        u0_d = SVector{N_}(zeros(T_el, N_))
        prob = LinearStateSpaceProblem(
            A_d, nothing, u0_d, (0, length(y_s));
            observables_noise = R_d,
            observables = y_s,
        )
        sol = solve(prob, ConditionalLikelihood())
        return sol.logpdf
    end

    _cl_fd_static_grad(a) = _cl_loglik_static(a, H_s, y_s, Val(N))

    A_vec = collect(vec(Matrix(A)))
    ForwardDiff.gradient(_cl_fd_static_grad, A_vec)

    @benchmarkable ForwardDiff.gradient($_cl_fd_static_grad, $(copy(A_vec)))
end

CL_FD
