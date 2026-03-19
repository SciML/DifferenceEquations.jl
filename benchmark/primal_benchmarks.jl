# Primal-only benchmarks for DifferenceEquations.jl
# This script benchmarks only forward solves (no AD/gradients).
# Can be run on any branch to compare primal performance.
#
# Usage: julia --project=benchmark benchmark/primal_benchmarks.jl

using DifferenceEquations, BenchmarkTools
using DelimitedFiles, Distributions, LinearAlgebra

# Check if MKL or not
julia_mkl = any(contains("mkl"), getfield.(LinearAlgebra.BLAS.get_config().loaded_libs, :libname))

if !julia_mkl
    openblas_threads = min(4, Int64(round(Sys.CPU_THREADS / 2)))
    BLAS.set_num_threads(openblas_threads)
end

println("Julia $(VERSION)")
println("Threads.nthreads = $(Threads.nthreads()), MKL = $julia_mkl, BLAS.num_threads = $(BLAS.get_num_threads())")
println()

# Helper functions — Linear
function joint_likelihood_1(A, B, C, u0, noise, observables, D; kwargs...)
    prob = LinearStateSpaceProblem(
        A, B, u0, (0, size(observables, 2)); C,
        observables_noise = D, noise, observables, kwargs...
    )
    return solve(prob).logpdf
end

function kalman_likelihood(A, B, C, u0_prior_var, observables, D; kwargs...)
    prob = LinearStateSpaceProblem(
        A, B, zeros(size(A, 1)), (0, size(observables, 2)); C,
        u0_prior_var, u0_prior_mean = zeros(size(A, 1)),
        observables_noise = D, noise = nothing, observables, kwargs...
    )
    return solve(prob).logpdf
end

function simulate_model_no_noise_1(A, B, C, u0, observables, D; kwargs...)
    prob = LinearStateSpaceProblem(
        A, B, u0, (0, size(observables, 2)); C,
        observables_noise = D, observables, kwargs...
    )
    return solve(prob).retcode
end

function simulate_model_no_observations_1(A, B, C, u0, T; kwargs...)
    prob = LinearStateSpaceProblem(A, B, u0, (0, T); C, kwargs...)
    return solve(prob).retcode
end

# Helper — Quadratic via GenericStateSpaceProblem
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

function joint_likelihood_2(A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, observables, D; kwargs...)
    f!!, g!! = make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
    prob = GenericStateSpaceProblem(
        f!!, g!!, u0, (0, size(observables, 2));
        n_shocks = size(B, 2), n_obs = length(C_0),
        observables_noise = D, noise = noise, observables = observables, kwargs...
    )
    return solve(prob).logpdf
end

# ──── Linear model data ────

const A_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
const B_rbc = reshape([0.0; -0.01], 2, 1)
const C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
const D_rbc = abs2.([0.1, 0.1])
const u0_rbc = zeros(2)
const u0_prior_var_rbc = diagm(ones(length(u0_rbc)))

const observables_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ',')' |> collect
const noise_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ',')' |> collect
const T_rbc = size(observables_rbc, 2)

const A_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A.csv"), ',')
const B_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
const C_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C.csv"), ',')
const D_FVGQ = abs2.(ones(6) * 1.0e-3)
const observables_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_observables.csv"), ',')' |> collect
const noise_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"), ',')' |> collect
const u0_FVGQ = zeros(size(A_FVGQ, 1))
const u0_prior_var_FVGQ = diagm(ones(length(u0_FVGQ)))
const T_FVGQ = size(observables_FVGQ, 2)

# ──── Quadratic model data ────

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
const observables_2_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ',')' |> collect
const noise_2_rbc = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ',')' |> collect

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
const observables_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_observables.csv"), ',')' |> collect
const noise_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"), ',')' |> collect

# ──── Warmup ────

joint_likelihood_1(A_rbc, B_rbc, C_rbc, u0_rbc, noise_rbc, observables_rbc, D_rbc)
kalman_likelihood(A_rbc, B_rbc, C_rbc, u0_prior_var_rbc, observables_rbc, D_rbc)
joint_likelihood_1(A_FVGQ, B_FVGQ, C_FVGQ, u0_FVGQ, noise_FVGQ, observables_FVGQ, D_FVGQ)
kalman_likelihood(A_FVGQ, B_FVGQ, C_FVGQ, u0_prior_var_FVGQ, observables_FVGQ, D_FVGQ)
joint_likelihood_2(A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_2_rbc, noise_2_rbc, observables_2_rbc, D_2_rbc)
joint_likelihood_2(A_0_FVGQ, A_1_FVGQ, A_2_FVGQ, B_2_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ, u0_2_FVGQ, noise_2_FVGQ, observables_2_FVGQ, D_2_FVGQ)

# ──── Run benchmarks ────

println("=" ^ 70)
println("LINEAR MODEL — RBC (2×2, T=$(T_rbc))")
println("=" ^ 70)

print("  joint likelihood (DirectIteration):  ")
display(@benchmark joint_likelihood_1($A_rbc, $B_rbc, $C_rbc, $u0_rbc, $noise_rbc, $observables_rbc, $D_rbc))

print("  kalman likelihood (KalmanFilter):    ")
display(@benchmark kalman_likelihood($A_rbc, $B_rbc, $C_rbc, $u0_prior_var_rbc, $observables_rbc, $D_rbc))

print("  simulate (no noise):                 ")
display(@benchmark simulate_model_no_noise_1($A_rbc, $B_rbc, $C_rbc, $u0_rbc, $observables_rbc, $D_rbc))

print("  simulate (no observations):          ")
display(@benchmark simulate_model_no_observations_1($A_rbc, $B_rbc, $C_rbc, $u0_rbc, $T_rbc))

println()
println("=" ^ 70)
println("LINEAR MODEL — FVGQ ($(size(A_FVGQ,1))×$(size(A_FVGQ,2)), T=$(T_FVGQ))")
println("=" ^ 70)

print("  joint likelihood (DirectIteration):  ")
display(@benchmark joint_likelihood_1($A_FVGQ, $B_FVGQ, $C_FVGQ, $u0_FVGQ, $noise_FVGQ, $observables_FVGQ, $D_FVGQ))

print("  kalman likelihood (KalmanFilter):    ")
display(@benchmark kalman_likelihood($A_FVGQ, $B_FVGQ, $C_FVGQ, $u0_prior_var_FVGQ, $observables_FVGQ, $D_FVGQ))

print("  simulate (no noise):                 ")
display(@benchmark simulate_model_no_noise_1($A_FVGQ, $B_FVGQ, $C_FVGQ, $u0_FVGQ, $observables_FVGQ, $D_FVGQ))

print("  simulate (no observations):          ")
display(@benchmark simulate_model_no_observations_1($A_FVGQ, $B_FVGQ, $C_FVGQ, $u0_FVGQ, $T_FVGQ))

println()
println("=" ^ 70)
println("QUADRATIC (Generic) MODEL — RBC (2×2, T=$(size(observables_2_rbc, 2)))")
println("=" ^ 70)

print("  joint likelihood (DirectIteration):  ")
display(@benchmark joint_likelihood_2($A_0_rbc, $A_1_rbc, $A_2_rbc, $B_2_rbc, $C_0_rbc, $C_1_rbc, $C_2_rbc, $u0_2_rbc, $noise_2_rbc, $observables_2_rbc, $D_2_rbc))

println()
println("=" ^ 70)
println("QUADRATIC (Generic) MODEL — FVGQ ($(size(A_1_FVGQ,1))×$(size(A_1_FVGQ,2)), T=$(size(observables_2_FVGQ, 2)))")
println("=" ^ 70)

print("  joint likelihood (DirectIteration):  ")
display(@benchmark joint_likelihood_2($A_0_FVGQ, $A_1_FVGQ, $A_2_FVGQ, $B_2_FVGQ, $C_0_FVGQ, $C_1_FVGQ, $C_2_FVGQ, $u0_2_FVGQ, $noise_2_FVGQ, $observables_2_FVGQ, $D_2_FVGQ))

# ──── Static arrays benchmarks ────

using StaticArrays
using DifferenceEquations: mul!!, muladd!!

# Single set of callbacks that work for BOTH mutable and static arrays
@inline function f_lss!!(x_p, x, w, p, t)
    x_p = mul!!(x_p, p.A, x)
    return muladd!!(x_p, p.B, w)
end

@inline function g_lss!!(y, x, p, t)
    return mul!!(y, p.C, x)
end

function bench_generic_mutable(f!!, g!!, u0, p, nse, obs, D)
    prob = GenericStateSpaceProblem(f!!, g!!, u0, (0, size(obs, 2)), p;
        n_shocks = size(p.B, 2), n_obs = size(p.C, 1),
        observables_noise = D, noise = nse, observables = obs)
    return solve(prob).logpdf
end

function bench_generic_static(f!!, g!!, u0, p, nse, obs, D, n_shocks, n_obs)
    prob = GenericStateSpaceProblem(f!!, g!!, u0, (0, size(obs, 2)), p;
        n_shocks = n_shocks, n_obs = n_obs,
        observables_noise = D, noise = nse, observables = obs)
    return solve(prob).logpdf
end

# Mutable data
const p_m = (; A = A_rbc, B = B_rbc, C = C_rbc)

# Static data — same values wrapped in SMatrix/SVector
const A_s = SMatrix{2, 2}(A_rbc)
const B_s = SMatrix{2, 1}(B_rbc)
const C_s = SMatrix{2, 2}(C_rbc)
const u0_s = SVector{2}(u0_rbc)
const D_s = SVector{2}(D_rbc)
const obs_s = observables_rbc  # kept as Matrix (observables are external data)
const nse_s = [SVector{1}(noise_rbc[:, t]) for t in 1:size(noise_rbc, 2)]
const p_s = (; A = A_s, B = B_s, C = C_s)

# Warmup
bench_generic_mutable(f_lss!!, g_lss!!, u0_rbc, p_m, noise_rbc, observables_rbc, D_rbc)
bench_generic_static(f_lss!!, g_lss!!, u0_s, p_s, nse_s, obs_s, D_rbc, 1, 2)

println()
println("=" ^ 70)
println("GENERIC LINEAR — MUTABLE vs STATIC (same !! callbacks, RBC 2×2, T=$(T_rbc))")
println("=" ^ 70)

print("  mutable (Vector/Matrix):   ")
display(@benchmark bench_generic_mutable($f_lss!!, $g_lss!!, $u0_rbc, $p_m, $noise_rbc, $observables_rbc, $D_rbc))

print("  static  (SVector/SMatrix): ")
display(@benchmark bench_generic_static($f_lss!!, $g_lss!!, $u0_s, $p_s, $nse_s, $obs_s, $D_rbc, 1, 2))

println()
println("Done.")
