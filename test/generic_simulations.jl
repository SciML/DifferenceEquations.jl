using DifferenceEquations, Distributions, LinearAlgebra, Test, Random
using DelimitedFiles, DiffEqBase

# =============================================================================
# Helper: create quadratic callbacks matching the old QuadraticStateSpaceProblem
# =============================================================================

"""
    make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)

Create transition and observation callbacks that reproduce the quadratic SSM:
    u_f(t+1) = A_1 * u_f(t) + B * w(t)
    u(t+1) = A_0 + A_1 * u(t) + quad(A_2, u_f(t)) + B * w(t)
    z(t) = C_0 + C_1 * u(t) + quad(C_2, u_f(t))

Returns (f!!, g!!) with captured mutable state for u_f tracking.
Each call creates fresh closures — safe for single-use per solve.
"""
function make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
    n_x = length(u0)
    n_obs = length(C_0)
    u_f = copy(u0)          # tracks linear-part state, initialized to u0
    u_f_new = similar(u0)   # workspace for updating u_f

    function f!!(x_next, x, w, p, t)
        # Compute new linear-part: u_f_new = A_1 * u_f + B * w
        mul!(u_f_new, A_1, u_f)
        mul!(u_f_new, B, w, 1.0, 1.0)

        # Full transition: x_next = A_0 + A_1 * x + quad(A_2, u_f) + B * w
        copyto!(x_next, A_0)
        mul!(x_next, A_1, x, 1.0, 1.0)
        @inbounds for i in 1:n_x
            x_next[i] += dot(u_f, view(A_2, i, :, :), u_f)
        end
        mul!(x_next, B, w, 1.0, 1.0)

        # Advance u_f for next step
        copyto!(u_f, u_f_new)

        return x_next
    end

    function g!!(y, x, p, t)
        # y = C_0 + C_1 * x + quad(C_2, u_f)
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
# RBC model matrices (linear)
# =============================================================================

A_rbc = [
    0.9568351489231076 6.209371005755285;
    3.0153731819288737e-18 0.20000000000000007
]
B_rbc = reshape([0.0; -0.01], 2, 1)
C_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
D_rbc = abs2.([0.1, 0.1])
u0_rbc = zeros(2)

observables_rbc = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ','
)' |> collect
noise_rbc = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ','
)' |> collect

# =============================================================================
# Tests: Linear callbacks match LinearStateSpaceProblem
# =============================================================================

@testset "Generic linear matches LinearStateSpaceProblem — with observations and noise" begin
    Random.seed!(1234)
    sol_linear = solve(LinearStateSpaceProblem(A_rbc, B_rbc, u0_rbc, (0, 5); C = C_rbc))

    linear_f!! = (x_next, x, w, p, t) -> begin
        mul!(x_next, p.A, x)
        mul!(x_next, p.B, w, 1.0, 1.0)
        return x_next
    end
    linear_g!! = (y, x, p, t) -> begin
        mul!(y, p.C, x)
        return y
    end
    p = (; A = A_rbc, B = B_rbc, C = C_rbc)

    Random.seed!(1234)
    sol_generic = solve(GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, 5), p;
        n_shocks = 1, n_obs = 2
    ))

    @test sol_linear.u ≈ sol_generic.u
    @test sol_linear.z ≈ sol_generic.z
    @test sol_linear.W ≈ sol_generic.W
    @test sol_linear.logpdf === nothing
    @test sol_generic.logpdf === nothing
end

@testset "Generic linear matches — with explicit noise and observables" begin
    T = 5
    obs = observables_rbc[:, 1:T]
    nse = noise_rbc[:, 1:T]

    sol_linear = solve(LinearStateSpaceProblem(
        A_rbc, B_rbc, u0_rbc, (0, T); C = C_rbc,
        observables_noise = D_rbc, noise = nse, observables = obs
    ))

    linear_f!! = (x_next, x, w, p, t) -> begin
        mul!(x_next, p.A, x)
        mul!(x_next, p.B, w, 1.0, 1.0)
        return x_next
    end
    linear_g!! = (y, x, p, t) -> begin
        mul!(y, p.C, x)
        return y
    end
    p = (; A = A_rbc, B = B_rbc, C = C_rbc)

    sol_generic = solve(GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, noise = nse, observables = obs
    ))

    @test sol_linear.u ≈ sol_generic.u
    @test sol_linear.z ≈ sol_generic.z
    @test sol_linear.logpdf ≈ sol_generic.logpdf
end

# =============================================================================
# Tests: No observation process
# =============================================================================

@testset "Generic no observation" begin
    Random.seed!(1234)
    linear_f!! = (x_next, x, w, p, t) -> begin
        mul!(x_next, p.A, x)
        mul!(x_next, p.B, w, 1.0, 1.0)
        return x_next
    end
    p = (; A = A_rbc, B = B_rbc)

    sol = solve(GenericStateSpaceProblem(
        linear_f!!, nothing, [1.0, 0.5], (0, 5), p;
        n_shocks = 1, n_obs = 0
    ))
    @test sol.z === nothing
    @test length(sol.u) == 6

    # Compare to LinearStateSpaceProblem with C=nothing
    Random.seed!(1234)
    sol_linear = solve(LinearStateSpaceProblem(
        A_rbc, B_rbc, [1.0, 0.5], (0, 5); C = nothing
    ))
    # Must use same seed → same random noise
    Random.seed!(1234)
    sol_generic = solve(GenericStateSpaceProblem(
        linear_f!!, nothing, [1.0, 0.5], (0, 5), p;
        n_shocks = 1, n_obs = 0
    ))
    @test sol_linear.u ≈ sol_generic.u
end

# =============================================================================
# Tests: No noise (n_shocks = 0)
# =============================================================================

@testset "Generic no noise" begin
    linear_f!! = (x_next, x, w, p, t) -> begin
        mul!(x_next, p.A, x)
        return x_next
    end
    linear_g!! = (y, x, p, t) -> begin
        mul!(y, p.C, x)
        return y
    end
    p = (; A = A_rbc, C = C_rbc)

    sol = solve(GenericStateSpaceProblem(
        linear_f!!, linear_g!!, [1.0, 0.5], (0, 5), p;
        n_shocks = 0, n_obs = 2
    ))

    @test sol.W === nothing
    @test length(sol.u) == 6
    @test length(sol.z) == 6

    # Compare to LinearStateSpaceProblem with B=nothing
    sol_linear = solve(LinearStateSpaceProblem(
        A_rbc, nothing, [1.0, 0.5], (0, 5); C = C_rbc
    ))
    @test sol_linear.u ≈ sol.u
    @test sol_linear.z ≈ sol.z
end

# =============================================================================
# Tests: Observation noise
# =============================================================================

@testset "Generic with observation noise" begin
    T = 20
    B_no_noise = reshape([0.0; 0.0], 2, 1)
    u0 = [1.0, 0.5]

    linear_f!! = (x_next, x, w, p, t) -> begin
        mul!(x_next, p.A, x)
        mul!(x_next, p.B, w, 1.0, 1.0)
        return x_next
    end
    linear_g!! = (y, x, p, t) -> begin
        mul!(y, p.C, x)
        return y
    end
    p = (; A = A_rbc, B = B_no_noise, C = C_rbc)

    sol_no_noise = solve(GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0, (0, T), p;
        n_shocks = 1, n_obs = 2
    ))

    sol_obs_noise = solve(GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0, (0, T), p;
        n_shocks = 1, n_obs = 2, observables_noise = D_rbc
    ))

    # Tiny observation noise → nearly deterministic
    sol_tiny = solve(GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0, (0, T), p;
        n_shocks = 1, n_obs = 2, observables_noise = [1.0e-16, 1.0e-16]
    ))
    @test maximum(maximum.(sol_tiny.z - sol_no_noise.z)) < 1.0e-7
    @test maximum(maximum.(sol_tiny.z - sol_no_noise.z)) > 0.0
end

# =============================================================================
# Tests: Quadratic callbacks — RBC model
# =============================================================================

# Quadratic RBC model matrices
A_0_rbc = [-7.824904812740593e-5, 0.0]
A_1_rbc = [0.9568351489231076 6.209371005755285; 3.0153731819288737e-18 0.20000000000000007]
A_2_rbc = cat(
    [-0.00019761505863889124 0.03375055315837927; 0.0 0.0],
    [0.03375055315837913 3.128758481817603; 0.0 0.0]; dims = 3
)
B_2_rbc = reshape([0.0; -0.01], 2, 1)
C_0_rbc = [7.824904812740593e-5, 0.0]
C_1_rbc = [0.09579643002426148 0.6746869652592109; 1.0 0.0]
C_2_rbc = cat(
    [-0.00018554166974717046 0.0025652363153049716; 0.0 0.0],
    [0.002565236315304951 0.3132705036896446; 0.0 0.0]; dims = 3
)
D_2_rbc = abs2.([0.1, 0.1])
u0_2_rbc = zeros(2)

observables_2_rbc = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_observables.csv"), ','
)' |> collect

@testset "Quadratic RBC basic inference, simulated noise" begin
    f!!, g!! = make_quadratic_callbacks(
        A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_2_rbc
    )
    prob = GenericStateSpaceProblem(
        f!!, g!!, u0_2_rbc, (0, size(observables_2_rbc, 2));
        n_shocks = 1, n_obs = 2,
        observables_noise = D_2_rbc, observables = observables_2_rbc
    )
    sol = solve(prob)
    @test sol.logpdf isa Number
end

@testset "Quadratic RBC simulation, no observations" begin
    T = 20
    f!!, g!! = make_quadratic_callbacks(
        A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_2_rbc
    )
    prob = GenericStateSpaceProblem(
        f!!, g!!, u0_2_rbc, (0, T);
        n_shocks = 1, n_obs = 2
    )
    sol = solve(prob)
    @test length(sol.u) == T + 1
    @test length(sol.z) == T + 1
    @test sol.logpdf === nothing
end

@testset "Quadratic RBC deterministic with observation noise" begin
    T = 20
    B_no_noise = reshape([0.0; 0.0], 2, 1)
    u0 = [1.0, 0.5]

    f_nn!!, g_nn!! = make_quadratic_callbacks(
        A_0_rbc, A_1_rbc, A_2_rbc, B_no_noise, C_0_rbc, C_1_rbc, C_2_rbc, u0
    )
    sol_no_noise = solve(GenericStateSpaceProblem(
        f_nn!!, g_nn!!, u0, (0, T);
        n_shocks = 1, n_obs = 2
    ))

    f_on!!, g_on!! = make_quadratic_callbacks(
        A_0_rbc, A_1_rbc, A_2_rbc, B_no_noise, C_0_rbc, C_1_rbc, C_2_rbc, u0
    )
    sol_obs_noise = solve(GenericStateSpaceProblem(
        f_on!!, g_on!!, u0, (0, T);
        n_shocks = 1, n_obs = 2, observables_noise = D_2_rbc
    ))

    f_ti!!, g_ti!! = make_quadratic_callbacks(
        A_0_rbc, A_1_rbc, A_2_rbc, B_no_noise, C_0_rbc, C_1_rbc, C_2_rbc, u0
    )
    sol_tiny = solve(GenericStateSpaceProblem(
        f_ti!!, g_ti!!, u0, (0, T);
        n_shocks = 1, n_obs = 2, observables_noise = [1.0e-16, 1.0e-16]
    ))
    @test maximum(maximum.(sol_tiny.z - sol_no_noise.z)) < 1.0e-7
    @test maximum(maximum.(sol_tiny.z - sol_no_noise.z)) > 0.0
end

# =============================================================================
# Tests: Quadratic likelihood regression values
# =============================================================================

function quadratic_joint_likelihood(
        A_0, A_1, A_2, B, C_0, C_1, C_2, u0, noise, observables, D;
        kwargs...
    )
    f!!, g!! = make_quadratic_callbacks(A_0, A_1, A_2, B, C_0, C_1, C_2, u0)
    problem = GenericStateSpaceProblem(
        f!!, g!!, u0, (0, size(observables, 2));
        n_shocks = size(B, 2), n_obs = length(C_0),
        observables_noise = D, noise = noise, observables = observables,
        kwargs...
    )
    return solve(problem).logpdf
end

noise_2_rbc = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/RBC_noise.csv"), ','
)' |> collect
T_rbc = 5
observables_2_rbc_short = observables_2_rbc[:, 1:T_rbc]
noise_2_rbc_short = noise_2_rbc[:, 1:T_rbc]

@testset "Quadratic RBC basic inference with known noise" begin
    f!!, g!! = make_quadratic_callbacks(
        A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc, u0_2_rbc
    )
    prob = GenericStateSpaceProblem(
        f!!, g!!, u0_2_rbc, (0, size(observables_2_rbc_short, 2));
        n_shocks = 1, n_obs = 2,
        observables_noise = D_2_rbc, noise = noise_2_rbc_short,
        observables = observables_2_rbc_short
    )
    DiffEqBase.get_concrete_problem(prob, false)
    sol = solve(prob)
end

@testset "Quadratic RBC joint likelihood" begin
    @test quadratic_joint_likelihood(
        A_0_rbc, A_1_rbc, A_2_rbc, B_2_rbc, C_0_rbc, C_1_rbc, C_2_rbc,
        u0_2_rbc, noise_2_rbc_short, observables_2_rbc_short, D_2_rbc
    ) ≈ -690.81094364573
end

# Load FVGQ data
A_0_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_0.csv"), ',')
A_0_FVGQ = vec(A_0_raw)
A_1_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_1.csv"), ',')
A_2_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_A_2.csv"), ',')
A_2_FVGQ = reshape(A_2_raw, length(A_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
B_2_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_B.csv"), ',')
C_0_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_0.csv"), ',')
C_0_FVGQ = vec(C_0_raw)
C_1_FVGQ = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_1.csv"), ',')
C_2_raw = readdlm(joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_C_2.csv"), ',')
C_2_FVGQ = reshape(C_2_raw, length(C_0_FVGQ), length(A_0_FVGQ), length(A_0_FVGQ))
D_2_FVGQ = ones(6) * 1.0e-3
u0_2_FVGQ = zeros(size(A_1_FVGQ, 1))

observables_2_FVGQ = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_observables.csv"), ','
)' |> collect
noise_2_FVGQ = readdlm(
    joinpath(pkgdir(DifferenceEquations), "test/data/FVGQ20_noise.csv"), ','
)' |> collect

@testset "Quadratic FVGQ joint likelihood" begin
    @test quadratic_joint_likelihood(
        A_0_FVGQ, A_1_FVGQ, A_2_FVGQ, B_2_FVGQ, C_0_FVGQ, C_1_FVGQ, C_2_FVGQ,
        u0_2_FVGQ, noise_2_FVGQ, observables_2_FVGQ, D_2_FVGQ
    ) ≈ -1.4728927648336522e7
end

# =============================================================================
# Tests: StaticArrays with !! pattern (matching differentiable_economics)
# =============================================================================

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

@testset "Generic !! callbacks — mutable vs static consistency" begin
    A_m = [0.9 0.1; 0.0 0.8]
    B_m = reshape([0.0; 0.1], 2, 1)
    C_m = [1.0 0.0; 0.0 1.0]
    u0_m = [0.5, 0.3]
    noise_vals = [randn(1) for _ in 1:9]

    # Mutable version
    p_m = (; A = A_m, B = B_m, C = C_m)
    prob_m = GenericStateSpaceProblem(
        f_lss!!, g_lss!!, u0_m, (0, 9), p_m;
        n_shocks = 1, n_obs = 2, noise = noise_vals
    )
    sol_m = solve(prob_m)

    # Static version — same callbacks, same data, just wrapped in SMatrix/SVector
    A_s = SMatrix{2, 2}(A_m)
    B_s = SMatrix{2, 1}(B_m)
    C_s = SMatrix{2, 2}(C_m)
    u0_s = SVector{2}(u0_m)
    noise_s = [SVector{1}(n) for n in noise_vals]

    p_s = (; A = A_s, B = B_s, C = C_s)
    prob_s = GenericStateSpaceProblem(
        f_lss!!, g_lss!!, u0_s, (0, 9), p_s;
        n_shocks = 1, n_obs = 2, noise = noise_s
    )
    sol_s = solve(prob_s)

    # Results must match exactly
    for t in eachindex(sol_m.u)
        @test Vector(sol_s.u[t]) ≈ sol_m.u[t]
    end
    for t in eachindex(sol_m.z)
        @test Vector(sol_s.z[t]) ≈ sol_m.z[t]
    end

    # Verify static types are preserved
    @test eltype(sol_s.u) <: SVector{2, Float64}
    @test eltype(sol_s.z) <: SVector{2, Float64}
end

@testset "Generic !! callbacks — static matches LinearStateSpaceProblem" begin
    A = @SMatrix [0.9 0.1; 0.0 0.8]
    B = @SMatrix [0.0; 0.1;;]
    C = @SMatrix [1.0 0.0; 0.0 1.0]
    u0 = @SVector [0.5, 0.3]
    noise = [SVector{1, Float64}(randn()) for _ in 1:9]

    prob_linear = LinearStateSpaceProblem(A, B, u0, (0, 9); C = C, noise = noise)
    sol_linear = solve(prob_linear)

    p = (; A = A, B = B, C = C)
    prob_generic = GenericStateSpaceProblem(
        f_lss!!, g_lss!!, u0, (0, 9), p;
        n_shocks = 1, n_obs = 2, noise = noise
    )
    sol_generic = solve(prob_generic)

    for t in eachindex(sol_linear.u)
        @test sol_linear.u[t] ≈ sol_generic.u[t]
    end
    for t in eachindex(sol_linear.z)
        @test sol_linear.z[t] ≈ sol_generic.z[t]
    end
end

@testset "Generic !! callbacks — static no noise" begin
    A = @SMatrix [0.9 0.1; 0.0 0.8]
    C = @SMatrix [1.0 0.0; 0.0 1.0]
    u0 = @SVector [1.0, 0.5]

    prob_linear = LinearStateSpaceProblem(A, nothing, u0, (0, 5); C = C)
    sol_linear = solve(prob_linear)

    # f_lss!! handles w=nothing via muladd!!(x_p, B, nothing) → x_p
    p = (; A = A, B = nothing, C = C)
    prob_generic = GenericStateSpaceProblem(
        f_lss!!, g_lss!!, u0, (0, 5), p;
        n_shocks = 0, n_obs = 2
    )
    sol_generic = solve(prob_generic)

    for t in eachindex(sol_linear.u)
        @test sol_linear.u[t] ≈ sol_generic.u[t]
    end
    for t in eachindex(sol_linear.z)
        @test sol_linear.z[t] ≈ sol_generic.z[t]
    end
end

# =============================================================================
# Tests: init/solve! cache reuse
# =============================================================================

@testset "Generic init/solve! matches solve" begin
    T = 5
    obs = observables_rbc[:, 1:T]
    nse = noise_rbc[:, 1:T]

    linear_f!! = (x_next, x, w, p, t) -> begin
        mul!(x_next, p.A, x)
        mul!(x_next, p.B, w, 1.0, 1.0)
        return x_next
    end
    linear_g!! = (y, x, p, t) -> begin
        mul!(y, p.C, x)
        return y
    end
    p = (; A = A_rbc, B = B_rbc, C = C_rbc)

    prob = GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, noise = nse, observables = obs
    )

    sol_direct = solve(prob)
    ws = init(prob, DirectIteration())
    sol_ws = solve!(ws)

    @test sol_ws.u ≈ sol_direct.u
    @test sol_ws.z ≈ sol_direct.z
    @test sol_ws.logpdf ≈ sol_direct.logpdf
end

@testset "Generic repeated solve! gives consistent results" begin
    T = 5
    obs = observables_rbc[:, 1:T]
    nse = noise_rbc[:, 1:T]

    linear_f!! = (x_next, x, w, p, t) -> begin
        mul!(x_next, p.A, x)
        mul!(x_next, p.B, w, 1.0, 1.0)
        return x_next
    end
    linear_g!! = (y, x, p, t) -> begin
        mul!(y, p.C, x)
        return y
    end
    p = (; A = A_rbc, B = B_rbc, C = C_rbc)

    prob = GenericStateSpaceProblem(
        linear_f!!, linear_g!!, u0_rbc, (0, T), p;
        n_shocks = 1, n_obs = 2,
        observables_noise = D_rbc, noise = nse, observables = obs
    )

    ws = init(prob, DirectIteration())
    sol1 = solve!(ws)
    sol2 = solve!(ws)

    @test sol1.u ≈ sol2.u
    @test sol1.z ≈ sol2.z
    @test sol1.logpdf ≈ sol2.logpdf
end
