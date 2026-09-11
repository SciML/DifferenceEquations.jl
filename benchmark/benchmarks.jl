using DifferenceEquations, BenchmarkTools
using StableRNGs, LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Small linear state-space model (RBC-style): x_t = A x_{t-1} + B w_t, z_t = C x_t
nx, nw, nz, T = 3, 2, 2, 200
A = [0.9 0.1 0.0; 0.0 0.8 0.1; 0.0 0.0 0.5]
B = [0.5 0.0; 0.0 0.5; 0.0 0.2]
C = [1.0 0.0 0.0; 0.0 1.0 0.0]
D = Diagonal([0.1, 0.1])
u0 = zeros(nx)
noise = [randn(rng, nw) for _ in 1:T]
obs = [randn(rng, nz) for _ in 1:T]
u0_prior_mean = zeros(nx)
u0_prior_var = Matrix{Float64}(I, nx, nx)

prob_direct = LinearStateSpaceProblem(A, B, u0, (0, T); noise = noise)
prob_obs = LinearStateSpaceProblem(A, B, u0, (0, T); C = C, noise = noise)
prob_kalman = LinearStateSpaceProblem(
    A, B, u0, (0, T);
    C = C, observables_noise = D, observables = obs,
    u0_prior_mean = u0_prior_mean, u0_prior_var = u0_prior_var
)

# =============================================================================
# Solves
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["direct"] = @benchmarkable solve($prob_direct, DirectIteration())
SUITE["solve"]["direct_obs"] = @benchmarkable solve(
    $prob_obs, DirectIteration()
)
SUITE["solve"]["kalman"] = @benchmarkable solve($prob_kalman, KalmanFilter())

# =============================================================================
# Workspace reuse path
# =============================================================================

SUITE["workspace"] = BenchmarkGroup()

SUITE["workspace"]["init"] = @benchmarkable init($prob_obs, DirectIteration())
SUITE["workspace"]["solve!"] = @benchmarkable solve!(ws) setup = (
    ws = init($prob_obs, DirectIteration())
)
SUITE["workspace"]["solve!_kalman"] = @benchmarkable solve!(ws) setup = (
    ws = init($prob_kalman, KalmanFilter())
)
