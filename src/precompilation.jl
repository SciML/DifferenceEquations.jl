using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload
using LinearAlgebra: I

@setup_workload begin
    @compile_workload begin
        # Common matrices for state-space models (2x2 system)
        A = [0.9 0.1; 0.0 0.8]
        B = reshape([0.0; 0.1], 2, 1)
        C = [1.0 0.0; 0.0 1.0]
        D = [0.01, 0.01]
        u0 = [0.0, 0.0]
        T = 10

        # === LinearStateSpaceProblem with DirectIteration ===
        # Simulation without observations
        prob_sim = LinearStateSpaceProblem(A, B, u0, (0, T))
        sol_sim = solve(prob_sim)

        # Simulation with observation equation
        prob_obs = LinearStateSpaceProblem(A, B, u0, (0, T); C = C)
        sol_obs = solve(prob_obs)

        # Simulation with observation noise
        prob_noise = LinearStateSpaceProblem(A, B, u0, (0, T); C = C, observables_noise = D)
        sol_noise = solve(prob_noise)

        # === init/solve! API ===
        ws = CommonSolve.init(prob_obs, DirectIteration())
        sol_ws = CommonSolve.solve!(ws)

        # === LinearStateSpaceProblem with KalmanFilter ===
        observables = [randn(2) for _ in 1:T]
        u0_prior_mean = zeros(2)
        u0_prior_var = Matrix{Float64}(I, 2, 2)

        prob_kalman = LinearStateSpaceProblem(
            A, B, u0, (0, length(observables));
            C = C, observables_noise = D, observables = observables,
            u0_prior_mean = u0_prior_mean, u0_prior_var = u0_prior_var
        )
        sol_kalman = solve(prob_kalman)

        # Kalman init/solve!
        ws_k = CommonSolve.init(prob_kalman, KalmanFilter())
        sol_k = CommonSolve.solve!(ws_k)

        # === LinearStateSpaceProblem with no noise matrix ===
        prob_no_noise = LinearStateSpaceProblem(A, nothing, u0, (0, T); C = C)
        sol_no_noise = solve(prob_no_noise)

        # === StateSpaceProblem with DirectIteration ===
        gen_f!! = (x_next, x, w, p, t) -> begin
            mul!(x_next, A, x)
            mul!(x_next, B, w, 1.0, 1.0)
            return x_next
        end
        gen_g!! = (y, x, p, t) -> begin
            mul!(y, C, x)
            return y
        end
        prob_gen = StateSpaceProblem(
            gen_f!!, gen_g!!, u0, (0, T);
            n_shocks = 1, n_obs = 2,
            syms = (:x1, :x2), obs_syms = (:y1, :y2)
        )
        sol_gen = solve(prob_gen)
        sol_gen[:x1]   # precompile state indexing
        sol_gen[:y1]   # precompile obs indexing

        # Generic init/solve!
        ws_gen = CommonSolve.init(prob_gen, DirectIteration())
        sol_gen_ws = CommonSolve.solve!(ws_gen)

        # Generic without observations
        prob_gen_no_obs = StateSpaceProblem(
            gen_f!!, nothing, u0, (0, T);
            n_shocks = 1, n_obs = 0
        )
        sol_gen_no_obs = solve(prob_gen_no_obs)

        # === StaticArrays workload ===
        A_s = SMatrix{2, 2}(0.9, 0.0, 0.1, 0.8)
        B_s = SMatrix{2, 1}(0.0, 0.1)
        C_s = SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0)
        u0_s = SVector{2}(0.0, 0.0)
        noise_s = [SVector{1}(randn()) for _ in 1:T]

        prob_s = LinearStateSpaceProblem(A_s, B_s, u0_s, (0, T); C = C_s, noise = noise_s)
        sol_s = solve(prob_s)
    end
end
