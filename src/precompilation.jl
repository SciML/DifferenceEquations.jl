using PrecompileTools
using LinearAlgebra: I

@setup_workload begin
    # Minimal setup data for precompilation workload
    # Use simple 2x2 system that's typical for state-space models

    @compile_workload begin
        # Common matrices for state-space models (2x2 system)
        A = [0.9 0.1; 0.0 0.8]
        B = reshape([0.0; 0.1], 2, 1)
        C = [1.0 0.0; 0.0 1.0]
        D = [0.01, 0.01]
        u0 = [0.0, 0.0]
        T = 10

        # === LinearStateSpaceProblem with DirectIteration (most common) ===
        # Simulation without observations
        prob_sim = LinearStateSpaceProblem(A, B, u0, (0, T))
        sol_sim = solve(prob_sim)

        # Simulation with observation equation
        prob_obs = LinearStateSpaceProblem(A, B, u0, (0, T); C = C)
        sol_obs = solve(prob_obs)

        # Simulation with observation noise
        prob_noise = LinearStateSpaceProblem(A, B, u0, (0, T); C = C, observables_noise = D)
        sol_noise = solve(prob_noise)

        # === LinearStateSpaceProblem with KalmanFilter ===
        # Generate fake observables for Kalman filter
        # For tspan = (0, T), we get T+1 time points, so need T observables
        observables = randn(2, T)
        u0_prior_mean = zeros(2)
        u0_prior_var = Matrix{Float64}(I, 2, 2)

        prob_kalman = LinearStateSpaceProblem(
            A, B, u0, (0, size(observables, 2));
            C = C, observables_noise = D, observables = observables,
            u0_prior_mean = u0_prior_mean, u0_prior_var = u0_prior_var
        )
        sol_kalman = solve(prob_kalman)

        # === LinearStateSpaceProblem with no noise matrix ===
        prob_no_noise = LinearStateSpaceProblem(A, nothing, u0, (0, T); C = C)
        sol_no_noise = solve(prob_no_noise)

        # === QuadraticStateSpaceProblem with DirectIteration ===
        # Use proper dimensions: B has 1 column, so noise needs 1 row
        # For tspan = (0, T), we get T+1 time points, so need T noise samples
        A_0 = zeros(2)
        A_1 = A
        A_2 = zeros(2, 2, 2)
        C_0 = zeros(2)
        C_1 = C
        C_2 = zeros(2, 2, 2)
        noise_quad = randn(1, T)

        prob_quad = QuadraticStateSpaceProblem(
            A_0, A_1, A_2, B, u0, (0, T);
            C_0 = C_0, C_1 = C_1, C_2 = C_2, noise = noise_quad
        )
        sol_quad = solve(prob_quad)
    end
end
