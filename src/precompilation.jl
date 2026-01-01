using PrecompileTools

@setup_workload begin
    # Minimal setup - small matrices to keep precompilation fast
    N = 3  # state dimension
    M = 2  # observation dimension
    K = 2  # noise dimension
    T = 5  # short time horizon

    A = [0.9 0.1 0.0; 0.0 0.8 0.1; 0.0 0.0 0.7]
    B = [0.1 0.0; 0.0 0.1; 0.1 0.1]
    C = [1.0 0.0 0.0; 0.0 1.0 0.0]
    u0 = [1.0, 0.5, 0.3]
    tspan = (0.0, Float64(T - 1))

    # For Kalman filter
    u0_prior_mean = u0
    u0_prior_var = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    observables_noise = [0.1, 0.1]
    observables = [0.1 0.2 0.3 0.4; 0.2 0.3 0.4 0.5]

    # For Quadratic problem
    A_0 = zeros(N)
    A_1 = A
    A_2 = zeros(N, N, N)
    C_0 = zeros(M)
    C_1 = C
    C_2 = zeros(M, N, N)
    noise = zeros(K, T - 1)

    @compile_workload begin
        # LinearStateSpaceProblem with DirectIteration (most common use case)
        prob_linear = LinearStateSpaceProblem(A, B, u0, tspan)
        sol_linear = solve(prob_linear)

        # LinearStateSpaceProblem with observation equation
        prob_linear_obs = LinearStateSpaceProblem(A, B, u0, tspan; C = C)
        sol_linear_obs = solve(prob_linear_obs)

        # LinearStateSpaceProblem with KalmanFilter
        prob_kf = LinearStateSpaceProblem(A, B, u0, tspan;
            u0_prior_mean = u0_prior_mean,
            u0_prior_var = u0_prior_var,
            C = C,
            observables_noise = observables_noise,
            observables = observables)
        sol_kf = solve(prob_kf, KalmanFilter())

        # QuadraticStateSpaceProblem with DirectIteration
        prob_quad = QuadraticStateSpaceProblem(A_0, A_1, A_2, B, u0, tspan;
            C_0 = C_0, C_1 = C_1, C_2 = C_2, noise = noise)
        sol_quad = solve(prob_quad)
    end
end
