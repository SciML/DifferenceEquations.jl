
function DiffEqBase.__solve(prob::QuadraticStateSpaceProblem{uType, uPriorMeanType,
                                                             uPriorVarType,
                                                             tType, P, NP, F, A0Type,
                                                             A1Type, A2Type, BType, C0Type,
                                                             C1Type,
                                                             C2Type, RType, ObsType, K},
                            alg::DirectIteration, args...;
                            kwargs...) where {uType, uPriorMeanType, uPriorVarType, tType,
                                              P, NP, F,
                                              A0Type, A1Type, A2Type,
                                              BType, C0Type, C1Type, C2Type, RType, ObsType,
                                              K}
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    noise = get_concrete_noise(prob, prob.noise, prob.B, T - 1)  # concrete noise for simulations as required.    
    observables_noise = make_observables_noise(prob.observables_noise)
    # checks on bounds
    @assert size(noise, 1) == size(prob.B, 2)
    @assert size(noise, 2) == T - 1
    @assert maybe_check_size(prob.observables, 2, T - 1)

    @unpack A_0, A_1, A_2, B, C_0, C_1, C_2 = prob

    C_2_vec = [C_2[i, :, :] for i in 1:size(C_2, 1)] # should be the native datastructure
    A_2_vec = [A_2[i, :, :] for i in 1:size(A_2, 1)] # should be the native datastructure

    u_f = [zero(prob.u0) for _ in 1:T]  # TODO: move to internal algorithm cache
    u = [zero(prob.u0) for _ in 1:T] # TODO: move to internal algorithm cache
    z = [zero(C_0) for _ in 1:T] # TODO: move to internal algorithm cache

    u[1] .= prob.u0
    u_f[1] .= prob.u0
    z[1] .= C_0
    mul!(z[1], C_1, prob.u0, 1, 1)
    quad_muladd!(z[1], C_2_vec, prob.u0) #z0 .+= quad(C_2, prob.u0)

    loglik = 0.0
    @inbounds @views for t in 2:T
        mul!(u_f[t], A_1, u_f[t - 1])
        mul!(u_f[t], B, view(noise, :, t - 1), 1, 1)

        u[t] .= A_0
        mul!(u[t], A_1, u[t - 1], 1, 1)
        quad_muladd!(u[t], A_2_vec, u_f[t - 1]) # u[t] .+= quad(A_2, u_f[t - 1])
        mul!(u[t], B, view(noise, :, t - 1), 1, 1)

        z[t] .= C_0
        mul!(z[t], C_1, u[t], 1, 1)
        quad_muladd!(z[t], C_2_vec, u_f[t]) # z[t] .+= quad(C_2, u_f[t])
        loglik += maybe_logpdf(observables_noise, prob.observables, t - 1, z, t)
    end

    maybe_add_observation_noise!(z, observables_noise, prob.observables)
    t_values = prob.tspan[1]:prob.tspan[2]
    return build_solution(prob, alg, t_values, u; W = noise,
                          logpdf = ObsType <: Nothing ? nothing : loglik, z,
                          retcode = :Success)
end

function ChainRulesCore.rrule(::typeof(DiffEqBase.solve), prob::QuadraticStateSpaceProblem,
                              alg::DirectIteration, args...; kwargs...)
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    noise = get_concrete_noise(prob, prob.noise, prob.B, T - 1)  # concrete noise for simulations as required.    
    @assert !isnothing(prob.noise)  # need to have concrete noise for this simple method
    # checks on bounds
    observables_noise = make_observables_noise(prob.observables_noise)
    @assert typeof(observables_noise) <: ZeroMeanDiagNormal  # can extend to more general in rrule

    @assert size(noise, 1) == size(prob.B, 2)
    @assert maybe_check_size(prob.observables, 2, T - 1)
    @assert size(noise, 2) == T - 1

    @unpack A_0, A_1, A_2, B, C_0, C_1, C_2 = prob

    C_2_vec = [C_2[i, :, :] for i in 1:size(C_2, 1)] # should be the native datastructure
    A_2_vec = [A_2[i, :, :] for i in 1:size(A_2, 1)] # should be the native datastructure

    u_f = [zero(prob.u0) for _ in 1:T]  # TODO: move to internal algorithm cache
    u = [zero(prob.u0) for _ in 1:T] # TODO: move to internal algorithm cache
    z = [zero(C_0) for _ in 1:T] # TODO: move to internal algorithm cache

    u[1] .= prob.u0
    u_f[1] .= prob.u0
    z[1] .= C_0
    mul!(z[1], C_1, prob.u0, 1, 1)
    quad_muladd!(z[1], C_2_vec, prob.u0) #z0 .+= quad(C_2, prob.u0)

    loglik = 0.0
    @inbounds @views for t in 2:T
        mul!(u_f[t], A_1, u_f[t - 1])
        mul!(u_f[t], B, view(noise, :, t - 1), 1, 1)

        u[t] .= A_0
        mul!(u[t], A_1, u[t - 1], 1, 1)
        quad_muladd!(u[t], A_2_vec, u_f[t - 1]) # u[t] .+= quad(A_2, u_f[t - 1])
        mul!(u[t], B, view(noise, :, t - 1), 1, 1)

        z[t] .= C_0
        mul!(z[t], C_1, u[t], 1, 1)
        quad_muladd!(z[t], C_2_vec, u_f[t]) # z[t] .+= quad(C_2, u_f[t])
        loglik += logpdf(observables_noise, view(prob.observables, :, t - 1) - z[t])
    end
    t_values = prob.tspan[1]:prob.tspan[2]
    maybe_add_observation_noise!(z, observables_noise, prob.observables)
    sol = build_solution(prob, alg, t_values, u; W = noise, logpdf = loglik, z,
                         retcode = :Success)

    function solve_pb(??sol)
        # Currently only changes in the logpdf are supported in the rrule
        @assert ??sol.u == ZeroTangent()
        @assert ??sol.W == ZeroTangent()

        ??logpdf = ??sol.logpdf
        if iszero(??logpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end
        ??A_0 = zero(A_0)
        ??A_1 = zero(A_1)
        ??A_2_vec = [zero(A) for A in A_2_vec] # should be native datastructure
        ??A_2 = zero(A_2)

        ??B = zero(B)
        ??C_0 = zero(C_0)
        ??C_1 = zero(C_1)
        ??C_2_vec = [zero(A) for A in C_2_vec] # should be native datastructure
        ??C_2 = zero(C_2)
        ??u_f_sum = zero(u[1])

        ??noise = similar(noise)
        ??u = [zero(prob.u0) for _ in 1:T]
        ??u_f = [zero(prob.u0) for _ in 1:T]
        A_2_vec_sum = [(A + A') for A in A_2_vec] # prep the sum since we will use it repeatedly
        C_2_vec_sum = [(A + A') for A in C_2_vec] # prep the sum since we will use it repeatedly

        # Assert checked above about being diagonal
        observables_noise_cov = prob.observables_noise

        @views @inbounds for t in T:-1:2
            ??z = ??logpdf * (view(prob.observables, :, t - 1) - z[t]) ./
                 observables_noise_cov # More generally, it should be ??^-1 * (z_obs - z)

            # inplace adoint of quadratic form with accumulation
            quad_muladd_pb!(??C_2_vec, ??u_f[t], ??z, C_2_vec_sum, u_f[t])
            mul!(??u[t], C_1', ??z, 1, 1)

            quad_muladd_pb!(??A_2_vec, ??u_f[t - 1], ??u[t], A_2_vec_sum, u_f[t - 1])
            mul!(??u[t - 1], A_1', ??u[t])
            mul!(??u_f[t - 1], A_1', ??u_f[t], 1, 1)

            # ??u_f_sum = ??u[t] .+ ??u_f[t]
            copy!(??u_f_sum, ??u[t])
            ??u_f_sum .+= ??u_f[t]

            mul!(view(??noise, :, t - 1), B', ??u_f_sum)
            # Now, deal with the coefficients
            ??A_0 += ??u[t]
            mul!(??A_1, ??u[t], u[t - 1]', 1, 1)
            mul!(??A_1, ??u_f[t], u_f[t - 1]', 1, 1)
            mul!(??B, ??u_f_sum, view(noise, :, t - 1)', 1, 1)
            ??C_0 += ??z
            mul!(??C_1, ??z, u[t]', 1, 1)
        end

        # Remove once the vector of matrices or column-major organized 3-tensor is the native datastructure for C_2/A_2
        @views @inbounds for (i, ??A_2_slice) in enumerate(??A_2_vec)
            ??A_2[i, :, :] .= ??A_2_slice
        end
        @views @inbounds for (i, ??C_2_slice) in enumerate(??C_2_vec)
            ??C_2[i, :, :] .= ??C_2_slice
        end

        return (NoTangent(),
                Tangent{typeof(prob)}(; A_0 = ??A_0, A_1 = ??A_1, A_2 = ??A_2, B = ??B,
                                      C_0 = ??C_0,
                                      C_1 = ??C_1, C_2 = ??C_2, u0 = ??u[1] + ??u_f[1],
                                      noise = ??noise,
                                      observables = NoTangent(), # not implemented
                                      observables_noise = NoTangent()), NoTangent(),
                map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
