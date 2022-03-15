
function DiffEqBase.__solve(prob::QuadraticStateSpaceProblem, alg::DirectIteration, args...;
                            kwargs...)
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    noise = get_concrete_noise(prob, prob.noise, prob.B, T - 1)  # concrete noise for simulations as required.    
    # checks on bounds
    @assert size(noise, 1) == size(prob.B, 2)
    @assert size(noise, 2) == size(prob.observables, 2)
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
        loglik += logpdf(prob.observables_noise, view(prob.observables, :, t - 1) - z[t])
    end

    t_values = prob.tspan[1]:prob.tspan[2]
    return build_solution(prob, alg, t_values, u; W = noise, logpdf = loglik, retcode = :Success)
end

function ChainRulesCore.rrule(::typeof(DiffEqBase.solve), prob::QuadraticStateSpaceProblem,
                              alg::DirectIteration, args...; kwargs...)
    T = convert(Int64, prob.tspan[2] - prob.tspan[1] + 1)
    noise = get_concrete_noise(prob, prob.noise, prob.B, T - 1)  # concrete noise for simulations as required.    
    # checks on bounds
    @assert size(noise, 1) == size(prob.B, 2)
    @assert size(noise, 2) == size(prob.observables, 2)
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
        loglik += logpdf(prob.observables_noise, view(prob.observables, :, t - 1) - z[t])
    end
    t_values = prob.tspan[1]:prob.tspan[2]
    sol = build_solution(prob, alg, t_values, u; W = noise, logpdf = loglik, retcode = :Success)

    function solve_pb(Δsol)
        # Currently only changes in the logpdf are supported in the rrule
        @assert Δsol.u == ZeroTangent()
        @assert Δsol.W == ZeroTangent()

        Δlogpdf = Δsol.logpdf
        if iszero(Δlogpdf)
            return (NoTangent(), Tangent{typeof(prob)}(), NoTangent(),
                    map(_ -> NoTangent(), args)...)
        end
        ΔA_0 = zero(A_0)
        ΔA_1 = zero(A_1)
        ΔA_2_vec = [zero(A) for A in A_2_vec] # should be native datastructure
        ΔA_2 = zero(A_2)

        ΔB = zero(B)
        ΔC_0 = zero(C_0)
        ΔC_1 = zero(C_1)
        ΔC_2_vec = [zero(A) for A in C_2_vec] # should be native datastructure
        ΔC_2 = zero(C_2)
        Δu_f_sum = zero(u[1])

        Δnoise = similar(noise)
        Δu = [zero(prob.u0) for _ in 1:T]
        Δu_f = [zero(prob.u0) for _ in 1:T]
        A_2_vec_sum = [(A + A') for A in A_2_vec] # prep the sum since we will use it repeatedly
        C_2_vec_sum = [(A + A') for A in C_2_vec] # prep the sum since we will use it repeatedly

        @views @inbounds for t in T:-1:2
            Δz = Δlogpdf * (view(prob.observables, :, t - 1) - z[t]) ./
                 diag(prob.observables_noise.Σ) # More generally, it should be Σ^-1 * (z_obs - z)

            # inplace adoint of quadratic form with accumulation
            quad_muladd_pb!(ΔC_2_vec, Δu_f[t], Δz, C_2_vec_sum, u_f[t])
            mul!(Δu[t], C_1', Δz, 1, 1)

            quad_muladd_pb!(ΔA_2_vec, Δu_f[t - 1], Δu[t], A_2_vec_sum, u_f[t - 1])
            mul!(Δu[t - 1], A_1', Δu[t])
            mul!(Δu_f[t - 1], A_1', Δu_f[t], 1, 1)

            # Δu_f_sum = Δu[t] .+ Δu_f[t]
            copy!(Δu_f_sum, Δu[t])
            Δu_f_sum .+= Δu_f[t]

            mul!(view(Δnoise, :, t - 1), B', Δu_f_sum)
            # Now, deal with the coefficients
            ΔA_0 += Δu[t]
            mul!(ΔA_1, Δu[t], u[t - 1]', 1, 1)
            mul!(ΔA_1, Δu_f[t], u_f[t - 1]', 1, 1)
            mul!(ΔB, Δu_f_sum, view(noise, :, t - 1)', 1, 1)
            ΔC_0 += Δz
            mul!(ΔC_1, Δz, u[t]', 1, 1)
        end

        # Remove once the vector of matrices or column-major organized 3-tensor is the native datastructure for C_2/A_2
        @views @inbounds for (i, ΔA_2_slice) in enumerate(ΔA_2_vec)
            ΔA_2[i, :, :] .= ΔA_2_slice
        end
        @views @inbounds for (i, ΔC_2_slice) in enumerate(ΔC_2_vec)
            ΔC_2[i, :, :] .= ΔC_2_slice
        end

        return (NoTangent(),
                Tangent{typeof(prob)}(; A_0 = ΔA_0, A_1 = ΔA_1, A_2 = ΔA_2, B = ΔB, C_0 = ΔC_0,
                                      C_1 = ΔC_1, C_2 = ΔC_2, u0 = Δu[1] + Δu_f[1], noise = Δnoise,
                                      observables = NoTangent(), # not implemented
                                      observables_noise = NoTangent()), NoTangent(),
                map(_ -> NoTangent(), args)...)
    end
    return sol, solve_pb
end
