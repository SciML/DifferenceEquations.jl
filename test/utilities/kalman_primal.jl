using ChainRulesTestUtils, DifferenceEquations, Distributions, LinearAlgebra, Test, Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

A_kalman = [0.0495388  0.0109918  0.0960529   0.0767147  0.0404643;
            0.020344   0.0627784  0.00865501  0.0394004  0.0601155;
            0.0260677  0.039467   0.0344606   0.033846   0.00224089;
            0.0917289  0.081082   0.0341586   0.0591207  0.0411927;
            0.0837549  0.0515705  0.0429467   0.0209615  0.014668]
B_kalman = [0.589064  0.97337   2.32677;
            0.864922  0.695811  0.618615;
            2.07924   1.11661   0.721113;
            0.995325  1.8416    2.30442;
            1.76884   1.56082   0.749023]
C_kalman = [0.0979797  0.114992   0.0964536  0.110065   0.0946794;
            0.110095   0.0856981  0.0841296  0.0981172  0.0811817;
            0.109134   0.103406   0.112622   0.0925896  0.112384;
            0.0848231  0.0821602  0.099332   0.113586   0.115105]
D_kalman = MvNormal(Diagonal(abs2.(ones(4) * 0.1)))
u0_kalman = zeros(5)

T = 200
function _solve(observables, A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::MvNormal, u0, tspan)
    # hardcoded right now for tspan = (0, T) for T+1 points
    T = tspan[2]
    @assert tspan[1] == 0
    @assert size(observables)[2] == T # i.e. we do not calculate the likelihood of the initial condition

    # Gaussian Prior
    u0_mean = u0.μ
    u0_variance = u0.Σ
    R = D.Σ
    B_prod = B * B'

    # TODO: when saveall = false, etc. don't allocate everything, or at least don't save it
    u = Zygote.Buffer(Vector{Vector{Float64}}(undef, T+1)) # prior mean
    P = Zygote.Buffer(Vector{Matrix{Float64}}(undef, T+1)) # prior variance
    z = Zygote.Buffer(Vector{Vector{Float64}}(undef, T+1)) # mean observation

    u[1] = u0_mean
    P[1] = u0_variance
    z[1] = C * u[1]
    loglik = 0.0
    for i in 2:T+1
        # Kalman iteration
        u[i] = A * u[i - 1]
        P[i] = A * P[i - 1] * A' + B_prod
        z[i] = C * u[i]

        CP_i = C * P[i]
        V = Symmetric(CP_i * C' + R)
        loglik += logpdf(MvNormal(z[i], V), observables[:, i-1])
        K = CP_i' / V  # gain
        u[i] += K * (observables[:, i-1] - z[i])
        P[i] -= K * CP_i    
    end
    return copy(z),copy(u),copy(P),loglik
end

observables_kalman = readdlm(joinpath(pkgdir(DifferenceEquations),
                                    "test/data/Kalman_observables.csv"), ',')' |> collect
z, u, P, loglik = _solve(observables_kalman, A_kalman, B_kalman, C_kalman, D_kalman, MvNormal(u0_kalman, diagm(ones(length(u0_kalman)))), [0, T])
println(loglik)