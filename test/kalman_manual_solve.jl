using ChainRulesCore, ChainRulesTestUtils, DifferenceEquations, Distributions,
      LinearAlgebra, Test,
      Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

function unvech_5(v)
    LowerTriangular(hcat(v[1:5],
                         [zeros(1);
                          v[6:9]],
                         [zeros(2);
                          v[10:12]],
                         [zeros(3);
                          v[13:14]],
                         [zeros(4);
                          v[15]]))
end

function solve_manual_cov_lik(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix,
                              u0_mean,
                              u0_variance_vech, observables, R, tspan)
    # hardcoded right now for tspan = (0, T) for T+1 points
    T = tspan[2]
    @assert tspan[1] == 0
    @assert size(observables)[2] == T # i.e. we do not calculate the likelihood of the initial condition

    # Gaussian Prior
    # u0 prior taken from params
    u0_variance_cholesky = unvech_5(u0_variance_vech)
    u0_variance = u0_variance_cholesky * u0_variance_cholesky'
    B_prod = B * B'

    u = u0_mean
    P = u0_variance
    z = C * u
    loglik = 0.0
    for i in 2:(T + 1)
        # Kalman iteration
        u = A * u
        P = A * P * A' + B_prod
        z = C * u

        CP_i = C * P
        V_temp = CP_i * C' + R
        V = (V_temp + V_temp') / 2
        loglik += logpdf(MvNormal(z, V), observables[:, i - 1])
        K = CP_i' / V  # gain
        u += K * (observables[:, i - 1] - z)
        P -= K * CP_i
    end
    return loglik
end

A_kalman = [0.0495388 0.0109918 0.0960529 0.0767147 0.0404643;
            0.020344 0.0627784 0.00865501 0.0394004 0.0601155;
            0.0260677 0.039467 0.0344606 0.033846 0.00224089;
            0.0917289 0.081082 0.0341586 0.0591207 0.0411927;
            0.0837549 0.0515705 0.0429467 0.0209615 0.014668]
B_kalman = [0.589064 0.97337 2.32677;
            0.864922 0.695811 0.618615;
            2.07924 1.11661 0.721113;
            0.995325 1.8416 2.30442;
            1.76884 1.56082 0.749023]
C_kalman = [0.0979797 0.114992 0.0964536 0.110065 0.0946794;
            0.110095 0.0856981 0.0841296 0.0981172 0.0811817;
            0.109134 0.103406 0.112622 0.0925896 0.112384;
            0.0848231 0.0821602 0.099332 0.113586 0.115105]
observables_kalman = readdlm(joinpath(pkgdir(DifferenceEquations),
                                      "test/data/Kalman_observables.csv"), ',')' |> collect
R_kalman = [0.01 0.0 0.0 0.0;
            0.0 0.02 0.005 0.01;
            0.0 0.005 0.03 0.0;
            0.0 0.01 0.0 0.04]
u0_mean = [0.46278392661230217, -0.35157252508544934, -0.33952978655645105,
    -0.3486954393399204,
    0.6934920135433433]
u0_var_vech = [1.1193770675024004, -0.1755391543370492, -0.8351442110561855,
    0.6799242624030147,
    -0.7627861222280011, 0.1346800868329039, 0.46537792458084976,
    -0.16223737917345768, 0.1772417632124954, 0.2722945202387173,
    -0.3971349857502508, -0.1474011998331263, 0.18113754883619412,
    0.13433861105247683, 0.029171596025489813]
T = 200
@testset "manual Kalman code" begin
    #loglik = solve_manual_cov_lik(A_kalman, B_kalman, C_kalman, u0_mean, u0_var_vech,
    #                              observables_kalman,
    #                              R_kalman, [0, T])

    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> solve_manual_cov_lik(args..., B_kalman, C_kalman, u0_mean,
                                                 u0_var_vech,
                                                 observables_kalman, R_kalman, [0, T]),
               A_kalman; rrule_f = rrule_via_ad, check_inferred = false)

    grad_values = gradient((args...) -> solve_manual_cov_lik(args..., [0, T]), A_kalman,
                           B_kalman,
                           C_kalman,
                           u0_mean,
                           u0_var_vech, observables_kalman,
                           R_kalman)

    @test grad_values[1]≈
    finite_difference_gradient(A -> solve_manual_cov_lik(A, B_kalman, C_kalman, u0_mean,
                                                         u0_var_vech,
                                                         observables_kalman,
                                                         R_kalman, [0, T]), A_kalman) rtol=1e-7

    @test grad_values[4]≈
    finite_difference_gradient(u0_mean_vec -> solve_manual_cov_lik(A_kalman, B_kalman,
                                                                   C_kalman,
                                                                   u0_mean_vec,
                                                                   u0_var_vech,
                                                                   observables_kalman,
                                                                   R_kalman, [0, T]),
                               u0_mean) rtol=1e-6

    @test grad_values[5]≈
    finite_difference_gradient(u0_var -> solve_manual_cov_lik(A_kalman, B_kalman, C_kalman,
                                                              u0_mean,
                                                              u0_var,
                                                              observables_kalman,
                                                              R_kalman, [0, T]),
                               u0_var_vech) rtol=2e-4
end
