using ChainRulesCore, ChainRulesTestUtils, Distributions, LinearAlgebra, Test,
      Zygote
using DelimitedFiles
using DiffEqBase
using FiniteDiff: finite_difference_gradient

unvech_5(v) = LowerTriangular(hcat(v[1:5],
                                   [zeros(1);
                                    v[6:9]],
                                   [zeros(2);
                                    v[10:12]],
                                   [zeros(3);
                                    v[13:14]],
                                   [zeros(4);
                                    v[15]]))

function solve_manual_cov_lik(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, u0_mean,
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
observables_kalman = [0.31354480285995573 0.4977172855703752 0.8643726330281335 0.3353879837120676 2.7943808975308753 2.4784940786804484 -0.7953618543788595 -1.1542415140442166 -1.0032659133344395 0.15030889172584777 -1.0230954970766097 0.027396066619363096 0.6078360208889371 -0.04889168317929201 1.3285244434702224 1.2797703084242653 -0.6397304851985723 -1.5156413649391616 -2.0995468953315224 1.5116645385455025;
                      0.31431128833362193 0.46802194362163846 0.8258242423779844 0.2817780025864479 2.5196871452871554 2.1902203846214587 -0.7095604753292026 -1.0178404992243366 -0.8899897738926988 0.09399789064143303 -0.9130824792850585 0.038571553013637516 0.6007645007824418 -0.0252494319797941 1.1635483065399235 1.1292181988289307 -0.5601197292911211 -1.3659208843257462 -1.916358342360954 1.3192285044143524;
                      0.3196522539069838 0.4993169676612231 0.8594447714073081 0.37692786444073 2.8830646616609537 2.6100731057069027 -0.8268829685226933 -1.2183177557635536 -1.0461499926732583 0.21386534757351938 -1.0881529325637262 0.0038055601595428715 0.6039173936713405 -0.06256713171134264 1.4104741685640503 1.350982828074402 -0.674635005204274 -1.5530648089424002 -2.136635967489282 1.6093457947500194;
                      0.2814743650649635 0.484986101251371 0.826041252131933 0.3412222178044595 2.7963909209532685 2.4865701876936166 -0.8216415732815219 -1.1630628688727378 -1.0068634173699698 0.17937491109654216 -1.021433636439214 0.02809456201389481 0.5608419184193344 -0.06433774515551927 1.3522980880077078 1.2850575560132205 -0.6626677662030529 -1.5177408689915581 -2.076982043550849 1.5566657502220353]
R_kalman = Diagonal(abs2.(ones(4) * 0.1))
u0_mean = [0.0, 0.0, 0.0, 0.0, 0.0]
u0_var_vech = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
T = 20
@testset "manual Kalman code" begin
    #loglik = solve_manual_cov_lik(A_kalman, B_kalman, C_kalman, u0_mean, u0_var_vech,
    #                              observables_kalman,
    #                              R_kalman, [0, T])

    test_rrule(Zygote.ZygoteRuleConfig(),
               (args...) -> solve_manual_cov_lik(args..., B_kalman, C_kalman, u0_mean, u0_var_vech,
                                                 observables_kalman, R_kalman, [0, T]),
               A_kalman; rrule_f = rrule_via_ad, check_inferred = false)

    grad_values = gradient((args...) -> solve_manual_cov_lik(args..., [0, T]), A_kalman,
                           B_kalman,
                           C_kalman,
                           u0_mean,
                           u0_var_vech, observables_kalman,
                           R_kalman)

    @test grad_values[1] ≈
          finite_difference_gradient(A -> solve_manual_cov_lik(A, B_kalman, C_kalman, u0_mean,
                                                               u0_var_vech,
                                                               observables_kalman,
                                                               R_kalman, [0, T]), A_kalman) rtol = 1e-2
end
