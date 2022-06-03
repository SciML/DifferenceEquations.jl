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
prob = LinearStateSpaceProblem(A_kalman, B_kalman, u0_kalman, (0, T); C = C_kalman, syms = [:a, :b])
sol = solve(prob)
writedlm("test/data/Kalman_observables.csv", sol.z[2:201], ",")
writedlm("test/data/Kalman_noise.csv", transpose(sol.W), ",")