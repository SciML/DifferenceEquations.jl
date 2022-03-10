
# Note that u0 as a distribution becomes a draw with the internal get_concrete_problem.  The u0_prior ensures it isn't lost in that call
DiffEqBase.__solve(prob::LinearStateSpaceProblem, alg::Nothing, args...; kwargs...) = DiffEqBase.__solve(prob,
                                                                                                         default_alg(prob),
                                                                                                         args...;
                                                                                                         kwargs...)
