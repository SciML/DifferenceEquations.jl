
# Note that u0 as a distribution becomes a draw with the internal get_concrete_problem.  The u0_prior ensures it isn't lost in that call
DiffEqBase.__solve(prob::LinearStateSpaceProblem, alg::Nothing, args...; kwargs...) = DiffEqBase.__solve(prob,
                                                                                                         default_alg(prob),
                                                                                                         args...;
                                                                                                         kwargs...)

#                                                                                                          # Apply default algorithm to adjoints as
# DiffEqBase._concrete_solve_adjoint(prob::LinearStateSpaceProblem, alg::Nothing, args...; kwargs...) = DiffEqBase._concrete_solve_adjoint(prob,
#                                                                                                                                          default_alg(prob),
#                                                                                                                                          args...;
#                                                                                                                                          kwargs...)
DiffEqBase.solve(prob::LinearStateSpaceProblem; kwargs...) = DiffEqBase.solve(prob,
                                                                              default_alg(prob);
                                                                              kwargs...)
