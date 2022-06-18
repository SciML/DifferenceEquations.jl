struct StateSpaceSolution{T,N,uType,uType2,DType,tType,randType,P,A,IType,DE,PosteriorType,
                          logpdfType,zType} <: SciMLBase.AbstractRODESolution{T,N,uType}
    u::uType
    u_analytic::uType2
    errors::DType
    t::tType
    W::randType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    destats::DE
    retcode::Symbol
    P::PosteriorType
    logpdf::logpdfType
    z::zType
end

function SciMLBase.build_solution(prob::AbstractStateSpaceProblem, alg, t, u; P = nothing,
                                  logpdf = nothing,
                                  W = nothing, timeseries_errors = length(u) > 2, dense = false,
                                  dense_errors = dense, calculate_error = true,
                                  interp = SciMLBase.ConstantInterpolation(t, u),
                                  retcode = :Default,
                                  destats = nothing, z = nothing, kwargs...)
    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    # TODO: add support for has_analytic in the future
    sol = StateSpaceSolution{T,N,typeof(u),Nothing,Nothing,typeof(t),typeof(W),typeof(prob),
                             typeof(alg),typeof(interp),typeof(destats),typeof(P),typeof(logpdf),
                             typeof(z)}(u, nothing, nothing, t, W, prob, alg, interp, dense, 0,
                                        destats, retcode, P, logpdf, z)
    return sol
end

# Just using ConstantInterpolation for now.  Worth specializing?
# (sol::StateSpaceSolution)(t, ::Type{deriv} = Val{0}; idxs = nothing, continuity = :left) where {deriv} = _interpolate(sol, t, idxs)
# _interpolate(sol::StateSpaceSolution, t::Integer, idxs::Nothing) = sol.u[t]
# _interpolate(sol::StateSpaceSolution, t::Number, idxs::Nothing) = sol.u[Integer(round(t))]
# _interpolate(sol::StateSpaceSolution, t::Integer, idxs) = sol.u[t][idxs]

# For recipes
SciMLBase.getindepsym(sol::StateSpaceSolution) = :t
SciMLBase.getindepsym_defaultt(sol::StateSpaceSolution) = :t