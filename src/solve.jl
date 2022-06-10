
abstract type AbstractDifferenceEquationAlgorithm <: DiffEqBase.DEAlgorithm end
struct DirectIteration <: AbstractDifferenceEquationAlgorithm end
struct KalmanFilter <: AbstractDifferenceEquationAlgorithm end

# The typical algorithm in discrete-time is DirectIteration()
# Unlike continuous time, there aren't many simple variations
default_alg(prob::AbstractStateSpaceProblem) = DirectIteration()

# If a normal prior, normal observational noise, no noise given, and observables provided then can use a kalman filter
default_alg(prob::LinearStateSpaceProblem{uType,uPriorMeanType,uPriorVarType,tType,P,NP,F,AType,BType,CType,RType,ObsType,K}) where {uType,uPriorMeanType,uPriorVarType<:AbstractMatrix,tType,P,NP<:Nothing,F,AType,BType,CType,RType<:Union{AbstractVector,AbstractMatrix},ObsType,K} = KalmanFilter()

# Select default algorithm if not provided
DiffEqBase.solve(prob::AbstractStateSpaceProblem; kwargs...) = DiffEqBase.solve(prob,
                                                                                default_alg(prob);
                                                                                kwargshandle = DiffEqBase.KeywordArgSilent,
                                                                                kwargs...)
DiffEqBase.solve(prob::AbstractStateSpaceProblem, alg::Nothing, args...; kwargs...) = DiffEqBase.solve(prob,
                                                                                                       default_alg(prob),
                                                                                                       args...;
                                                                                                       kwargshandle = DiffEqBase.KeywordArgSilent,
                                                                                                       kwargs...)