
abstract type AbstractDifferenceEquationAlgorithm end
struct DirectIteration <: AbstractDifferenceEquationAlgorithm end
struct KalmanFilter <: AbstractDifferenceEquationAlgorithm end

default_alg(prob::LinearStateSpaceProblem{uType,uPriorType,tType,P,NP,AType,BType,CType,RType,ObsType,K,SymsType}) where {uType,uPriorType<:AbstractVector,tType,P,NP,AType,BType,CType,RType,ObsType,K,SymsType} = DirectIteration()

# If a normal prior, normal observational noise, no noise given, and observables provided then can use a kalman filter
default_alg(prob::LinearStateSpaceProblem{uType,uPriorType,tType,P,NP,AType,BType,CType,RType,ObsType,K,SymsType}) where {uType,uPriorType<:MvNormal,tType,P,NP<:Nothing,AType,BType,CType,RType<:MvNormal,ObsType,K,SymsType} = KalmanFilter()