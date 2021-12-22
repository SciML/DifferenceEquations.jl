abstract type AbstractNoise end

# Noise without an explicit type def
## Given scaling matrix
noise(s::AbstractMatrix, t) = s * randn(size(s, 1))
## Given diagonal noise vector
noise(s::AbstractVector, t) = s .* randn(length(s))

struct Gaussian{T, Btype} <: AbstractNoise
    size::T
    B::Btype # XXX: Probably should be the covariance matrix and not nothing, save on alloc
end

StandardGaussian(size::T) where T = Gaussian{T, Nothing}(size, nothing)

# TODO: Must include RNG support here
noise(s::Gaussian{<:Integer, <:Nothing}, t) = randn(s.size)
noise(s::Gaussian{<:Tuple, <:Nothing}, t) = randn(s.size...)

# Scaling draws by a matrix
noise(s::Gaussian{<:Integer, AbstractArray}, t) = s.B * randn(s.size)
noise(s::Gaussian{<:Tuple, AbstractArray}, t) = s.B * randn(s.size...)


struct DefinedNoise{T} <: AbstractNoise
    values::T
end

noise(s::DefinedNoise{<:Vector}, t) = s.values[t]

# Likelihood definitions
StatsBase.loglikelihood(x, noise::Gaussian{<:Any, <:Nothing}, t) = loglikelihood(MvNormal(diagm(ones(noise.size))), x)
StatsBase.loglikelihood(x, noise::Gaussian, t) = loglikelihood(MvNormal(noise.B), x)
StatsBase.loglikelihood(x, noise::AbstractMatrix, t) = loglikelihood(MvNormal(noise), x)