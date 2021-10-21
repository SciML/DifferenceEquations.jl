abstract type AbstractNoise end

struct StandardGaussian{T} <: AbstractNoise
    size::T
end

# Must include RNG support here
noise(s::StandardGaussian{<:Integer}, t) = randn(s.size)
noise(s::StandardGaussian{<:Tuple}, t) = randn(s.size...)

struct DefinedNoise{T} <: AbstractNoise
    values::T
end

noise(s::DefinedNoise{<:Vector}, t) = s.values[t]
