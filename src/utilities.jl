
# old quad and adjoint replaced by inplace accumulation versions.
# function quad(A::AbstractArray{<:Number,3}, x)
#     return map(j -> dot(x, view(A, j, :, :), x), 1:size(A, 1))
# end
# # quadratic form pullback
# function quad_pb(Δres::AbstractVector, A::AbstractArray{<:Number,3}, x::AbstractVector)
#     ΔA = similar(A)
#     Δx = zeros(length(x))
#     tmp = x * x'
#     for i in 1:size(A, 1)
#         ΔA[i, :, :] .= tmp .* Δres[i]
#         Δx += (A[i, :, :] + A[i, :, :]') * x .* Δres[i]
#     end
#     return ΔA, Δx
# end

# y += quad(A, x)
# The quad_muladd! uses on a vector of matrices for A
function quad_muladd!(y, A, x)
    @inbounds for j in 1:size(A, 1)
        @views y[j] += dot(x, A[j], x)
    end
    return y
end

# inplace version with accumulation and using the cache of A[i] + A[i]', etc.
function quad_muladd_pb!(ΔA_vec, Δx, Δres, A_vec_sum, x)
    tmp = x * x'  # could add in a temp here
    @inbounds for (i, A_sum) in enumerate(A_vec_sum)  # @views @inbounds  ADD
        ΔA_vec[i] .+= tmp .* Δres[i]
        Δx .+= A_sum * x .* Δres[i]
    end
    return nothing
end

# Temporary.  Eventually, move to sciml NoiseProcess with better rng support/etc.
get_concrete_noise(prob, noise, B, T) = noise # maybe do a promotion to an AbstractVectorOfVector type
get_concrete_noise(prob, noise::Nothing, B::Nothing, T) = nothing # if no noise covariance, then no noise required
get_concrete_noise(prob, noise, B::Nothing, T) = nothing # if no noise covariance, then no noise required
get_concrete_noise(prob, noise::Nothing, B, T) = randn(eltype(B), size(B, 2), T) # default is unit Gaussian
get_concrete_noise(prob, noise::UnivariateDistribution, B, T) = rand(noise, size(B, 2), T) # iid

# Utility functions to conditionally check size if not-nothing
maybe_check_size(m::AbstractArray, index, val) = (size(m, index) == val)
maybe_check_size(m::Nothing, index, val) = true

maybe_check_size(m1::AbstractArray, index1, m2::AbstractArray, index2) = (size(m1, index1) ==
                                                                          size(m2, index2))
maybe_check_size(m1::Nothing, index1, m2, index2) = true
maybe_check_size(m1, index1, m2::Nothing, index2) = true
maybe_check_size(m1::Nothing, index1, m2::Nothing, index2) = true
