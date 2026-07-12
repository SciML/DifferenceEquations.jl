# Shared utilities for ForwardDiff AD tests

"""
    promote_array(::Type{T}, x)

Convert array `x` to element type `T`. No-op if already the right type.
"""
promote_array(::Type{T}, x::AbstractArray{T}) where {T} = x
promote_array(::Type{T}, x::AbstractArray) where {T} = T.(x)

"""
    fdm_gradient(f, x; h=1e-7)

Central finite-difference gradient of scalar function `f` at point `x`.
"""
function fdm_gradient(f, x; h = 1.0e-7)
    n = length(x)
    grad = zeros(n)
    xp = copy(x)
    xm = copy(x)
    for i in 1:n
        xp[i] = x[i] + h
        xm[i] = x[i] - h
        grad[i] = (f(xp) - f(xm)) / (2h)
        xp[i] = x[i]
        xm[i] = x[i]
    end
    return grad
end
