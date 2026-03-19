# Utility functions for generic array operations
# These work with both mutable arrays (Vector) and immutable arrays (SVector)

"""
    mul!!(Y, A, B)

Computes `Y = A * B`.
- If `Y` is mutable (e.g., Vector), it mutates `Y` in-place using `mul!`.
- If `Y` is immutable (e.g., SVector), it returns a new result.
"""
@inline function mul!!(Y, A, B)
    if ismutable(Y)
        mul!(Y, A, B)
        return Y
    else
        return A * B
    end
end

"""
    mul!!(Y, A, B, α, β)

Computes `Y = α * A * B + β * Y` (5-argument form).
- If `Y` is mutable, it mutates `Y` in-place using `mul!(Y, A, B, α, β)`.
- If `Y` is immutable, it returns `α * (A * B) + β * Y`.
"""
@inline function mul!!(Y, A, B, α, β)
    if ismutable(Y)
        mul!(Y, A, B, α, β)
        return Y
    else
        return α * (A * B) + β * Y
    end
end

"""
    muladd!!(Y, A, B)

Computes `Y = Y + A * B`.
- If `Y` is mutable (e.g., Vector), it mutates `Y` in-place using `mul!`.
- If `Y` is immutable (e.g., SVector), it returns a new generic result.
- If `A` or `B` is `nothing`, returns `Y` unchanged (no-op).
"""
@inline function muladd!!(Y, A, B)
    if ismutable(Y)
        mul!(Y, A, B, 1.0, 1.0)
        return Y
    else
        return Y + A * B
    end
end

# Specializations for nothing arguments (no-op)
@inline muladd!!(Y, ::Nothing, B) = Y
@inline muladd!!(Y, A, ::Nothing) = Y
@inline muladd!!(Y, ::Nothing, ::Nothing) = Y

"""
    ldiv!!(y, F, x)

Computes `y = F \\ x` (linear solve with factorization F).
- If `y` is mutable, it mutates `y` in-place using `ldiv!(y, F, x)`.
- If `y` is immutable, it returns `F \\ x`.
"""
@inline function ldiv!!(y, F, x)
    if ismutable(y)
        ldiv!(y, F, x)
        return y
    else
        return F \ x
    end
end

"""
    ldiv!!(F, x)

Computes `x = F \\ x` in-place (2-argument form).
- If `x` is mutable, it modifies `x` in-place using `ldiv!(F, x)`.
- If `x` is immutable, it returns `F \\ x`.
"""
@inline function ldiv!!(F, x)
    if ismutable(x)
        ldiv!(F, x)
        return x
    else
        return F \ x
    end
end

"""
    copyto!!(Y, X)

Copies `X` to `Y`.
- If `Y` is mutable, it mutates `Y` in-place using `copyto!(Y, X)`.
- If `Y` is immutable, it returns `X` directly (immutables are values).
"""
@inline function copyto!!(Y, X)
    if ismutable(Y)
        copyto!(Y, X)
        return Y
    else
        return X
    end
end

"""
    assign!!(Y, X)

Copy `X` into `Y` using an explicit loop (Enzyme-safe activity analysis).
- If `Y` is mutable, copies element-by-element with `@inbounds` and returns `Y`.
- If `Y` is immutable (e.g., `SVector`), returns `X` directly.
"""
@inline function assign!!(Y, X)
    if ismutable(Y)
        @inbounds for i in eachindex(X)
            Y[i] = X[i]
        end
        return Y
    else
        return X
    end
end

"""
    cholesky!!(A, uplo::Symbol=:U)

Computes Cholesky factorization of symmetric matrix A.
- If `A` is mutable, uses `cholesky!(Symmetric(A, uplo), NoPivot(); check=false)`.
- If `A` is immutable, uses `cholesky(Symmetric(A, uplo))`.
"""
@inline function cholesky!!(A, uplo::Symbol = :U)
    if ismutable(A)
        return cholesky!(Symmetric(A, uplo), NoPivot(); check = false)
    else
        return cholesky(Symmetric(A, uplo))
    end
end

"""
    transpose!!(Y, X)

Transposes `X` into `Y`.
- If `Y` is mutable, uses `transpose!(Y, X)`.
- If `Y` is immutable, returns `transpose(X)`.
"""
@inline function transpose!!(Y, X)
    if ismutable(Y)
        transpose!(Y, X)
        return Y
    else
        return transpose(X)
    end
end

"""
    logdet_chol(F)

Compute log-determinant from Cholesky factorization without allocations.
Uses: logdet(A) = logdet(U'U) = 2*sum(log(diag(U))) for upper Cholesky.
"""
@inline function logdet_chol(F)
    U = F.U
    result = zero(eltype(U))
    @inbounds for i in axes(U, 1)
        result += log(U[i, i])
    end
    return 2 * result
end

"""
    symmetrize_upper!!(L, A, eps=0.0)

Symmetrize matrix A into upper triangular form with optional diagonal perturbation.
- If `L` is mutable, modifies in-place and returns `L`
- If `L` is immutable, returns `(A + A')/2 + eps*I`
"""
@noinline function symmetrize_upper!!(L, A, eps = 0.0)
    if ismutable(L)
        @inbounds for j in axes(A, 2)
            for i in 1:j
                v = (A[i, j] + A[j, i]) * 0.5
                L[i, j] = (i == j) ? v + eps : v
            end
            for i in (j + 1):size(A, 1)
                L[i, j] = 0
            end
        end
        return L
    else
        sym = (A + A') / 2
        if eps != 0
            return sym + eps * one(A)
        else
            return sym
        end
    end
end

# =============================================================================
# Prototype-based allocation utilities
# =============================================================================

"""
    alloc_like(x)
    alloc_like(x, dims::Int...)

Allocate an array matching the type family of `x`.
"""
@inline alloc_like(x::AbstractArray) = similar(x)
@inline alloc_like(::SVector{N, T}) where {N, T} = zeros(SVector{N, T})
@inline alloc_like(::SMatrix{N, M, T}) where {N, M, T} = zeros(SMatrix{N, M, T})

# Different dimensions
@inline alloc_like(x::AbstractArray, dims::Int...) = similar(x, dims...)
@inline alloc_like(::SVector{<:Any, T}, n::Int) where {T} = zeros(SVector{n, T})
@inline alloc_like(::SMatrix{<:Any, <:Any, T}, n::Int, m::Int) where {T} =
    zeros(SMatrix{n, m, T})
@inline alloc_like(::SMatrix{<:Any, <:Any, T}, n::Int) where {T} = zeros(SVector{n, T})

# =============================================================================
# Generic zeroing utility for Enzyme compatibility
# =============================================================================

"""
    fill_zero!!(x)

Zero out all elements of `x`.
"""
@inline fill_zero!!(::SVector{N, T}) where {N, T} = zeros(SVector{N, T})
@inline fill_zero!!(::SMatrix{N, M, T}) where {N, M, T} = zeros(SMatrix{N, M, T})
@inline function fill_zero!!(x::AbstractArray{T}) where {T}
    fill!(x, zero(T))
    return x
end

# =============================================================================
# Quadratic form utilities using flattened matrix layout
# =============================================================================

"""
    quad_form_flat!!(result, A_flat, x, tmp_flat)

Compute `result[i] = x' * A_flat_i * x` where `A_flat` is a `(n_out*n_x, n_x)` matrix
formed by vertically stacking the n_out individual `(n_x, n_x)` matrices.

Uses a single `mul!` call for the entire batch, then dot-products per output element.
"""
@inline function quad_form_flat!!(result, A_flat, x, tmp_flat)
    if ismutable(result)
        mul!(tmp_flat, A_flat, x)
        n_x = length(x)
        @inbounds for i in eachindex(result)
            s = zero(eltype(x))
            offset = (i - 1) * n_x
            for j in 1:n_x
                s += x[j] * tmp_flat[offset + j]
            end
            result[i] = s
        end
        return result
    else
        return _quad_form_flat_static(A_flat, x)
    end
end

# SMatrix path for StaticArrays
@inline function _quad_form_flat_static(A_flat::SMatrix{NK, NX}, x::SVector{NX}) where {NK, NX}
    tmp_flat = A_flat * x
    n_out = NK ÷ NX
    result = ntuple(n_out) do i
        s = zero(eltype(x))
        offset = (i - 1) * NX
        for j in 1:NX
            s += x[j] * tmp_flat[offset + j]
        end
        s
    end
    return SVector(result)
end

# Fallback for non-SMatrix immutable types
@inline function _quad_form_flat_static(A_flat, x)
    n_x = length(x)
    tmp_flat = A_flat * x
    n_out = length(tmp_flat) ÷ n_x
    result = ntuple(n_out) do i
        s = zero(eltype(x))
        offset = (i - 1) * n_x
        for j in 1:n_x
            s += x[j] * tmp_flat[offset + j]
        end
        s
    end
    return SVector(result)
end

"""
    quad_muladd_flat!!(result, A_flat, x, tmp_flat)

Compute `result .+= quad_form(A_flat, x)` — accumulate quadratic form into result.

For mutable arrays, computes quad form into a temp and adds element-wise.
For immutable arrays, returns `result + quad_form`.
"""
@inline function quad_muladd_flat!!(result, A_flat, x, tmp_flat)
    if ismutable(result)
        mul!(tmp_flat, A_flat, x)
        n_x = length(x)
        @inbounds for i in eachindex(result)
            s = zero(eltype(x))
            offset = (i - 1) * n_x
            for j in 1:n_x
                s += x[j] * tmp_flat[offset + j]
            end
            result[i] += s
        end
        return result
    else
        return result + _quad_form_flat_static(A_flat, x)
    end
end

"""
    flatten_quad_tensor(A_3d)

Convert a 3D tensor `A[i,:,:]` (n_out × n_x × n_x) to a flattened `(n_out*n_x, n_x)` matrix
by vertically stacking slices `A[i,:,:]` for i = 1:n_out.
"""
function flatten_quad_tensor(A_3d::AbstractArray{T, 3}) where {T}
    n_out = size(A_3d, 1)
    n_x = size(A_3d, 2)
    @assert size(A_3d, 3) == n_x
    A_flat = Matrix{T}(undef, n_out * n_x, n_x)
    @inbounds for i in 1:n_out
        offset = (i - 1) * n_x
        for c in 1:n_x
            for r in 1:n_x
                A_flat[offset + r, c] = A_3d[i, r, c]
            end
        end
    end
    return A_flat
end
