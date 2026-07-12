# Utility functions for generic array operations
# These work with both mutable arrays (Vector) and immutable arrays (SVector)

"""
    mul!!(Y, A, B)

Computes `Y = A * B`.
- If `Y` is mutable (e.g., Vector), it mutates `Y` in-place.
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

@inline function mul!!(Y::AbstractVector, A::AbstractMatrix, B::AbstractVector)
    if ismutable(Y)
        matvec_no_blas!(Y, A, B)
        return Y
    else
        return A * B
    end
end

"""
    mul!!(Y, A, B, α, β)

Computes `Y = α * A * B + β * Y` (5-argument form).
- If `Y` is mutable, it mutates `Y` in-place.
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

@inline function mul!!(Y::AbstractVector, A::AbstractMatrix, B::AbstractVector, α, β)
    if ismutable(Y)
        matvec_no_blas!(Y, A, B, α, β)
        return Y
    else
        return α * (A * B) + β * Y
    end
end

"""
    muladd!!(Y, A, B)

Computes `Y = Y + A * B`.
- If `Y` is mutable (e.g., Vector), it mutates `Y` in-place.
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

@inline function muladd!!(Y::AbstractVector, A::AbstractMatrix, B::AbstractVector)
    if ismutable(Y)
        T = promote_type(eltype(Y), eltype(A), eltype(B))
        matvec_no_blas!(Y, A, B, one(T), one(T))
        return Y
    else
        return Y + A * B
    end
end

# Specializations for nothing arguments (no-op)
@inline muladd!!(Y, ::Nothing, B) = Y
@inline muladd!!(Y, A, ::Nothing) = Y
@inline muladd!!(Y, ::Nothing, ::Nothing) = Y

@inline function matvec_no_blas!(Y, A, B)
    @boundscheck begin
        size(Y, 1) == size(A, 1) && size(A, 2) == size(B, 1) ||
            throw(DimensionMismatch("matrix-vector dimensions must match"))
    end
    @inbounds for i in axes(A, 1)
        acc = zero(promote_type(eltype(A), eltype(B)))
        for j in axes(A, 2)
            acc += A[i, j] * B[j]
        end
        Y[i] = acc
    end
    return Y
end

@inline function matvec_no_blas!(Y, A, B, α, β)
    @boundscheck begin
        size(Y, 1) == size(A, 1) && size(A, 2) == size(B, 1) ||
            throw(DimensionMismatch("matrix-vector dimensions must match"))
    end
    @inbounds for i in axes(A, 1)
        acc = zero(promote_type(eltype(A), eltype(B)))
        for j in axes(A, 2)
            acc += A[i, j] * B[j]
        end
        Y[i] = α * acc + β * Y[i]
    end
    return Y
end

@inline function dot_no_blas(x::AbstractVector, y::AbstractVector)
    @boundscheck begin
        size(x, 1) == size(y, 1) || throw(DimensionMismatch("vector dimensions must match"))
    end
    acc = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for i in eachindex(x, y)
        acc += conj(x[i]) * y[i]
    end
    return acc
end

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

@inline function ldiv!!(y::AbstractVector, F::LinearAlgebra.Cholesky, x::AbstractVector)
    if ismutable(y)
        cholesky_solve_no_blas!(y, F, x)
        return y
    else
        return F \ x
    end
end

@inline function ldiv!!(Y::AbstractMatrix, F::LinearAlgebra.Cholesky, X::AbstractMatrix)
    if ismutable(Y)
        cholesky_solve_no_blas!(Y, F, X)
        return Y
    else
        return F \ X
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

@inline function ldiv!!(F::LinearAlgebra.Cholesky, x::AbstractVector)
    if ismutable(x)
        cholesky_solve_no_blas!(x, F, x)
        return x
    else
        return F \ x
    end
end

@inline function ldiv!!(F::LinearAlgebra.Cholesky, X::AbstractMatrix)
    if ismutable(X)
        cholesky_solve_no_blas!(X, F, X)
        return X
    else
        return F \ X
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
- If `A` is mutable, factors `A` in-place and returns a `Cholesky` factorization.
- If `A` is immutable, uses `cholesky(Symmetric(A, uplo))`.
"""
@inline function cholesky!!(A, uplo::Symbol = :U)
    if ismutable(A)
        cholesky_no_lapack!(A, uplo)
        return LinearAlgebra.Cholesky(A, uplo, 0)
    else
        return cholesky(Symmetric(A, uplo))
    end
end

@inline function cholesky_no_lapack!(A, uplo::Symbol)
    @boundscheck begin
        size(A, 1) == size(A, 2) || throw(DimensionMismatch("matrix must be square"))
    end
    if uplo === :U
        return cholesky_upper_no_lapack!(A)
    elseif uplo === :L
        return cholesky_lower_no_lapack!(A)
    else
        throw(ArgumentError("uplo must be :U or :L"))
    end
end

@inline function cholesky_upper_no_lapack!(A)
    n = size(A, 1)
    @inbounds for j in 1:n
        for k in 1:(j - 1)
            acc = A[k, j]
            for i in 1:(k - 1)
                acc -= A[i, k] * A[i, j]
            end
            A[k, j] = acc / A[k, k]
        end
        diag = A[j, j]
        for i in 1:(j - 1)
            diag -= A[i, j] * A[i, j]
        end
        real(diag) > 0 || throw(LinearAlgebra.PosDefException(j))
        A[j, j] = sqrt(diag)
    end
    @inbounds for j in 1:n
        for i in (j + 1):n
            A[i, j] = zero(eltype(A))
        end
    end
    return A
end

@inline function cholesky_lower_no_lapack!(A)
    n = size(A, 1)
    @inbounds for j in 1:n
        for k in 1:(j - 1)
            acc = A[j, k]
            for i in 1:(k - 1)
                acc -= A[k, i] * A[j, i]
            end
            A[j, k] = acc / A[k, k]
        end
        diag = A[j, j]
        for i in 1:(j - 1)
            diag -= A[j, i] * A[j, i]
        end
        real(diag) > 0 || throw(LinearAlgebra.PosDefException(j))
        A[j, j] = sqrt(diag)
    end
    @inbounds for j in 1:n
        for i in 1:(j - 1)
            A[i, j] = zero(eltype(A))
        end
    end
    return A
end

@inline function cholesky_solve_no_blas!(y::AbstractVector, F::LinearAlgebra.Cholesky, x::AbstractVector)
    U = F.U
    n = size(U, 1)
    @boundscheck begin
        size(y, 1) == size(x, 1) || throw(DimensionMismatch("right-hand side dimensions must match"))
        size(x, 1) == n ||
            throw(DimensionMismatch("factor and right-hand side dimensions must match"))
    end
    @inbounds for i in 1:n
        y[i] = x[i]
    end
    cholesky_solve_vector_no_blas!(y, U)
    return y
end

@inline function cholesky_solve_no_blas!(Y::AbstractMatrix, F::LinearAlgebra.Cholesky, X::AbstractMatrix)
    U = F.U
    n = size(U, 1)
    @boundscheck begin
        size(Y) == size(X) || throw(DimensionMismatch("right-hand side dimensions must match"))
        size(X, 1) == n ||
            throw(DimensionMismatch("factor and right-hand side dimensions must match"))
    end
    @inbounds for j in axes(X, 2)
        for i in 1:n
            Y[i, j] = X[i, j]
        end
        cholesky_solve_vector_no_blas!(view(Y, :, j), U)
    end
    return Y
end

@inline function cholesky_solve_vector_no_blas!(y, U)
    n = size(U, 1)
    @inbounds for i in 1:n
        acc = y[i]
        for k in 1:(i - 1)
            acc -= U[k, i] * y[k]
        end
        y[i] = acc / U[i, i]
    end
    @inbounds for i in n:-1:1
        acc = y[i]
        for k in (i + 1):n
            acc -= U[i, k] * y[k]
        end
        y[i] = acc / U[i, i]
    end
    return y
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
    mul_aat!!(Y, A, A_t)

Computes `Y = A * A'` without triggering the BLAS `syrk` self-transpose path.
Workaround for Enzyme syrk adjoint bug (https://github.com/EnzymeAD/Enzyme.jl/issues/2355):
when `A` is rectangular, `mul!(Y, A, transpose(A))` dispatches to `syrk` whose Enzyme
reverse-mode rule generates a `DSYMM` call with invalid leading dimension.

- If `Y` is mutable, materializes `transpose(A)` into buffer `A_t`, then calls `mul!(Y, A, A_t)`.
- If `Y` is immutable, returns `A * transpose(A)` (StaticArrays don't use BLAS).
"""
@inline function mul_aat!!(Y, A, A_t)
    if ismutable(Y)
        transpose!(A_t, A)
        mul!(Y, A, A_t)
        return Y
    else
        return A * transpose(A)
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
@inline function symmetrize_upper!!(L, A, eps = 0.0)
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
