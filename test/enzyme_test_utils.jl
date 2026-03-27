# Shared utilities for Enzyme AD tests

using LinearAlgebra: LowerTriangular, Symmetric, cholesky

"""
    vech_length(n)

Number of elements in the lower triangle of an n×n matrix.
"""
vech_length(n) = n * (n + 1) ÷ 2

"""
    vech(L::AbstractMatrix)

Extract lower-triangular elements of L into a vector (column-major order).
"""
function vech(L::AbstractMatrix)
    n = size(L, 1)
    v = zeros(eltype(L), vech_length(n))
    k = 1
    for j in 1:n
        for i in j:n
            v[k] = L[i, j]
            k += 1
        end
    end
    return v
end

"""
    unvech(v, n)

Reconstruct an n×n `LowerTriangular` matrix from a vech vector.
"""
function unvech(v, n)
    L = zeros(eltype(v), n, n)
    k = 1
    for j in 1:n
        for i in j:n
            L[i, j] = v[k]
            k += 1
        end
    end
    return LowerTriangular(L)
end

"""
    make_posdef_from_vech(v, n)

Construct a guaranteed positive-definite matrix from a vech vector.
Computes L = unvech(v, n), then returns L * L' as a plain Matrix
(not Symmetric, to avoid type instability with Enzyme AD).
"""
function make_posdef_from_vech(v, n)
    L = unvech(v, n)
    # Use Matrix(L) * Matrix(L') to avoid LowerTriangular BLAS dispatch
    # which Enzyme cannot differentiate (trmm! has no derivative rule).
    L_mat = Matrix(L)
    return L_mat * L_mat'
end

"""
    make_vech_for(M::AbstractMatrix)

Given a positive-definite matrix M, compute its Cholesky L factor and return vech(L).
Round-trips: make_posdef_from_vech(make_vech_for(M), n) ≈ Symmetric(M).
"""
function make_vech_for(M::AbstractMatrix)
    F = cholesky(Symmetric(M))
    return vech(F.L)
end

"""
    fdm_gradient(f, x; h=1e-7)

Central finite-difference gradient of scalar function `f` at vector `x`.
"""
function fdm_gradient(f, x; h = 1e-7)
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
