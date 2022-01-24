using DifferenceEquations, BenchmarkTools, LoopVectorization

# quadratic form sizes
const A_2_rbc_benchmark = cat([-0.00019761505863889124 0.03375055315837927; 0.0 0.0], [0.03375055315837913 3.128758481817603; 0.0 0.0]; dims = 3)
A_2_raw_benchmark = Matrix(DataFrame(CSV.File(joinpath(path, "$(file_prefix)_A_2.csv"); header = false)))
const A_2_fvgq_benchmark = reshape(A_2_raw, 14, 14, 14)
const x_rbc_benchmark = rand(2)
const x_fvgq_benchmark = rand(14)

# naive quadratic forms

function quad(A::AbstractArray{<:Number,3}, x)
    return map(j -> dot(x, view(A, j, :, :), x), 1:size(A, 1))
end

function quad_pb(Δres::AbstractVector, A::AbstractArray{<:Number,3}, x::AbstractVector)
    ΔA = similar(A)
    Δx = zeros(length(x))
    tmp = x * x'
    for i in 1:size(A, 1)
        ΔA[i, :, :] .= tmp .* Δres[i]
        Δx += (A[i, :, :] + A[i, :, :]') * x .* Δres[i]
    end
    return ΔA, Δx
end


const UTILITIES = BenchmarkGroup()

# RBC sized is 2x2x2 for the A matrix.
