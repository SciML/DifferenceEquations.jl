
# promote to work with a standardized interface
struct MatrixVectorOfArray{T,N,A} <: RecursiveArrayTools.AbstractVectorOfArray{T,N,A}
    u::A # A <: AbstractMatrix{T} where size(u, 1) == N
end

Base.Array(VA::MatrixVectorOfArray{T,N,A}) where {T,N,A} = VA.u

MatrixVectorOfArray(mat::AbstractMatrix) = MatrixVectorOfArray{Float64,2,typeof(mat)}(mat)

# Interface for the linear indexing. This is just a view of the underlying nested structure
@inline Base.firstindex(VA::MatrixVectorOfArray) = firstindex(VA.u, 2)
@inline Base.lastindex(VA::MatrixVectorOfArray) = lastindex(VA.u, 2)

@inline Base.length(VA::MatrixVectorOfArray) = size(VA.u, 2)
@inline Base.eachindex(VA::MatrixVectorOfArray) = Base.OneTo(size(VA.u, 2))
@inline Base.IteratorSize(VA::MatrixVectorOfArray) = Base.HasLength()
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
Base.@propagate_inbounds function Base.getindex(VA::MatrixVectorOfArray{T,N}, I::Int) where {T,N}
    return view(VA.u, :, I)
end
Base.@propagate_inbounds function Base.getindex(VA::MatrixVectorOfArray{T,N}, I::Colon) where {T,N}
    return view(VA.u, :, I)
end
Base.@propagate_inbounds function Base.getindex(VA::MatrixVectorOfArray{T,N}, i::Int,
                                                ::Colon) where {T,N}
    return [VA.u[i, j] for j in 1:length(VA)]
end
Base.@propagate_inbounds function Base.getindex(VA::MatrixVectorOfArray{T,N},
                                                ii::CartesianIndex) where {T,N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[jj, i]
end
Base.@propagate_inbounds function Base.setindex!(VA::MatrixVectorOfArray{T,N}, v,
                                                 I::Int) where {T,N}
    return VA.u[:, I] = v
end
Base.@propagate_inbounds function Base.setindex!(VA::MatrixVectorOfArray{T,N}, v,
                                                 I::Colon) where {T,N}
    return VA.u[:, I] = v
end
Base.@propagate_inbounds function Base.setindex!(VA::MatrixVectorOfArray{T,N}, v,
                                                 I::AbstractArray{Int}) where {T,N}
    return VA.u[:, I] = v
end
Base.@propagate_inbounds function Base.setindex!(VA::MatrixVectorOfArray{T,N}, v, i::Int,
                                                 ::Colon) where {T,N}
    for j in 1:length(VA)
        VA.u[i, j] = v[j]
    end
    return v
end
Base.@propagate_inbounds function Base.setindex!(VA::MatrixVectorOfArray{T,N}, x,
                                                 ii::CartesianIndex) where {T,N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[jj, i] = x
end

# Interface for the two-dimensional indexing, a more standard AbstractArray interface
@inline Base.size(VA::MatrixVectorOfArray) = size(VA.u)
Base.@propagate_inbounds function Base.getindex(VA::MatrixVectorOfArray{T,N}, I::Int...) where {T,N}
    return VA.u[I...]
end
Base.@propagate_inbounds function Base.setindex!(VA::MatrixVectorOfArray{T,N}, v,
                                                 I::Int...) where {T,N}
    return VA.u[I...] = v
end

# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
# using view just as eachcol(VA.u) would on the undrelying matrix
function Base.iterate(VA::MatrixVectorOfArray, state = 1)
    return state >= size(VA.u, 2) + 1 ? nothing : (view(VA, :, state), state + 1)
end

# function Base.append!(VA::AbstractVectorOfArray{T,N}, new_item::AbstractVectorOfArray{T,N}) where {T,N}
#     for item in copy(new_item)
#         push!(VA, item)
#     end
#     return VA
# end

# Tools for creating similar objects
@inline function Base.similar(VA::MatrixVectorOfArray, ::Type{T} = eltype(VA)) where {T}
    return MatrixVectorOfArray(similar(VA.u))
end

# fill!
# For DiffEqArray it ignores ts and fills only u
Base.fill!(VA::MatrixVectorOfArray, x) = fill!(VA.u, x)

# make it show just like its data
function Base.show(io::IO, m::MIME"text/plain", x::MatrixVectorOfArray)
    return (println(io, summary(x), ':'); show(io, m, x.u))
end
Base.summary(A::MatrixVectorOfArray) = string("MatrixVectorOfArray{", eltype(A), ",", ndims(A), "}")

Base.map(f, A::MatrixVectorOfArray) = map(f, eachcol(A.u))
function Base.mapreduce(f, op, A::MatrixVectorOfArray)
    return mapreduce(f, op, (mapreduce(f, op, x) for x in eachcol(A.u)))
end