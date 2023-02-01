using SparseArrays
using LinearAlgebra
using StaticArrays
import Base: reinterpret

function  compress_matrix(Γ::AbstractMatrix{T}, friction_indices) where {T}
    return Γ[friction_indices,friction_indices]
end 
function  compress_matrix(Γ::Diagonal, friction_indices)
    return Diagonal(diag(Γ)[friction_indices])
end 


function reinterpret(::Type{Matrix{SMatrix{3,3,T,9}}}, dmat::Matrix{T}) where {T <: Number}
    n, m  = size(dmat)
    @assert  n % 3 == 0 && m % 3 == 0
    N,M = Int(n/3), Int(m/3)
    bmat = zeros(SMatrix{3,3,T,9},N,M)
    for k1=1:N
        for k2 =1:M
            bmat[k1,k2]= SMatrix{3,3,T,9}(dmat[(3*(k1-1)+1):(3*k1), (3*(k2-1)+1):(3*k2)])
        end
    end
    return bmat
end

function reinterpret(::Type{Matrix}, mat::Matrix{SMatrix{3, 3, T, 9}}) where {T<:Number}
    n,m = size(mat)
    smat = fill(0.0, 3*n, 3*m )
    for i=1:n
        for j =1:m
            smat[(3*(i-1)+1):(3*i), (3*(j-1)+1):(3*j)] = mat[i,j]  
        end
    end
    return smat
end

reinterpret(::Type{Matrix{SMatrix{3,3,T,9}}}, mat::Matrix{SMatrix{3, 3, T, 9}}) where {T<:Number} = mat


function reinterpret(::Type{Vector{T}}, c_vec::Vector{SVector{N, T}}) where {N,T}
    return [c[i] for i=1:N for c in c_vec]
end
function reinterpret(::Type{Vector{SVector{N, T}}}, c_vec::Vector{T}) where {N,T}
    m = Int(length(c_vec)/N)
    return [ SVector{N}([c_vec[j+(i-1)*m] for i=1:N]) for j=1:m ]
end

# :native <-> :native
function reinterpret(::Type{Vector{SVector{N_rep, T}}}, c_vec::Vector{SVector{N_rep, T}}) where {N_rep,T<:Number}
    return c_vec
end

# :native <-> :svector

function reinterpret(::Type{SVector{Vector{T}}}, c_vec::Vector{SVector{N_rep, T}}) where {N_rep,T<:Number}
    return SVector{N_rep}([[c[i] for c in c_vec ] for i=1:N_rep])
end

function reinterpret(::Type{Vector{SVector{T}}}, c_vec::SVector{N_rep,Vector{T}}) where {N_rep,T<:Number}
    m = length(c_vec[1])
    @assert all(length(c_vec[i]) == m for i=1:N_rep)
    return [SVector{N_rep}([c_vec[i][j] for i=1:N_rep]) for j=1:m]
end

# :native <-> :matrix
function reinterpret(::Type{Matrix{T}}, c_vec::Vector{SVector{N_rep, T}}, transposed=false) where {N_rep,T<:Number}
    """
    input: c_vec each 
    N_basis = length(c_vec)
    N_rep = numbero of channels
    """
    c_matrix = Array{T}(undef,length(c_vec),N_rep)
    for j in eachindex(c_vec)
        c_matrix[j,:] = c_vec[j]
    end
    return (transposed ? copy(transpose(c_matrix)) : c_matrix)
end

function reinterpret(::Type{Vector{SVector{T}}}, c_matrix::Matrix{T}, transposed=false) where {T<:Number}
    c_matrix = (transposed ? copy(transpose(c_matrix)) : c_matrix)
    N_basis, N_rep = size(c_matrix)
    return [SVector{N_rep,T}(c_matrix[j,:]) for j=1:N_basis ] 
end

function reinterpret(::Type{Vector{SVector{N_rep,T}}}, c_matrix::Matrix{T}, transposed=false) where {N_rep,T<:Number}
    c_matrix = (transposed ? copy(transpose(c_matrix)) : c_matrix)
    @assert N_rep == size(c_matrix,2)
    N_basis = size(c_matrix,1)
    return [SVector{N_rep,T}(c_matrix[j,:]) for j=1:N_basis ] 
end

# :matrix <-> :svector
function reinterpret(::Type{SVector{N_rep,Vector{T}}}, c_matrix::Matrix{T}, transposed=false) where {N_rep,T<:Number}
    c_matrix = (transposed ? copy(transpose(c_matrix)) : c_matrix)
    @assert N_rep == size(c_matrix,2)
    return SVector{N_rep}[c_matrix[:,r] for r=1:N_rep ] 
end

function reinterpret(::Type{Matrix{T}}, c_vec::SVector{N_rep,Vector{T}}, transposed=false) where {N_rep,T<:Number}
    N_basis = length(c_vec[1])
    @assert all(length(c_vec[i]) == N_basis for i=1:N_rep)
    c_matrix = Array{T}(undef,N_basis, N_rep)
    for i=1:N_basis
        for r=1:N_rep
            c_matrix[i,r] = c_vec[r][i]
        end
    end
    return (transposed ? copy(transpose(c_matrix)) : c_matrix)
end


########################


# function reinterpret(::Type{SVector{Vector{T}}}, cc::Matrix{T}) where {T}#where {N<:Int,T<:Number}
#     return SVector{size(cc,1)}(cc[i,:] for i=1:size(cc,1))
# end


function reinterpret(::Type{Matrix}, M::Matrix{SVector{3,T}}) where {T}
    m,n = size(M)
    M_new = zeros(3*m,n)
    for i=1:m
        for j=1:n
            M_new[(1+3*(i-1)):(3*i),j] = M[i,j]
        end
    end 
    return M_new
end



function array2svector(x::Array{T,2}) where {T}
    return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
end
