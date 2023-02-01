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

function reinterpret(::Type{Vector{SVector{N, T}}}, c_vec::Vector{SVector{N, T}}) where {N,T}
    return c_vec
end


function reinterpret(::Type{SVector{Vector{T}}}, c_vec::Vector{SVector{N, T}}) where {N,T}#where {N<:Int,T<:Number}
    return SVector{N}([[c[i] for c in c_vec ] for i=1:N])
end

function reinterpret(::Type{SVector{Vector{T}}}, cc::Matrix{T}) where {T}#where {N<:Int,T<:Number}
    return SVector{size(cc,1)}(cc[i,:] for i=1:size(cc,1))
end

function reinterpret(::Type{Vector{SVector{T}}}, c_vec::SVector{N,Vector{T}}) where {N,T}#where {N<:Int,T<:Number}
    m = length(c_vec[1])
    @assert all(length(c_vec[i]) == m for i=1:N)
    return [SVector{N}([c_vec[i][j] for i=1:N]) for j=1:m]
end


function reinterpret(::Type{Vector{T}}, c_vec::Vector{SVector{N, T}}) where {N,T}
    return [c[i] for i=1:N for c in c_vec]
end
function reinterpret(::Type{Vector{SVector{N, T}}}, c_vec::Vector{T}) where {N,T}
    m = Int(length(c_vec)/N)
    return [ SVector{N}([c_vec[j+(i-1)*m] for i=1:N]) for j=1:m ]
end


function reinterpret(::Type{Matrix{T}}, c_vec::Vector{SVector{N, T}}) where {N,T}
    """
    input: c_vec each 
    N_basis = length(c_vec)
    N = numbero of channels
    """
    c_matrix = Array{T}(undef,N,length(c_vec))
    for j in eachindex(c_vec)
        c_matrix[:,j] = c_vec[j]
    end
    return c_matrix
end

function reinterpret(::Type{Vector{SVector{T}}}, c_matrix::Matrix{T}) where {T}
    N, N_basis = size(c_matrix)
    return [SVector{N,T}(c_matrix[:,j]) for j=1:N_basis ] 
end

function reinterpret(::Type{Vector{SVector{N,T}}}, c_matrix::Matrix{T}) where {N,T}
    @assert N == size(c_matrix,1)
    N_basis = size(c_matrix,2)
    return [SVector{N,T}(c_matrix[:,j]) for j=1:N_basis ] 
end


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
