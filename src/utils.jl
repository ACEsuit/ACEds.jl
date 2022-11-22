module Utils

using StaticArrays
using ProgressMeter
using SparseArrays
#using ACEFit





include("./bondutils2.jl")
include("./butils.jl")


using LinearAlgebra
using SparseArrays
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

function array2svector(x::Array{T,2}) where {T}
    return [ SVector{3}(x[i,:]) for i in 1:size(x)[1] ]
end


# using LinearAlgebra

# row_info(data)

# linear_fill!(A, Y, W, dat, basis; row_start=1)

#n = 4
#a = [i+j for i in 1:n, j =1:n]
#B = [@SVector [i+j,2*i+j,3*i+j] for i in 1:n, j =1:n]
#b = [2*i+j for i in 1:n, j =1:n]
#a
#B
#C = a*B
#sum(a.*b)
#dot(a,b)
end