


module MRandom

using ACEds
using ACEds.MatrixModels
using SparseArrays, StaticArrays

export mrand, mtype

mtype(::Type{SVector{N,T}}) where {N,T} = T
mtype(::Type{SMatrix{N1,N2,T}}) where {N1,N2,T} = SVector{N2,T}

function mrand(::PWCMatrixModel, Σ::SparseMatrixCSC{SMatrix{3, 3, T, 9}, TI}) where {T<: Real, TI<:Int}
    I, J, _ = findnz(Σ)
    Rnz = sparse(I,J,randn(SVector{3,T}, length(J)))

    return sum(Σ.* (Rnz + transpose(Rnz))./sqrt(2), dims=1)
end

function mrand(Σ::Matrix{T}) where {T<:Real}
    return Σ * randn(size(Σ,2))
end

function mrand(Σ::SparseMatrixCSC{SMatrix{3, 3, T, 9}, TI}) where {T<: Real, TI<:Int}
    m = size(Σ,2)
    _, j_vec, _ = findnz(Σ)
    J = unique(j_vec)
    Rnz = randn(SVector{3,T}, length(J)) 
    return Σ*sparsevec(J,Rnz,m)
end

function mrand(Σ::SparseMatrixCSC{SVector{3, T}, TI}) where {T<: Real, TI<:Int}
    m = size(Σ,2)
    _, j_vec, _ = findnz(Σ)
    J = unique(j_vec)
    Rnz = randn(T, length(J)) 
    return Σ*sparsevec(J,Rnz,m)
end

function mrand(Σ::SparseMatrixCSC{T,TI}) where {T<: Real, TI<:Int}
    m = size(Σ,2)
    _, j_vec, _ = findnz(Σ)
    J = unique(j_vec)
    Rnz = randn(T, length(J)) 
    return Σ*sparsevec(J,Rnz,m)
end

function mrand(M_vec::Vector)
    return sum(mrand(M) for M in M_vec)
end

end