module PWMatrix

using SparseArrays, StaticArrays

export PWNoiseMatrix, PWNoiseMatrixZ, add!, set_zero!, square

mutable struct PWNoiseMatrix{MT,T,N}
    vals::Vector{MT}
    pairs::Vector{Tuple{Int64,Int64}}
    npairs::Int
end

PWNoiseMatrix(N::Int,nmax::Int, T=Float64, MT=SVector{3,T}) = PWNoiseMatrix{MT,T,N}(Array{MT}(undef, nmax), Array{Tuple{Int64,Int64}}(undef, nmax),0)

_n(M::PWNoiseMatrix) = M.npairs 
_N(::PWNoiseMatrix{MT,T,N}) where  {MT,T,N} = N
_pairs(M::PWNoiseMatrix) = M.pairs[1:_n(M)]
_vals(M::PWNoiseMatrix) = M.vals[1:_n(M)]

_msort(z1,z2) = (z1<=z2 ? (z1,z2) : (z2,z1))

function square(Σ::PWNoiseMatrix{MT,T,N}, format=:sparse) where {MT,T,N}
    """
    computes M*M^T
    """

    if format == :sparse
        npairs = Σ.npairs
        nvals = 4*npairs
        I, J, V = Array{Int64}(undef,nvals), Array{Int64}(undef,nvals), Array{SMatrix{3, 3, T, 9}}(undef,nvals)
        k = 1
        for ((i,j),σij) in zip(_pairs(Σ),_vals(Σ))
            Γij = σij * σij'
            I[k], J[k], V[k] = i,j,-Γij 
            I[k+1], J[k+1], V[k+1] = j,i,-Γij 
            I[k+2], J[k+2], V[k+2] = i,i, Γij 
            I[k+3], J[k+3], V[k+3] = j,j, Γij 
            k+=4
        end
        A = sparse(I, J, V,N, N)
    else
        A = zeros(SMatrix{3, 3, T, 9},N,N)
        for ((i,j),σij) in zip(_pairs(Σ),_vals(Σ))
            Γij = σij * σij'
            A[i,j] -= Γij 
            A[j,i] -= Γij 
            A[i,i] += Γij 
            A[j,j] += Γij
        end
    end
    return A

end

# function square(Σ::PWNoiseMatrix{MT,T,N}) where {MT,T,N}
#     """
#     computes M*M^T
#     """


#     npairs = Σ.npairs
#     nvals = 4*npairs
#     I, J, V = Array{Int64}(undef,nvals), Array{Int64}(undef,nvals), Array{SMatrix{3, 3, T, 9}}(undef,nvals)
#     k = 1
#     for ((i,j),σij) in zip(_pairs(Σ),_vals(Σ))
#         Γij = σij * σij'
#         I[k], J[k], V[k] = i,j,-Γij 
#         I[k+1], J[k+1], V[k+1] = j,i,-Γij 
#         I[k+2], J[k+2], V[k+2] = i,i, Γij 
#         I[k+3], J[k+3], V[k+3] = j,j, Γij 
#         k+=4
#     end
#     A = sparse(I, J, V,N, N)
#     return A

# end


function set_zero!(Σ::PWNoiseMatrix{MT,T}) where {MT,T<:Real} 
    Σ.npairs = 0 
end


function add!(Σ::PWNoiseMatrix{MT,T}, σ::MT, i::Int,j::Int) where {MT,T<:Real} 
    if !(Σ.npairs < length(Σ.vals))
        nnew = max(length(Σ.pairs),1)
        append!(Σ.pairs,Array{eltype(Σ.pairs)}(undef,nnew))
        append!(Σ.vals, Array{eltype(Σ.vals)}(undef, nnew))
    end
    Σ.npairs+=1
    Σ.pairs[Σ.npairs] = _msort(i,j)
    Σ.vals[Σ.npairs] = σ
end

Base.eltype(::PWNoiseMatrix{MT,T}) where {MT,T} = MT
Base.size(Σ::PWNoiseMatrix{MT,T,N}) where {MT,T,N} = (N,Σ.npairs)
Base.size(Σ::PWNoiseMatrix,i::Int) = size(Σ)[i] 


using JuLIP: AtomicNumber
# mutable struct PWNoiseMatrixZ{MT,T,N}
#     sites::Dict{Tuple{AtomicNumber,AtomicNumber},PWNoiseMatrix{MT,T,N}}
# end

const PWNoiseMatrixZ{MT,T,N} = Dict{Tuple{AtomicNumber,AtomicNumber},PWNoiseMatrix{MT,T,N}}
# import Base

# Base.getindex(M::PWNoiseMatrixZ, zz) = M.sites[zz]

_N(::PWNoiseMatrixZ{MT,T,N}) where {MT,T,N} = N
_n(M::PWNoiseMatrixZ) = sum(m.npairs for m in values(M)) 
# _n(M::PWNoiseMatrixZ, zz::Tuple{AtomicNumber,AtomicNumber}) = M[zz].npairs
_pairs(M::PWNoiseMatrixZ) = Iterators.flatten(Tuple(m.pairs[1:_n(m)] for m in values(M)))
#Iterators.cat(m.pairs[1:_n(m)] for m in values(M))
_vals(M::PWNoiseMatrixZ) = Iterators.flatten(Tuple(m.vals[1:_n(m)] for m in values(M)))

# function square(Σ::PWNoiseMatrixZ{MT,T}, zz::Tuple{AtomicNumber,AtomicNumber}, sparse=:sparse) where {MT,T}
#     return square(Σ.sites[zz], sparse)
# end
function square(Σ::PWNoiseMatrixZ{MT,T}, sparse=:sparse) where {MT,T}
    return sum(square(m, sparse) for m in values(Σ))
end

# function add!(Σ::PWNoiseMatrixZ{MT,T}, σ::MT, i::Int,j::Int, zz::Tuple{AtomicNumber,AtomicNumber}) where {MT, T<:Real} 
#     add!(Σ.sites[zz], σ, i, j)
# end

Base.eltype(::PWNoiseMatrixZ{MT,T}) where {MT,T} = MT
Base.size(Σ::PWNoiseMatrixZ{MT,T,N}) where {MT,T,N} = (N,_n(Σ))
Base.size(Σ::PWNoiseMatrixZ,i::Int) = size(Σ)[i] 

# function mrand(M::PWNoiseMatrixZ,zz) 
#     return mrand(M[zz])
# end


end