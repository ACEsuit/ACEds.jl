using ACEbonds
using ACEbonds: BondEnvelope, cutoff_env, cutoff_radialbasis, EllipsoidBondEnvelope, cutoff
using ACE: EuclideanMatrix
using ACE: coco_dot, SymmetricBasis
using LinearAlgebra
using SparseArrays

function perm(A; varsym=:be)
    # built inverse of dictionary map
	D = Dict{Any, Int}() 
	for (i, val) in enumerate(A)
	   D[val] = i
	end
	P = spzeros(length(A),length(A))
	for j = 1:length(A)
		sgn = 0 #sig = sum of all m values
		U_temp = copy(A[j]) # U_temp = A[j] with sign of m-th entry flipped
        NP = eltype(U_temp)
        @assert NP == NamedTuple{(varsym, :n, :l, :m), Tuple{Symbol, Int64, Int64, Int64}}
		for (i,k) in enumerate(A[j])
			U_temp[i] = NP((getfield(k,varsym), k.n, k.l, -k.m))
			sgn += k.m
		end
		if !(U_temp in A) # Entries in U_temp might be in wrong order (inconsistent with orders in A), so need too replace by appropriately ordered (i.e., in order of the corresponding entry in A ) entry  
			for UU_temp in A
   				if sort(UU_temp) == sort(U_temp)
	   				U_temp = UU_temp
   				end
			end
		end
		@assert(U_temp in A)
		P[j,D[U_temp]] = (-1)^sgn
	end
	return P
end


Base.adjoint(φ::EuclideanMatrix{T}) where {T <: Number} = EuclideanMatrix{T}(adjoint(φ.val)) 
Base.transpose(φ::EuclideanMatrix{T}) where {T <: Number} = EuclideanMatrix{T}(transpose(φ.val))

notzero(U,a,b) = intersect(U[a,:].nzind, U[b,:].nzind)

function symmetrize(b; rtol = 1e-7, varsym = :be, varsumval = :bond)
    A = ACE.get_spec(b.pibasis)
    #U = dropzeros(adjoint.(b.A2Bmap) * perm(A)) #* sparse(diagm( [(-1)^(sort(A[j])[1].l) for j = 1 : length(A)] )))
    U = dropzeros(adjoint.(b.A2Bmap) * perm(A;varsym=varsym) * Diagonal([(-1)^(sum( a.l for a in A[j] if getfield(a,varsym) == varsumval )) for j in eachindex(A)]))
    U_new = dropzeros((b.A2Bmap + U).*.5)
    # get rid of linear dependence
    G = [ length(notzero(U_new,a,b)) == 0 ? 0 : sum( real(coco_dot(U_new[a,i], U_new[b,i])) for i in notzero(U_new,a,b) ) for a = 1:size(U_new)[1], b = 1:size(U_new)[1] ]
    #@show G
    svdC = svd(G)
    rk = rank(Diagonal(svdC.S), rtol = rtol)
    Ured = Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]'
    U_new = sparse(Ured * U_new)
    dropzeros!(U_new)

    # construct symmetric offsite basis
    return SymmetricBasis(b.pibasis,U_new,b.symgrp,b.real)
end

# function symmetrize_fast(b; rtol = 1e-7, varsym = :be, varsymval = :bond)
#     A = ACE.get_spec(b.pibasis)
#     #U = dropzeros(adjoint.(b.A2Bmap) * perm(A)) #* sparse(diagm( [(-1)^(sort(A[j])[1].l) for j = 1 : length(A)] )))
#     U = dropzeros(adjoint.(b.A2Bmap) * perm(A) * Diagonal([(-1)^(sum( a.l for a in A[j] if getfield(a,varsym) == varysumval )) for j in eachindex(A)]))
#     U_new = dropzeros((b.A2Bmap + U).*.5)
#     # get rid of linear dependence
#     G = [ length(notzero(U_new,a,b)) == 0 ? 0 : sum( real(coco_dot(U_new[a,i], U_new[b,i])) for i in notzero(U_new,a,b) ) for a = 1:size(U_new)[1], b = 1:size(U_new)[1] ]
#     #@show G
#     svdC = svd(G)
#     rk = rank(Diagonal(svdC.S), rtol = rtol)
#     Ured = Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]'
#     U_new = sparse(Ured * U_new)
#     dropzeros!(U_new)

#     # construct symmetric offsite basis
#     return SymmetricBasis(b.pibasis,U_new,b.symgrp,b.real)
# end


# using ACE: AbstractProperty


# #---------------------- Equivariant matrices

# struct EuclideanMatrix{T} <: AbstractProperty 
#     val::SMatrix{3, 3, T, 9}
#     symmetry::Symbol # :symmetric, :antisymmetric, :general
#  end
 
#  function Base.show(io::IO, φ::EuclideanMatrix)
#     # println(io, "3x3 $(typeof(φ)):")
#     println(io, "e[ $(φ.val[1,1]), $(φ.val[1,2]), $(φ.val[1,3]);")
#     println(io, "   $(φ.val[2,1]), $(φ.val[2,2]), $(φ.val[2,3]);")
#     print(io,   "   $(φ.val[3,1]), $(φ.val[3,2]), $(φ.val[3,3]) ]")
#  end
 
#  real(φ::EuclideanMatrix) = EuclideanMatrix(real.(φ.val))
#  complex(φ::EuclideanMatrix) = EuclideanMatrix(complex(φ.val))
#  complex(::Type{EuclideanMatrix{T}}) where {T} = EuclideanMatrix{complex(T)}
 
#  +(x::SMatrix{3}, y::EuclideanMatrix) = EuclideanMatrix(x + y.val) # include symmetries, i.e., :symmetric + :symmetric =   :symmetric, :antisymmetric + :antisymmetric = :antisymmetric, :antisymmetric + :symmetric = :general etc.
#  Base.convert(::Type{SMatrix{3, 3, T, 9}}, φ::EuclideanMatrix) where {T} =  convert(SMatrix{3, 3, T, 9}, φ.val)
 
#  isrealB(::EuclideanMatrix{T}) where {T} = (T == real(T))
#  isrealAA(::EuclideanMatrix) = false
 
 
#  #fltype(::EuclideanMatrix{T}) where {T} = T
 
#  EuclideanMatrix{T}() where {T <: Number} = EuclideanMatrix{T}(zero(SMatrix{3, 3, T, 9}), :general)
#  EuclideanMatrix(T::DataType=Float64) = EuclideanMatrix{T}()
#  EuclideanMatrix(T::DataType, symmetry::Symbol) = EuclideanMatrix{T}(zero(SMatrix{3, 3, T, 9}), symmetry)
#  EuclideanMatrix(val::SMatrix{3, 3, T, 9}) where {T <: Number} = EuclideanMatrix(val, :general) # should depend on symmetry of val
 
#  function filter(φ::EuclideanMatrix, grp::O3, bb::Array)
#     if length(bb) == 0  # no zero-correlations allowed 
#        return false 
#     end
#     if length(bb) == 1 #MS: Not sure if this should be here
#        return true
#     end
#     suml = sum( getl(grp, bi) for bi in bb )
#     if haskey(bb[1], msym(grp))  # depends on context whether m come along?
#        summ = sum( getm(grp, bi) for bi in bb )
#        return iseven(suml) && abs(summ) <= 2
#     end
#     return iseven(suml)
#  end
 
#  rot3Dcoeffs(::EuclideanMatrix,T=Float64) = Rot3DCoeffsEquiv{T,1}(Dict[], ClebschGordan(T))
 
#  write_dict(φ::EuclideanMatrix{T}) where {T} =
#        Dict("__id__" => "ACE_EuclideanMatrix",
#                "valr" => write_dict(real.(Matrix(φ.val))),
#                "vali" => write_dict(imag.(Matrix(φ.val))),
#                  "T" => write_dict(T),
#                  "symmetry" => φ.symmetry)
 
#  function read_dict(::Val{:ACE_EuclideanMatrix}, D::Dict)
#     T = read_dict(D["T"])
#     valr = SMatrix{3, 3, T, 9}(read_dict(D["valr"]))
#     vali = SMatrix{3, 3, T, 9}(read_dict(D["vali"]))
#     symmetry = Symbol(D["symmetry"])
#     return EuclideanMatrix{T}(valr + im * vali, symmetry)
#  end

# # cutoff(model::ACE.LinearACEModel) = cutoff(model.basis)

# # cutoff(basis::ACE.SymmetricBasis) = cutoff(basis.pibasis)

# # cutoff(basis::ACE.PIBasis) = cutoff(basis.basis1p)

# # cutoff(B1p::ACE.Product1pBasis) = minimum(cutoff.(B1p.bases))

# # cutoff(B1p::ACE.OneParticleBasis) = Inf

# # cutoff(Rn::ACE.B1pComponent) = 
# #       haskey(Rn.meta, "rcut") ? Rn.meta["rcut"]::Float64 : Inf


# # using NeighbourLists
# # using JuLIP: AbstractAtoms
# # NeighbourLists.sites(at::AbstractAtoms, rcut::AbstractFloat) =
# #       sites(neighbourlist(at, rcut))