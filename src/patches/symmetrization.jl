using ACEbonds
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