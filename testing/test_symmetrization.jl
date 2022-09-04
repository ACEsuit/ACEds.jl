using ACE, JuLIP
using ACEbonds: BondEnvelope, cutoff_env, cutoff_radialbasis, EllipsoidBondEnvelope, cutoff
n_bulk = 2
r0cut = 2.0*rnn(:Al)
rcut = 2.0 * rnn(:Al)
zcut = 2.0 * rnn(:Al) 


env = EllipsoidBondEnvelope(r0cut, rcut, zcut; p0=1, pr=1, floppy=false, λ= 0.0)
maxorder = 2
Bsel = ACE.SparseBasis(;maxorder=maxorder, p = 2, default_maxdeg = 4) 
RnYlm = ACE.Utils.RnYlm_1pbasis(;   r0 = 0.01, 
                                           rin = 0.01,
                                           pcut = 1,
                                           pin = 1, Bsel = Bsel,
                                           rcut = maximum([cutoff_env(env),rcut])
                                       )


# onsite_posdef = ACE.SymmetricBasis(EuclideanVector(Float64), RnYlm, Bsel;);
# onsite_em = ACE.SymmetricBasis(EuclideanMatrix(Float64), RnYlm, Bsel;);
b = ACEds.Utils.SymmetricBond_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm, bondsymmetry="Invariant");

A = ACE.get_spec(b.pibasis)

sort(A[100])

fieldnames(typeof(b))
fieldnames(typeof(b.pibasis))
fieldnames(typeof(b.A2Bmap))
fieldnames(typeof(offsite.A2Bmap.rowval))
offsite.A2Bmap.nzval
typeof(offsite.A2Bmap)
size(offsite.A2Bmap)
typeof(basis.pibasis.B_pool)
basis.pibasis.real


#%%
using SparseArrays, LinearAlgebra
function perm(A)
    # built inverse of dictionary map
	D = Dict{Any, Int}() 
	for (i, val) in enumerate(A)
	   D[val] = i
	end
	P = spzeros(length(A),length(A))
	for j = 1:length(A)
		sgn = 0 #sig = sum of all m values
		U_temp = copy(A[j]) # U_temp = A[j] with sign of m-th entry flipped
		for (i,k) in enumerate(A[j])
			U_temp[i] = (be = k.be, n = k.n, l = k.l, m = -k.m)
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


Base.adjoint(φ::EuclideanMatrix{T}) where {T <: Number} = EuclideanMatrix{T}(φ.val) 

notzero(U,a,b) = intersect(U[a,:].nzind, U[b,:].nzind)
using ACE: coco_dot, SymmetricBasis
using LinearAlgebra
function symmetrize!(b)
    A = ACE.get_spec(b.pibasis)
    U = dropzeros(adjoint.(b.A2Bmap) * perm(A) * sparse(diagm( [(-1)^(sort(A[j])[1].l) for j = 1 : length(A)] )))
    U_new = dropzeros((b.A2Bmap + U).*.5)
    # get rid of linear dependence
    G = [ length(notzero(U_new,a,b)) == 0 ? 0 : sum( real(coco_dot(U_new[a,i], U_new[b,i])) for i in notzero(U_new,a,b) ) for a = 1:size(U_new)[1], b = 1:size(U_new)[1] ]
    @show G
    svdC = svd(G)
    rk = rank(Diagonal(svdC.S), rtol = 1e-7)
    Ured = Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]'
    U_new = sparse(Ured * U_new)
    dropzeros!(U_new)

    # construct symmetric offsite basis
    return SymmetricBasis(b.pibasis,U_new,b.symgrp,b.real)
end


basis = ACEds.Utils.SymmetricBond_basis(EuclideanMatrix(Float64), env, Bsel; RnYlm = RnYlm);
sbasis = symmetrize!(b)
