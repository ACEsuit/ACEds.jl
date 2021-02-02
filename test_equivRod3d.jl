using JuLIP, ACE, JuLIP.Potentials, LinearAlgebra
using ACE.Testing: lsq, get_V0
using LinearAlgebra: qr, cond
using Plots


using ACE
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d
using JuLIP.Potentials: i2z, numz
using ACE.RPI
using ACEds
using ACEds.equivRotations3D
using ACEds.equivRotations3D: _mrange1, ERot3DCoeffs
#using ACE.SphericalHarmonics: *
using ACE.SphericalHarmonics
using ACE.SphericalHarmonics: PseudoSpherical, index_y, cart2spher, sizeY, sizeP, allocate_p, compute_coefficients, compute_p!, cYlm!


using JuLIP.MLIPs

using ReferenceFrameRotations: angle_to_dcm

using StaticArrays
##

maxdeg = 3
r0 = 1.0
rcut = 3.0

trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)


Nat = 4
P1 = ACE.BasicPSH1pBasis(Pr; species = :X)
Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, :X)
a = evaluate(P1, Rs, Zs, z0)  # A = ∑_j ϕ(r_j)
b= evaluate(P1, Rs[1], Zs[1], z0)
print(size(Rs))
print(size(a))
# evaluates ϕ(r_1)
# evaluate(P1, [Rs[1]], [Zs[1]], z0)

#phi = ACE.RPI.PSH1pBasisFcn(1,1,1,0)

#valuate!(JuLIP.MLIPs.alloc_B(phi, Rs[1]))
#evaluate!(A, tmp, basis::OneParticleBasis, R, z::AtomicNumber, z0)
#evaluate(phi,[0.0,0,0])



"""
evaluate complex spherical harmonics
"""
#ensure norm(Rs1)<=1
Rs1 = copy(Rs)
for i in 1:length(Rs)
	Rs1[i] =Rs[i]# rand(1) * Rs[i]/norm(Rs[i])
end

ll = @SVector [1,2]
N  = length(ll)
L = maximum(ll)
Y = [ Array{ComplexF64}(undef,sizeY(L)) for i in 1:Nat]
#Array{Array{ComplexF64,sizeY(L)},Nat}
#Array{ComplexF64}(undef, sizeY(L))
#S= Array{PseudoSpherical}(undef, Nat)
coeff = compute_coefficients(L)
P =  Array{Float64}(undef, sizeP(L))
for i in 1:Nat
	S = cart2spher(Rs1[i])
	compute_p!(L, S, coeff, P)
	print(P)
	cYlm!(Y[i], L, S, P)
end

Y_rot = [ Array{ComplexF64}(undef,sizeY(L)) for i in 1:Nat]
Q = angle_to_dcm(.5,.4,.3, :ZYX)

for i in 1:Nat
	S = cart2spher(Q*Rs1[i])
	compute_p!(L, S, coeff, P)
	ACE.SphericalHarmonics.cYlm!(Y_rot[i], L, S, P)
end


for (i,val) in enumerate(_mrange1(ll))
   print(i,",", val,"\n")
end

erotc = ACEds.equivRotations3D.ERot3DCoeffs(Float64)



function eB(Rs, k, A::ERot3DCoeffs{T}, ll::SVector, mm::SVector) where {T}
	n = length(Rs)
	N  = length(ll)
	@assert n == N
	L = maximum(ll)

	#precompute spherical harmonics
	Y = [ Array{ComplexF64}(undef,sizeY(L)) for i in 1:n]
	coeff = compute_coefficients(L)
	P =  Array{Float64}(undef, sizeP(L))
	for i in 1:n
		S = cart2spher(Rs[i])
		compute_p!(L, S, coeff, P)
		ACE.SphericalHarmonics.cYlm!(Y[i], L, S, P)
	end
	Y_vec = [ prod([ Y[j][index_y(ll[j],mvals[j])] for j in 1:N ])   for  (i,mvals) in enumerate(_mrange1(ll))]
	for (i,mu) in enunemerate(_mrange1(ll))
		 GG[im,ik] += dot(A(ll, mm, mu)[:,im],A(ll, kk, mu)[:,ik])
	end

	return Y_vec
end
#allocate_p(L)


eB(Rs1, k, A::ERot3DCoeffs{T}, ll::SVector, mm::SVector)

Y[4][index_y(1,1)]

mr2 = ACEds.equivRotations3D._mrange1(ll)
#for (i,val) in enumerate(mr2)


Y = Array{ComplexF64}(undef,sizeY(L))

coeff = compute_coefficients(L)
P =  Array{Float64}(undef, sizeP(L))
S = cart2spher(Rs1[1])
compute_p!(L, S, coeff, P)
ACE.SphericalHarmonics.cYlm!(Y, L, S, P)
