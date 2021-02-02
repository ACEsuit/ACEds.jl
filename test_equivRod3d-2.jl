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

using Random

## Libraries added by Matthias

using Combinatorics
using StatsBase
##

function rand_unit_vec(d::Int64)
	u = [rand() for i in 1:d ]
	u/=norm(u)
	return SVector{d}(u)
end




N = 3 # N-body interaction
r_vec = [rand_unit_vec(3) for i in 1:N]



"""
Specify parameters
"""
ll = @SVector [2,1,3]
@assert N  == length(ll)
L = maximum(ll)

coeff = compute_coefficients(L) # Precompute coefficients for evaluation of spherical Harmonics
"""
Precompute/evaluate complex spherical harmonics
"""
Y = [ Array{ComplexF64}(undef,sizeY(L)) for i in 1:N]
let P =  Array{Float64}(undef, sizeP(L)) # used only temporarily
	for i in 1:N
		S = cart2spher(r_vec[i])
		compute_p!(L, S, coeff, P)
		cYlm!(Y[i], L, S, P)
	end
end

"""Define rotation"""
Q = angle_to_dcm(.5,.4,-.3, :ZYX)

"""Precompute/evaluate rotated complex spherical harmonics"""
Y_rot = [ Array{ComplexF64}(undef,sizeY(L)) for i in 1:N]
let P =  Array{Float64}(undef, sizeP(L))
	for i in 1:N
		S = cart2spher(Q*r_vec[i])
		compute_p!(L, S, coeff, P)
		ACE.SphericalHarmonics.cYlm!(Y_rot[i], L, S, P)
	end
end


"""Precompute coefficients for rotational equivariant basis"""
erotc = ERot3DCoeffs(Float64)

function reB(Y, k::Int, ll::SVector, mm::SVector, A::ERot3DCoeffs{T}) where {T}
	"""
	compute symmetrized precomputed values Y of spherical harmonics
	"""
	N  = length(Y)
	@assert length(Y) == length(ll)
	L = maximum(ll)

	#Y_vec = [prod([ Y[j][index_y(ll[j],mu[j])] for j in 1:N ])
	#	for  (i,mu) in enumerate(_mrange1(ll))]


	eB_val = sum(
		[A(ll,mu,mm)[:,k] * prod([ Y[j][index_y(ll[j],mu[j])] for j in 1:N ])
		for  (i,mu) in enumerate(_mrange1(ll))]
	)

	return eB_val
end

function rpiB(Y, k::Int, ll::SVector, mm::SVector, A::ERot3DCoeffs{T}) where {T}
	"""
	compute symmetrized precomputed values Y of spherical harmonics
	"""
	n  = length(Y)
	@assert length(Y) == length(ll)
	L = maximum(ll)

	#Y_vec = [prod([ Y[j][index_y(ll[j],mu[j])] for j in 1:N ])
	#	for  (i,mu) in enumerate(_mrange1(ll))]
	erpiB_val = zeros(ComplexF64, 3)
	@assert length(Y) == length(ll)
	for p in Combinatorics.permutations(1:n)
		rpi_basis_val += reBreB(Y, k, ll, mm, A)
	end

	return erpiB_val/factorial(n)
end

#allocate_p(L)

#mm = @SVector [-1,1]
#eB_vec = [eB(Y, k, ll, mm, erotc ) for  (i,mm) in enumerate(_mrange1(ll))]
#eB_rot_vec = [eB(Y_rot, k, ll, mm, erotc ) for  (i,mm) in enumerate(_mrange1(ll))]
#print(eBk,"\n")
tol = 1E-10
for k in 1:3
	difference_vec = [Q*eB(Y, k, ll, mm, erotc ) - eB(Y_rot, k, ll, mm, erotc ) for  (i,mm) in enumerate(_mrange1(ll))]
	print("k = ", k, " ", all([norm(dd) < tol for dd in difference_vec]),"\n")
end
	#print([norm(dd) for dd in difference_vec])

for (im, (jm,mm)) in enumerate(Iterators.product(1:3, _mrange1(ll))), (ik, (jk,kk)) in enumerate(Iterators.product(1:3, _mrange1(ll)))
	print(im,", ", ik,"\n")
end

dd  = 3
function re_gramian(A::ERot3DCoeffs{T}, ll::SVector) where {T}
	len = length(_mrange1(ll))*3
	GG = zeros(ComplexF64, len, len) # = Gramian of rotational eqiuvariant span
	# kk = mprime
    for (i, (k,mm)) in enumerate(Iterators.product(1:3, _mrange1(ll))), (ip, (kp,mmp)) in enumerate(Iterators.product(1:3, _mrange1(ll)))
		for mu in _mrange1(ll)
		  	GG[i,ip] += dot(A(ll, mm, mu)[:,k],A(ll, mmp, mu)[:,kp])
		end
	end
	return GG
end

GG = re_gramian(erotc, ll)

"""return coefficients for a rotational invariant tensor basis"""
function re_basis(A::ERot3DCoeffs, ll::SVector{N, Int}) where {N}
	#Mre = collect(Iterators.product(1:3, _mrange1(ll)))
	GG = re_gramian(A, ll)
   	S = svd(GG)
   	rk = rank(GG; rtol =  1e-7)
	Ure = S.U[:, 1:rk]'

	return Diagonal(1. ./ sqrt.(S.S[1:rk])) * Ure
end

re_coeff  = re_basis(erotc, ll)
print("Basis size: ", size(re_coeff))


function eval_re_basis(re_coeff,Y, A::ERot3DCoeffs, ll::SVector{N, Int}) where {N}
	b_size, span_size  = size(re_coeff)
	re_basis_val = zeros(ComplexF64, 3, b_size)
	eB_vec = zeros(ComplexF64, 3, span_size)
	for (ii, (k,mm)) in enumerate(Iterators.product(1:3, _mrange1(ll)))
		eB_vec[:,ii] = eB(Y, k, ll, mm, A)
	end	#print(size(eB_vec))
	for i in 1:b_size
		re_basis_val[:,i] = sum([re_coeff[i,ii] * eB_vec[:,ii] for (ii, (k,mm)) in enumerate(Iterators.product(1:3, _mrange1(ll)))])
	end
	return re_basis_val
end


function rpi_basis(A::Rot3DCoeffs,
						 zz::SVector{N},
						 nn::SVector{N, Int},
						 ll::SVector{N, Int}) where {N}
	Ure = re_basis(A, ll)
	Mre = collect( Iterators.product(1:3, _mrange1(ll)) )   # rows...
	G = _gramian(zz, nn, ll, Ure, Mre)
   	S = svd(G)
   	rk = rank(G; rtol =  1e-7)
	Urpi = S.U[:, 1:rk]'
	return Diagonal(sqrt.(S.S[1:rk])) * Urpi * Ure, Mre
end

function rpi_gramian_old(A::ERot3DCoeffs{T}, ll::SVector) where {T}
	len = length(_mrange1(ll))*3
	n = length(ll)
	GG = zeros(ComplexF64, len, len) # = Gramian of rotational eqiuvariant span
	# kk = mprime
    for (i, (k,mm)) in enumerate(Iterators.product(1:3, _mrange1(ll))), (ip, (kp,mmp)) in enumerate(Iterators.product(1:3, _mrange1(ll)))
		for mu in _mrange1(ll)
			for perm in Combinatorics.permutations(1:n)
		  		GG[i,ip] += dot(A(ll[perm], mm[perm], mu[perm])[:,k],A(ll[perm], mmp[perm], mu[perm])[:,kp])
			end
		end
	end
	return GG
end

function rpi_gramian(zz, nn, ll, Ure, Mre)
   N = length(nn)
   nre = size(Ure, 1)
   @assert size(Ure, 1) == nre
   G = zeros(nre, nre)
   for σ in permutations(1:N)
      if (zz[σ] != zz) || (nn[σ] != nn) || (ll[σ] != ll); continue; end
      for (iU1, mm1) in enumerate(Mre), (iU2, mm2) in enumerate(Mre)
         if mm1[σ] == mm2
            for i1 = 1:nri, i2 = 1:nri
               G[i1, i2] += conj(Ure[i1, iU1]) * Ure[i2, iU2]
            end
         end
      end
   end
   return G
end

function ri_gramian(zz, nn, ll, Uri, Mri)
   N = length(nn)
   nri = size(Uri, 1)
   @assert size(Uri, 1) == nri
   G = zeros(nri, nri)
   for σ in permutations(1:N)
      if (zz[σ] != zz) || (nn[σ] != nn) || (ll[σ] != ll); continue; end
      for (iU1, mm1) in enumerate(Mri), (iU2, mm2) in enumerate(Mri)
         if mm1[σ] == mm2
            for i1 = 1:nri, i2 = 1:nri
               G[i1, i2] += conj(Uri[i1, iU1]) * Uri[i2, iU2]
            end
         end
      end
   end
   return G
end


rpi_basis(A::Rot3DCoeffs, zz, nn, ll) =
			rpi_basis(A, SVector(zz...), SVector(nn...), SVector(ll...))

function rpi_basis(A::Rot3DCoeffs,
						 zz::SVector{N},
						 nn::SVector{N, Int},
						 ll::SVector{N, Int}) where {N}
	Uri = ri_basis(A, ll)
	Mri = collect( _mrange(ll) )   # rows...
	G = _gramian(zz, nn, ll, Uri, Mri)
   	S = svd(G)
   	rk = rank(G; rtol =  1e-7)
	Urpi = S.U[:, 1:rk]'
	return Diagonal(sqrt.(S.S[1:rk])) * Urpi * Uri, Mri
end


function _gramian(zz, nn, ll, Uri, Mri)
   N = length(nn)
   nri = size(Uri, 1)
   @assert size(Uri, 1) == nri
   G = zeros(nri, nri)
   for σ in permutations(1:N)
      if (zz[σ] != zz) || (nn[σ] != nn) || (ll[σ] != ll); continue; end
      for (iU1, mm1) in enumerate(Mri), (iU2, mm2) in enumerate(Mri)
         if mm1[σ] == mm2
            for i1 = 1:nri, i2 = 1:nri
               G[i1, i2] += conj(Uri[i1, iU1]) * Uri[i2, iU2]
            end
         end
      end
   end
   return G
end





function eval_rpi_basis(re_coeff,Y, A::ERot3DCoeffs, ll::SVector{N, Int}) where {N}

	rpi_basis_val = zeros(ComplexF64, 3, size(re_coeff)...)
	n  = length(Y)
	@assert length(Y) == length(ll)
	for p in Combinatorics.permutations(1:n)
		rpi_basis_val += eval_re_basis(re_coeff,Y[p], A, ll)
	end
	re_basis_val/=factorial(n)
	return rpi_basis_val
end



eB_basis  = eval_re_basis(re_coeff, Y, erotc, ll)
eB_basis_rot  = eval_re_basis(re_coeff, Y_rot, erotc, ll)

b_size = size(eB_basis)[2]

difference_vec = [[Q*eB_basis[:,i] - eB_basis_rot[:,i]] for i in 1:b_size]
size(difference_vec)

print("Rotation symmetric: ", all([norm(difference_vec[i]) < tol for i in 1:b_size]),"\n")
n  = length(Y)


p = StatsBase.sample(1:4,4, replace=false)
eB_pi_basis  = eval_re_basis(re_coeff, Y, erotc, ll)
eB_pi_basis_rot  = eval_re_basis(re_coeff, Y_rot, erotc, ll)


difference_vec_perm = [[Q*eB_basis[:,i] - eB_basis_rot[:,i]] for i in 1:b_size]

[ (eval_re_basis(re_coeff, Y[p], erotc, ll) - eB_basis)  for p in Combinatorics.permutations(1:n)]

print("Rotation symmetric: ", all([norm(difference_vec_perm)< tol for i in 1:b_size]))

sample([rng], a, [wv::AbstractWeights], dims::Dims; replace=true, ordered=false)

end


difference_vec_perm = [Y[p]  for p in Combinatorics.permutations(1:n)]
