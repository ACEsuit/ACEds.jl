
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



module Rotations3D

using StaticArrays
using LinearAlgebra: norm, rank, svd, Diagonal, dot
using ACE.SphericalHarmonics: index_y
using Combinatorics: permutations

export ClebschGordan, Rot3DCoeffs, ri_basis, rpi_basis


"""
`ClebschGordan: ` storing precomputed Clebsch-Gordan coefficients; see
`?clebschgordan` for the convention that is use.
"""
struct ClebschGordan{T}
	vals::Dict{Tuple{Int, Int, Int, Int, Int, Int}, T}
end


"""
`ERot3DCoeffs: ` storing recursively precomputed matrix-valued coefficients for a
rotation-equivariant basis. L = 0 equiv invariant, L = 1 equiv equivariant
"""
struct Rot3DCoeffs{T,L}
   vals::Vector{Dict}
   cg::ClebschGordan{T}
end




# -----------------------------------
# iterating over an m collection
# -----------------------------------

_mvec(::CartesianIndex{0}) = SVector(Int(0))

_mvec(mpre::CartesianIndex) = SVector(Tuple(mpre)..., - sum(Tuple(mpre)))

struct MRange{N, T2}
   ll::SVector{N, Int}
   cartrg::T2
   L::Int
end

Base.length(mr::MRange) = sum(_->1, _mrange(mr.ll,mr.L))

_mrange(ll, L) = MRange(ll, Iterators.Stateful(
               filter((x) -> abs(sum(x))<= L, Tuple.(CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)))))
                     ),L)

function Base.iterate(mr::MRange, args...)
	while true
   		if isempty(mr.cartrg)
      		return nothing
   		end
   		mpre = popfirst!(mr.cartrg)
   		return SVector(mpre), nothing
	end
end



# ----------------------------------------------------------------------
#     ClebschGordan code
# ----------------------------------------------------------------------


cg_conditions(j1,m1, j2,m2, J,M) =
	cg_l_condition(j1, j2, J)   &&
	cg_m_condition(m1, m2, M)   &&
	(abs(m1) <= j1) && (abs(m2) <= j2) && (abs(M) <= J)

cg_l_condition(j1, j2, J) = (abs(j1-j2) <= J <= j1 + j2)

cg_m_condition(m1, m2, M) = (M == m1 + m2)


"""
`clebschgordan(j1, m1, j2, m2, J, M, T=Float64)` :

A reference implementation of Clebsch-Gordon coefficients based on

https://hal.inria.fr/hal-01851097/document
Equation (4-6)

This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks, but only precomputation.

The ordering of parameters corresponds to the following convention:
```
clebschgordan(j1, m1, j2, m2, J, M) = C_{j1m1j2m2}^{JM}
```
where
```
   D_{m1k1}^{l1} D_{m2k2}^{l2}}
	=
	∑_j  C_{l1m1l2m2}^{j(m1+m2)} C_{l1k1l2k2}^{j2(k1+k2)} D_{(m1+m2)(k1+k2)}^{j}
```
"""
function clebschgordan(j1, m1, j2, m2, J, M, T=Float64)
	if !cg_conditions(j1, m1, j2, m2, J, M)
		return zero(T)
	end

   N = (2*J+1) *
       factorial(big(j1+m1)) * factorial(big(j1-m1)) *
       factorial(big(j2+m2)) * factorial(big(j2-m2)) *
       factorial(big(J+M)) * factorial(big(J-M)) /
       factorial(big( j1+j2-J)) /
       factorial(big( j1-j2+J)) /
       factorial(big(-j1+j2+J)) /
       factorial(big(j1+j2+J+1))

   G = big(0)
   # 0 ≦ k ≦ j1+j2-J
   # 0 ≤ j1-m1-k ≤ j1-j2+J   <=>   j2-J-m1 ≤ k ≤ j1-m1
   # 0 ≤ j2+m2-k ≤ -j1+j2+J  <=>   j1-J+m2 ≤ k ≤ j2+m2
   lb = (0, j2-J-m1, j1-J+m2)
   ub = (j1+j2-J, j1-m1, j2+m2)
   for k in maximum(lb):minimum(ub)
      bk = big(k)
      G += (-1)^k *
           binomial(big( j1+j2-J), big(k)) *
           binomial(big( j1-j2+J), big(j1-m1-k)) *
           binomial(big(-j1+j2+J), big(j2+m2-k))
   end

   return T(sqrt(N) * G)
end


ClebschGordan(T=Float64) =
	ClebschGordan{T}(Dict{Tuple{Int,Int,Int,Int,Int,Int}, T}())

_cg_key(j1, m1, j2, m2, J, M) = (j1, m1, j2, m2, J, M)
	# Int.((index_y(j1,m1), index_y(j2,m2), index_y(J,M)))

function (cg::ClebschGordan{T})(j1, m1, j2, m2, J, M) where {T}
	if !cg_conditions(j1,m1, j2,m2, J,M)
		return zero(T)
	end
	key = _cg_key(j1, m1, j2, m2, J, M)
	if haskey(cg.vals, key)
		return cg.vals[key]
	end
	val = clebschgordan(j1, m1, j2, m2, J, M, T)
	cg.vals[key] = val
	return val
end

# ----------------------------------------------------------------------
#     IRot3DCoeffs code: generalized cg coefficients
#
#  Note: in this section kk is a tuple of m-values, it is not
#        related to the k index in the 1-p basis (or radial basis)
# ----------------------------------------------------------------------


function get0(L::Int,T=Float64)
	if L== 0
		return T(0)
	elseif L == 1
		return @SArray zeros(Complex{T},3,3)
		#SMatrix{3, 3, Complex{T}, 9}(0, 0, 0, 0, 0, 0, 0, 0, 0)
	else
		#For L== 2 @SArray zeros(Complex{T},3,3,3)
		ErrorException("Only types for L in {0,1} implemented")
	end
end
dicttype(A::Rot3DCoeffs,N::Integer) = dicttype(A::Rot3DCoeffs,Val(N))

dicttype(A::Rot3DCoeffs{T,L},::Val{N}) where {T,L,N} =
   Dict{Tuple{SVector{N,Int}, SVector{N,Int}, SVector{N,Int}}, typeof(get0(L,T))}

Rot3DCoeffs(L,T=Float64) = Rot3DCoeffs{T,L}(Dict[], ClebschGordan(T))


function get_vals(A::Rot3DCoeffs, valN::Val{N}) where {N}
	if length(A.vals) < N
		for n = length(A.vals)+1:N
			push!(A.vals, dicttype(A,n)())
		end
	end
   	return A.vals[N]::dicttype(A,valN)
end

_key(ll::StaticVector{N}, mm::StaticVector{N}, kk::StaticVector{N}) where {N} =
      (SVector{N, Int}(ll), SVector{N, Int}(mm), SVector{N, Int}(kk))


  function (A::Rot3DCoeffs{T,L})(ll::StaticVector{N},
                              mm::StaticVector{N},
                              kk::StaticVector{N}) where {T,L, N}
     if       abs(sum(mm)) > L ||
              abs(sum(kk)) > L ||
              !all(abs.(mm) .<= ll) ||
              !all(abs.(kk) .<= ll)
        return get0(L,T)
     end
     vals = get_vals(A, Val(N))  # this should infer the type!
     key = _key(ll, mm, kk)
     if haskey(vals, key)
        val  = vals[key]
     else
        val = _compute_val(A, key...)
        vals[key] = val
     end
     return val
  end


  function (A::Rot3DCoeffs{T,0})(ll::StaticVector{1},
                              mm::StaticVector{1},
                              kk::StaticVector{1}) where {T}
     if ll[1] == mm[1] == kk[1] == 0
        return T(8 * pi^2)
     else
        return get0(0,T)
     end
  end

  function (A::Rot3DCoeffs{T,1})(ll::StaticVector{1},
                              mm::StaticVector{1},
                              kk::StaticVector{1}) where {T}
     if ll[1] == 1 && abs(mm[1]) <= 1 && abs(kk[1]) <= 1
        return  rmatrices[(mm[1],kk[1])]
     else
        return get0(1,T)
     end
  end

#function get0val(m,k,L)

#end
rmatrices = Dict(
    (-1,-1) => SMatrix{3, 3, ComplexF64, 9}(1/6, 1im/6, 0, -1im/6, 1/6, 0, 0, 0, 0),
    (-1,0) => SMatrix{3, 3, ComplexF64, 9}(0, 0, 0, 0, 0, 0, 1/(3*sqrt(2)), 1im/(3*sqrt(2)), 0),
    (-1,1) => SMatrix{3, 3, ComplexF64, 9}(-1/6, -1im/6, 0, -1im/6, 1/6, 0, 0, 0, 0),
    (0,-1) => SMatrix{3, 3, ComplexF64, 9}(0, 0, 1/(3*sqrt(2)), 0, 0, -1im/(3*sqrt(2)), 0, 0, 0),
    (0,0) => SMatrix{3, 3, ComplexF64, 9}(0, 0, 0, 0, 0, 0, 0, 0, 1/3),
    (0,1) => SMatrix{3, 3, ComplexF64, 9}(0, 0, -1/(3*sqrt(2)), 0, 0, -1im/(3*sqrt(2)), 0, 0, 0),
    (1,-1) => SMatrix{3, 3, ComplexF64, 9}(-1/6, 1im/6, 0, 1im/6, 1/6, 0, 0, 0, 0),
    (1,0) => SMatrix{3, 3, ComplexF64, 9}(0, 0, 0, 0, 0, 0, -1/(3*sqrt(2)), 1im/(3*sqrt(2)), 0),
    (1,1) => SMatrix{3, 3, ComplexF64, 9}(1/6, -1im/6, 0, 1im/6, 1/6, 0, 0, 0, 0)
    )

function (A::Rot3DCoeffs{T,L})(ll::StaticVector{1},
							 mm::StaticVector{1},
							 kk::StaticVector{1}) where {T,L}
	ErrorException("Not implemented for L = " + string(L))
end


function _compute_val(A::Rot3DCoeffs{T,L}, ll::StaticVector{N},
                                        mm::StaticVector{N},
                                        kk::StaticVector{N}) where {T,L,N}

   val = get0(L,T)
   llp = ll[1:N-2]
   mmp = mm[1:N-2]
   kkp = kk[1:N-2]
   for j = abs(ll[N-1]-ll[N]):(ll[N-1]+ll[N])
      if abs(kk[N-1]+kk[N]) > j || abs(mm[N-1]+mm[N]) > j
         continue
      end
		cgk = try
			A.cg(ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
		catch
			@show (ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
			get0(L,T)
		end
		cgm = A.cg(ll[N-1], mm[N-1], ll[N], mm[N], j, mm[N-1]+mm[N])
		if cgk * cgm  != 0
			val += cgk * cgm * A( SVector(llp..., j),
								       SVector(mmp..., mm[N-1]+mm[N]),
								       SVector(kkp..., kk[N-1]+kk[N]) )
		end
   end
   return val
end



function re_basis(A::Rot3DCoeffs{T,0}, ll::SVector{N, Int}; ordered=false) where {T,N}
	"""computes coefficients for a rotational 0-equinvariant (invariant) tensor basis"""
	CC = compute_Al(A, ll, Val(ordered))
	svdC = svd(CC)
	rk = rank(Diagonal(svdC.S))
	return svdC.U[:, 1:rk]'
end

function compute_Al(A::Rot3DCoeffs{T,0}, ll::SVector, ::Val{false}) where {T}
	len = length(_mrange(ll,0))
   CC = zeros(T, len, len)
   for (im, mm) in enumerate(_mrange(ll,0)), (ik, kk) in enumerate(_mrange(ll,0))
      CC[ik, im] = A(ll, mm, kk)
   end
   return CC
end



function re_basis(A::Rot3DCoeffs{T,L}, ll::SVector{N, Int}) where {T,L,N}
	"""return coefficients for a rotational 1-equinvariant (covariant) tensor basis"""
	#Mre = collect(Iterators.product(1:3, _mrange(ll,1)))
	GG = _gramian(A, ll)
   	S = svd(GG)
   	rk = rank(GG; rtol =  1e-7)
	Ure = S.U[:, 1:rk]'

	return Diagonal(1. ./ sqrt.(S.S[1:rk])) * Ure
end

imag_tol = 1e-15

function _gramian(A::Rot3DCoeffs{T,1}, ll::SVector) where {T}
	len = length(_mrange(ll,1))*3
	GG = zeros(ComplexF64, len, len) # = Gramian of rotational eqiuvariant span
	# kk = mprime
    for (i, (k,mm)) in enumerate(Iterators.product(1:3, _mrange(ll,1))), (ip, (kp,mmp)) in enumerate(Iterators.product(1:3, _mrange(ll,1)))
		for mu in _mrange(ll,1)
		  	GG[i,ip] += dot(A(ll, mm, mu)[:,k],A(ll, mmp, mu)[:,kp])
		end
	end
	@assert all(abs.(imag(GG)) .<= imag_tol)
    return real.(GG)
end


rpi_basis(A::Rot3DCoeffs, zz, nn, ll) =
			rpi_basis(A, SVector(zz...), SVector(nn...), SVector(ll...))

function rpi_basis(A::Rot3DCoeffs{T,L},
						 zz::SVector{N},
						 nn::SVector{N, Int},
						 ll::SVector{N, Int}) where {T,L,N}
	Ure = re_basis(A, ll)
	Mre = collect( _mrange(ll, L))   # rows...
	G = _gramian(zz, nn, ll, Ure, Mre)
	#print("G type :", typeof(G),"\n")
   	S = svd(G)
   	rk = rank(G; rtol =  1e-7)
	Urpi = S.U[:, 1:rk]'
	#print("Urpi type :", typeof(Urpi),"\n")
	return Diagonal(sqrt.(S.S[1:rk])) * Urpi * Ure, Mre
end

function rpi_basis(A::Rot3DCoeffs{T,1},
						 zz::SVector{N},
						 nn::SVector{N, Int},
						 ll::SVector{N, Int}) where {T,N}
	Mre = collect( _mrange(ll, 1))   # rows...
	G = _gramian(A, zz, nn, ll, Mre)
	#print("G type :", typeof(G),"\n")
   	S = svd(G)
   	rk = rank(G; rtol =  1e-7)
	Urepi = Diagonal(1/sqrt.(S.S[1:rk])) * S.U[:, 1:rk]'
	Utilde = zeros(SArray{Tuple{3},Complex{T},1,3}, rk, length(Mre))
	for α in 1:rk
		for (imu, mu) in enumerate(Mre)
			for (i, (k,mm)) in enumerate(Iterators.product(1:3, Mre))
				Utilde[α,imu] += Urepi[α,i] * A(ll,mu,mm)[:,k]
			end
		end
	end
	#print("Urpi type :", typeof(Urpi),"\n")
	return Utilde, Mre
end


function _gramian(A::Rot3DCoeffs{T,1}, zz, nn, ll::SVector{N}, Mre) where{T,N}
  	len = length(Mre)*3
	GG = zeros(Complex{T}, len, len)

	for (i1, (k1,mm1)) in enumerate(Iterators.product(1:3, _mrange(ll, 1))), (i2, (k2,mm2)) in enumerate(Iterators.product(1:3, _mrange(ll, 1)))
		for σ_ in permutations(1:N)
			σ = SVector{N,Int}(σ_)
	      	if ( zz[σ] != zz) || (nn[σ] != nn) || (ll[σ] != ll); continue; end
			for mu1 in  _mrange(ll, 1), mu2 in _mrange(ll, 1)
	        	if mu1[σ] == mu2
					#print(ll,"\n")
					#A(ll, mm2, mu2)
					#@show typeof(mu2)
					#@show typeof(ll[σ])
					#GG[i1, i2] += sum(A(ll[σ], mu1[σ],mm1[σ])[k1,:] .* A(ll,mu2,mm2)[:,k2])
	               	#GG[i1, i2] += dot(A(ll[σ], mm1[σ], mu1[σ])[:,k1],A(ll, mm2, mu2)[:,k2]) #old verions correct?
					#GG[i1, i2] += sum( conj(A(ll,mu2,mm2)[:,k2]).* A(ll[σ], mu1[σ],mm1[σ])[:,k1]  )
					# works GG[i1, i2] += dot(A(ll, mm2, mu2)[:,k2],A(ll[σ], mm1[σ], mu1[σ])[:,k1])
					#GG[i1, i2] += dot(A(ll,mu2,mm2)[:,k2],A(ll[σ],mu1[σ],mm1[σ])[:,k1])
					GG[i1, i2] += dot(A(ll, mu2, mm2)[:,k2],A(ll[σ], mu1[σ], mm1[σ])[:,k1])
	         	end
	      	end
	   	end
	end
   	return GG
	#@assert all(abs.(imag(GG)) .<= imag_tol)
	#return real.(GG)
end

function _gramian(zz, nn, ll, Uri, Mri)
   N = length(nn)
   nri = size(Uri, 1)
   @assert size(Uri, 1) == nri
   G = zeros(Float64,nri, nri)

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
   #@assert all(abs.(imag(G)) .<= imag_tol)
   #return real.(G)
end


end
