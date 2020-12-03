using ACEds

using StaticArrays
using ACEds.equivRotations3D

cg=ClebschGordan()

cg(5, -4, 5, 4, 2, 0)

ll = @SVector [1,1,3,2]
mm = @SVector [1,-1,1,-1]
kk = @SVector [1,-1,1,-1]
irotc = ACEds.equivRotations3D.IRot3DCoeffs(Float64)
irotc(ll,mm,kk)


#a=3
#a::Int64

#dicttype(::Val{N}) where {N} =
#   Dict{Tuple{SVector{N,Int}, SVector{N,Int}, SVector{N,Int}}, Float64}

#SMatrix{3,3,Float64}
#


#SMatrix{3,3,Float64,9}
erotc = ACEds.equivRotations3D.ERot3DCoeffs(Float64)
erotc(ll,mm,kk)
