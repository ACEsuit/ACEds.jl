using ACEds
using ACE
using ACE.RPI.Rotations3D
#using ACEds.Rot3D2
using StaticArrays

ll = @SVector [1,1]
L=1
#a= ACEds.Rot3D2._mrange(ll, L)

#for k in ACEds.Rot3D2._mrange(ll, 1)
#    print(k,"\n")
#end

a = @SVector [1,1.,1]

len=4
A = zeros(SArray{Tuple{3},Float64,1,3}, len, len)
for i=1:3
    A[1,2] += @SVector [1,1.,1]
end

A[1,2]
#ACEds.Rot3D2.get0(1)

#ACEds.Rot3D2.Rot3DCoeffs(0)
