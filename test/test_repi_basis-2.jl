
#using ACEds.RPI: SparsePSHDegree, BasicPSH1pBasis


using ACE
using ACE: numz
#import ACE: standardevaluator, graphevaluator
using ACEds
using ACEds.RPI: RPIBasis, SparsePSHDegree, BasicPSH1pBasis, PIBasis, Rot3DCoeffs, _rpi_A2B_matrix
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate_ed
using JuLIP.MLIPs: combine
using StaticArrays

#---


@info("Basic test of RPIBasis construction and evaluation")
maxdeg = 10
N = 3
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SparsePSHDegree()
P1 = BasicPSH1pBasis(Pr; species = :X, D = D)


#---
pibasis = PIBasis(P1, N, D, maxdeg)
#rpibasis = RPIBasis(P1, N, D, maxdeg)
rpibasis = RPIBasis(P1, N, D, maxdeg,L=1)
#---



#---
@info("check equivariance")

using ReferenceFrameRotations: angle_to_dcm

Q = angle_to_dcm(.5,.4,-.3, :ZYX)

Nat = 5
Rs, Zs, z0 = rand_nhd(Nat, Pr, :X)

Rs_rot = deepcopy(Rs)
for i = 1:Nat
    Rs_rot[i] = Q*Rs_rot[i]
end

B = evaluate(rpibasis, Rs, Zs, z0)
B_rot = evaluate(rpibasis, Rs_rot, Zs, z0)


#print(Q*real.(B[1]), "     ", B_rot[1],"\n")
for i in 1:Nat
    print(norm(Q*B[i]-B_rot[i])<1E-5,"\n")
    print(Q*B[i], "     ", B_rot[i],"\n")
end
