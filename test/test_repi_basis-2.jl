
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
N = 4
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

using ReferenceFrameRotations: angle_to_dcm

Q = angle_to_dcm(.5,.4,-.3, :ZYX)

Nat = 10
Rs, Zs, z0 = rand_nhd(Nat, Pr, :X)

Rs_rot = deepcopy(Rs)
for i = 1:Nat
    Rs_rot[i] = Q*Rs_rot[i]
end

B = evaluate(rpibasis, Rs, Zs, z0)

#print(Q*real.(B[1]), "     ", B_rot[1],"\n")
N_basis = length(B)

@info("check for equivariance")

B_rot = evaluate(rpibasis, Rs_rot, Zs, z0)
#for i in 1:N_basis
#    print(norm(Q*B[i]-B_rot[i])<1E-5,"\n")
#    print(Q*B[i], "     ", B_rot[i],"\n")
#end
@test all([ norm(Q*B[i]-B_rot[i])<1E-5 for i in 1:length(B)])


@info("check for linear independence")

using LinearAlgebra: rank

N_eval = Integer(ceil(N_basis/3)) # may pass
N_eval = Integer(floor(N_basis/3)) # should not pass
N_eval = N_basis  # should pass

B_array = zeros(3*N_eval,N_basis)
atol = 1E-10
for k in 1:N_eval
    Rs, Zs, z0 = rand_nhd(Nat, Pr, :X)
    B = evaluate(rpibasis, Rs, Zs, z0)
    for i in 1:N_basis
        B_array[(3*(k-1)+1):(3*(k-1)+3), i] = B[i]
    end
end

print("Rank: ", rank(B_array,atol=atol),"\n")
@test rank(B_array,atol=atol) == N_basis
